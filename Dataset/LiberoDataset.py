import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset

class LiberoDataset(Dataset):
    def __init__(self, data_dir, selected_tasks, num_frames=4, use_features = False):
        """
        Args:
            data_dir (str): './processed_data', the generic path of all the data for all the tasks
            selected_tasks (list): ['libero_10', 'libero_goal', ...] a list of all the tasks chosen
            num_frames (int): the size of the window of frames passed to the model
            stride (int): how much frames we skip from a window to another
            use_features (bool): if True, we'll use the dataset preprocessed through vision and language backbones with extracted features
        """

        super(LiberoDataset, self).__init__()

        self.file_paths = []   # a list of all the file paths (Ex. ./processed_data/libero_10/task_0_demo_0.pt)
        self.T_window = num_frames // 2
        #self.use_features = use_feature
        # Each step is divided in 16x16 patches, then each instant between frames are 256 tokens
        self.num_patches = 256
        
        for task in selected_tasks:
            task_path = os.path.join(data_dir, task, "*.pt") # the generic path of the type "./processed_data/libero_10/*.pt"
            # with glob.glob we create a list of path ['.../task_0_demo_0', '.../task_0_demo_1', ...], one list for all the tasks
            # with extend we do not append list of lists but create a unique list for all the tasks paths
            files_found = sorted(glob.glob(task_path))
            self.file_paths.extend(files_found)
            
        self.window_indices = []

        for data_idx, path in enumerate(self.file_paths):
            data = torch.load(path, map_location='cpu', weights_only=True)
            N_tokens = data['z_obs'].shape[1] #if use_features else data['frames'].shape[0] # number of steps in that demo or number of tubelets if feature extracted
            T_demo = N_tokens // self.num_patches
            #if self.use_features:
            for start_idx in range(0, T_demo - self.T_window +1):
                self.window_indices.append((data_idx,start_idx))
            #else:
                #if self.T >= self.window_size:
                    #for start_idx in range(0, T - num_frames+1, self.stride):
                        #self.window_indices.append((data_idx, start_idx))

    def __len__(self):
        return len(self.window_indices) # the length of the data are all the windows that the model will see in training
    
    def __getitem__(self, index):
        data_idx, start_idx = self.window_indices[index] # keep one tuple Ex. (0, 5) (demo 0, start window 5)
        demo_path = self.file_paths[data_idx]  # take the corresponding demo path => './processed_data/libero_10/task_0_demo_0.pt'
        
        #demo = load_cached_demo(demo_path)    # Loading pytorch data
        demo = torch.load(demo_path, map_location='cpu')

        #if self.use_features:

        # Extracting the entire z_obs vision features tokens
        z_obs = demo['z_obs'].float().squeeze(0)

        # Computing the token of start and the token of end
        token_start_pos = start_idx*self.num_patches
        token_end_pos = start_idx*self.num_patches + self.T_window*self.num_patches

        # The resulting z_obs to get
        z_obs_window = z_obs[token_start_pos:token_end_pos]
        
        # Extracting the z_text text features
        z_text = demo['z_text'].float().squeeze(0)

        # All the actions and joints states
        actions = demo['actions'].float()
        joint_states = demo['joint_states'].float()
        
        end_idx = start_idx+self.T_window

        joint_input = joint_states[start_idx:end_idx]
        action_seq_target = actions[start_idx:end_idx]

        
        return {'vision_input': z_obs_window,
                'text_input': z_text,
                'joint_input': joint_input,
                'action_seq_target': action_seq_target,
                } 
        '''
        else:
            # Computing the end_idx index
            end_idx = start_idx + self.window_size

            # Extracting the frames, normalizing them and permuting to put channels before for pytorch
            win_frames = demo['frames'][start_idx:end_idx].float() / 255.0 
            win_frames = win_frames.permute(0,3,1,2)

            text_instruction = demo['text_instruction']

            # For the actions we take 1 action each 2 frames because of the VJEPA Model work in tubelets of 2 frames
            # Ex. for 16 frames V JEPA Encoder will produce an output of [1, 8, 256, 1408], it means 256*8 vectors of 1408 dimensions
            actions = demo['actions'][start_idx:end_idx]
            action_seq_target = actions[1::2] # it extract the 1, 3, 5 ,...
            
            return {'vision_input': win_frames,
                    'text_input': text_instruction,
                    'action_seq_target': action_seq_target,
                    } 
            '''
