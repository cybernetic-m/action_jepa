import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset

class LiberoDataset(Dataset):
    def __init__(self, data_dir, selected_tasks, window_size=16, stride=1):
        """
        Args:
            data_dir (str): './processed_data', the generic path of all the data for all the tasks
            selected_tasks (list): ['libero_10', 'libero_goal', ...] a list of all the tasks chosen
            window_size (int): how much frames are passed to the model
            stride (int): how much frames we skip from a window to another
        """

        super(LiberoDataset, self).__init__()

        self.file_paths = []   # a list of all the file paths (Ex. ./processed_data/libero_10/task_0_demo_0.pt)
        self.window_size = window_size
        self.stride = min(stride, window_size) # to avoid gaps in training, if you put stride > window_size, we keep non-overlapping windows with stride = window_size

        for task in selected_tasks:
            task_path = os.path.join(data_dir, task, "*.pt") # the generic path of the type "./processed_data/libero_10/*.pt"
            # with glob.glob we create a list of path ['.../task_0_demo_0', '.../task_0_demo_1', ...], one list for all the tasks
            # with extend we do not append list of lists but create a unique list for all the tasks paths
            files_found = sorted(glob.glob(task_path))
            self.file_paths.extend(files_found)
            

        # We create a list of the type [(0, 1), (0,2) ,.... (0, num_steps - window_size)] of the type (demo, start_window_frame)
        # if the stride is 1, we'll have for each demo a number of windows corresponding to the number of steps - 1 in that demo
        # Ex. 2 demo: task_0_demo_0 (227 steps) and task_0_demo_1 (100 steps)
        # The list is: [(0,1), (0,2), ..., (0,227-16), (1,0), (1,1), ..., (1,100-16)] 
        # When the getitem is called, we'll take the file corresponding to demo, starting from start_window_frame taking data in start_window_frame:start_window_frame+window_size 
        # With the stride > 1, we'll take windows shifted of the stride values
        # Ex. stride=2 => [(0,1), (0,3), ..., (0,227-16)]
        # We'll have in this case less windows => int((227-16)/2)+1
        self.window_indices = []

        for data_idx, path in enumerate(self.file_paths):
            data = torch.load(path, map_location='cpu', weights_only=True)
            T = data['actions'].shape[0] # number of steps in that demo
            if T >= self.window_size:
                for start_t in range(0, T - window_size+1, self.stride):
                    self.window_indices.append((data_idx, start_t))



    def __len__(self):
        return len(self.window_indices) # the length of the data are all the windows that the model will see in training
    
    def __getitem__(self, index):
        data_idx, start_t = self.window_indices[index] # keep one tuple Ex. (0, 5) (demo 0, start window 5)
        demo_path = self.file_paths[data_idx]  # take the corresponding demo path => './processed_data/libero_10/task_0_demo_0.pt'
        demo = torch.load(demo_path, map_location='cpu')    # Loading pytorch data

        # Computing the end_t index
        end_t = start_t + self.window_size

        # Extractin the frames, normalizing them and permuting to put channels before for pytorch
        win_frames = demo['frames'][start_t:end_t].float() / 255.0 
        win_frames = win_frames.permute(0,3,1,2)

        text_instruction = demo['text_instruction']

        # For the actions we take 1 action each 2 frames because of the VJEPA Model work in tubelets of 2 frames
        # Ex. for 16 frames V JEPA Encoder will produce an output of [1, 8, 256, 1408], it means 256*8 vectors of 1408 dimensions
        actions = demo['actions'][start_t:end_t]
        actor_action_seq_target = actions[1::2] # it extract the 1, 3, 5 ,...

        # For the refiner only the last action
        refiner_action_target = actions[-1].unsqueeze(0)
        
        return {'frames': win_frames,
                'text_instruction': text_instruction,
                'actor_action_seq_target': actor_action_seq_target,
                'refiner_action_target': refiner_action_target
                } 
