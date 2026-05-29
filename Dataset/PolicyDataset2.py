import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class PolicyDataset(Dataset):
    def __init__(self, datasets, task_ids, num_frames, full_load_ram = False):

        super(PolicyDataset, self).__init__()

        self.file_paths = []   # a list of all the file paths (Ex. ./resampled_data/libero_10/task_0_demo_0.pt)
        self.datasets = datasets
        self.task_ids = task_ids
        self.full_load_ram = full_load_ram
        self.num_frames = num_frames
        self.T = self.num_frames // 2
        
        for dataset in self.datasets:
            for task_id in self.task_ids:
                
                task_path = os.path.join('./resampled_data', dataset, str(task_id), 'data', "*.pt") # the generic path of the type "./processed_data/libero_10/*.pt"
                # with glob.glob we create a list of path ['.../task_0_demo_0', '.../task_0_demo_1', ...], one list for all the tasks
                # with extend we do not append list of lists but create a unique list for all the tasks paths
                files_found = sorted(glob.glob(task_path))
                self.file_paths.extend(files_found)
        
        self.file_paths.sort()
        self.all_actions = []
        self.window_indices = []
        
        if self.full_load_ram:
            self.loaded_demos = []

        for data_idx, path in enumerate(tqdm(self.file_paths, desc="Loading Dataset")):
            data = torch.load(path, map_location='cpu', weights_only=False)

            if self.full_load_ram:
                self.loaded_demos.append(data)

            #self.all_actions.append(data['actions'].float())

            steps = data['frames'].shape[0] 
            
            for start_idx in range(0, steps):
                self.window_indices.append((data_idx,start_idx))

        self.window_indices.sort()
            

    def __len__(self):
        return len(self.window_indices) # the length of the data are all the windows that the model will see in training
    
    def __getitem__(self, index):

        data_idx, start_idx = self.window_indices[index]

        if self.full_load_ram:
            demo = self.loaded_demos[data_idx]
        else:
            demo_path = self.file_paths[data_idx]
            demo = torch.load(demo_path, map_location='cpu', weights_only=False)
        
        frames = demo['frames']
        total_steps = frames.shape[0]
        end_idx = start_idx + self.num_frames

        if end_idx <= total_steps:
            vision_input = frames[start_idx:end_idx]
        else:
            available_frames = frames[start_idx:]
            pad_size = self.num_frames - available_frames.shape[0]
            
            if pad_size > 0:
                last_frame = available_frames[-1:]
                pad_frames = np.repeat(last_frame, pad_size, axis=0)
                vision_input = np.concatenate([available_frames, pad_frames], axis=0)
            else:
                vision_input = available_frames[:self.num_frames]
        vision_input = torch.from_numpy(vision_input).byte()
        
        text_instruction = demo['text_instruction']
        
        actions = demo['actions'].float()
        end_action_idx = start_idx + self.T
        
        if end_action_idx <= total_steps:
            action_output = actions[start_idx:end_action_idx]
        else:
            available_actions = actions[start_idx:]
            pad_size = self.T - available_actions.shape[0]
            
            if pad_size > 0:
                last_action = available_actions[-1:]
                pad_actions = last_action.repeat(pad_size, 1)
                action_output = torch.cat([available_actions, pad_actions], dim=0)
            else:
                action_output = available_actions[:self.T]
        
        joint_states = demo['joint_states'].float()
        joint_input = joint_states[start_idx]
 
        return {'vision_input': vision_input,
                'text_input': text_instruction,
                'joint_input': joint_input,
                'action_seq_target': action_output,
                } 

