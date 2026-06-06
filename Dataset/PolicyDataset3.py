import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class PolicyDataset(Dataset):
    def __init__(self, datasets, task_ids, num_frames, action_chunk_size, full_load_ram = False):

        super(PolicyDataset, self).__init__()

        self.file_paths = []   # a list of all the file paths (Ex. ./resampled_data/libero_10/task_0_demo_0.pt)
        self.datasets = datasets
        self.task_ids = task_ids
        self.full_load_ram = full_load_ram
        self.num_frames = num_frames
        self.action_chunk_size = action_chunk_size
        
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

            steps = data['frames'].shape[0]
            
            for start_idx in range(0, steps):
                self.window_indices.append((data_idx,start_idx))

        self.window_indices.sort()
            

    def __len__(self):
        return len(self.window_indices) # the length of the data are all the windows that the model will see in training
    
    def __getitem__(self, index):

        data_idx, current_idx = self.window_indices[index] 

        if self.full_load_ram:
            demo = self.loaded_demos[data_idx]
        else:
            demo_path = self.file_paths[data_idx]
            demo = torch.load(demo_path, map_location='cpu', weights_only=False)
        
        frames = demo['frames']        
        actions = demo['actions'].float()      
        joint_states = demo['joint_states'].float()

        total_steps = frames.shape[0]

        start_frame_idx = current_idx - self.num_frames + 1
        
        if start_frame_idx >= 0:
            vision_input = frames[start_frame_idx : current_idx + 1]
        else:
            available_frames = frames[0 : current_idx + 1]
            pad_size = self.num_frames - available_frames.shape[0]
            
            first_frame = frames[0:1] 
            
            pad_frames = np.repeat(first_frame, pad_size, axis=0)
            vision_input = np.concatenate([pad_frames, available_frames], axis=0)

        next_idx = current_idx + 1
        if next_idx < total_steps:
            vision_target = frames[next_idx : next_idx + 1]
        else:
            vision_target = frames[-1:]

        vision_input = torch.from_numpy(vision_input).byte().clone()
        vision_target = torch.from_numpy(vision_target).byte().clone()

        available_actions = actions[current_idx : current_idx + self.action_chunk_size]
        num_extracted = available_actions.shape[0]

        if num_extracted == self.action_chunk_size:
            action_output = available_actions
        elif num_extracted > 0:
            pad_size = self.action_chunk_size - num_extracted
            last_valid_action = available_actions[-1:] 
            pad_actions = last_valid_action.repeat(pad_size, 1)
            action_output = torch.cat([available_actions, pad_actions], dim=0)
        else:
            last_absolute_action = actions[-1:] 
            action_output = last_absolute_action.repeat(self.action_chunk_size, 1)
        
        action_output = action_output[:self.action_chunk_size]

        joint_input = joint_states[current_idx]
        
        text_instruction = demo['text_instruction']
 
        return {
            'vision_input': vision_input,     
            'text_input': text_instruction,   
            'joint_input': joint_input,       
            'action_seq_target': action_output,
            'vision_target': vision_target
        }

