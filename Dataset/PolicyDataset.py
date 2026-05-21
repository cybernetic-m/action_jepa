import torch
import glob
import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

class PolicyDataset(Dataset):
    def __init__(self, datasets, task_ids):

        super(PolicyDataset, self).__init__()

        self.file_paths = []   # a list of all the file paths (Ex. ./resampled_data/libero_10/task_0_demo_0.pt)
        self.datasets = datasets
        self.task_ids = task_ids
        
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

        for data_idx, path in enumerate(tqdm(self.file_paths, desc="Processing dataset files")):
            data = torch.load(path, map_location='cpu', mmap=True, weights_only=False)

            #self.all_actions.append(data['actions'].float())

            steps = data['frames'].shape[0] 
            
            for start_idx in range(0, steps-1):
                self.window_indices.append((data_idx,start_idx))

        #tensor_actions = torch.cat(self.all_actions, dim=0)
        #self.actions_min = tensor_actions.min(dim=0)[0]
        #self.actions_max = tensor_actions.max(dim=0)[0]

        #stats = {'min': self.actions_min.numpy(), 'max': self.actions_max.numpy()}
        #np.save(f'{datasets}_action_stats.npy', stats)
        #print(f"Stats saved in './Dataset' as {datasets}_action_stats.npy! Min: {self.actions_min}, Max: {self.actions_max}")
        self.window_indices.sort()
            

    def __len__(self):
        return len(self.window_indices) # the length of the data are all the windows that the model will see in training
    
    def __getitem__(self, index):

        data_idx, start_idx = self.window_indices[index] 
        demo_path = self.file_paths[data_idx]

        demo = torch.load(demo_path, map_location='cpu', weights_only=False)
        
        frames = demo['frames']
        vision_input = frames[start_idx:start_idx+2]
        
        text_instruction = demo['text_instruction']
        
        actions = demo['actions'].float()
        action_output = actions[start_idx]
        #action_output = 2 * (action_output - self.actions_min) / (self.actions_max - self.actions_min + 1e-8) - 1   # Normalizing action
        
        joint_states = demo['joint_states'].float()
        joint_input = joint_states[start_idx]
 
        return {'vision_input': vision_input,
                'text_input': text_instruction,
                'joint_input': joint_input,
                'action_seq_target': action_output,
                } 

