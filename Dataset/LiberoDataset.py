import os
from torch.utils.data import Dataset

class LiberoDataset(Dataset):
    def __init__(self, data_dir, selected_tasks):
        super(LiberoDataset, self).__init__()

        self.file_paths = []    # a list where to append all the demo file .pt to do torch.load
        self.data_dir_list

    def __len__(self):
        return len()
    
    def __getitem__(self, index):
        return index
