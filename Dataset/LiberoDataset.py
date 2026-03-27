import os
from torch.utils.data import Dataset

class LiberoDataset(Dataset):
    def __init__(self, data_dir):
        super(LiberoDataset, self).__init__()

        self.data_dir
