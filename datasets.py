import torch
from torch.utils.data import Dataset

class DatasetWithIndices(Dataset):
    def __init__(self, dataset, input_normalize_sym):
        self.dataset = dataset

        self.input_normalize_sym = input_normalize_sym

    def __getitem__(self, index):
        data, target = self.dataset[index]
        if self.input_normalize_sym:
            data = (data - .5) * 2
        return data, target, index

    def __len__(self):
        return len(self.dataset)
