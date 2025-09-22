import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]