import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas

class InfluencerBrandDataset(Dataset):
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        self.data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        self.data.fillna(0, inplace=True)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data.iloc[idx].to_numpy()
        return sample
        
    def __len__(self):
        return len(self.data)
    
def influencer_brand_dataloader(dataset, batch_size):
    return DataLoader(dataset, batch_size)
