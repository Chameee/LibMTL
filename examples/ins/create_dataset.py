import pandas as pd
from torch.utils.data import DataLoader, Dataset
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import pandas

class InfluencerBrandDataset(Dataset):
    def __init__(self, csv_path, task, mode):
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
    
def influencer_brand_dataloader(dataset, batchsize, root_path):
    data_loader = {}
    iter_data_loader = {}
    for k, d in enumerate(tasks):
        data_loader[d] = {}
        iter_data_loader[d] = {}
        for mode in ['train', 'val', 'test']:
            shuffle = True if mode == 'train' else False
            txt_dataset = InfluencerBrandDataset(dataset, root_path, d, mode)
#             print(d, mode, len(txt_dataset))
            data_loader[d][mode] = DataLoader(txt_dataset, 
                                              num_workers=0, 
                                              pin_memory=True, 
                                              batch_size=batchsize, 
                                              shuffle=shuffle)
            iter_data_loader[d][mode] = iter(data_loader[d][mode])
    return data_loader, iter_data_loader
