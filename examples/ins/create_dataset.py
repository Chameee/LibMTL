import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file, low_memory=False)
        self.data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        self.data.fillna(0,inplace=True)
        self.x = self.data[['status', 'country_x', 'instagram_followers', 'macromicro', 
      'listedby','decision_media_gained', 'decision_media_total',
       'decision_followers_gained', 'decision_followers_total',
       'decision_following_gained', 'decision_following_total',
      'aftercamp_media_gained', 'aftercamp_media_total',
       'aftercamp_followers_gained', 'aftercamp_followers_total',
       'aftercamp_following_gained', 'aftercamp_following_total',
      'follower_num'
     ]]
        self.y = self.data[['accept']]

        self.category_feature_names = ['status', 'country_x', 'macromicro', 'listedby']
        self.numerical_feature_names = ['decision_media_gained', 'decision_media_total',
       'decision_followers_gained', 'decision_followers_total',
       'decision_following_gained', 'decision_following_total',
      'aftercamp_media_gained', 'aftercamp_media_total',
       'aftercamp_followers_gained', 'aftercamp_followers_total',
       'aftercamp_following_gained', 'aftercamp_following_total',
      'follower_num']
        category_features_list = []
        for feat in self.category_feature_names:
            category_features = pd.get_dummies(self.x[feat]).values
            category_features_list.append(category_features)

        # Normalize numerical features
        numerical_features_list = []
        for feat in self.numerical_feature_names:
            numerical_features = self.x[[feat]].values.astype(np.float32)
            numerical_features = (numerical_features - numerical_features.mean(axis=0)) / numerical_features.std(axis=0)

        # Concatenate categorical and numerical features
        features = np.concatenate(category_features_list + numerical_features_list, axis=1)
        
        self.x = features
        self.y = self.y.to_numpy()
        
    def __len__(self):
        return len(self.x)
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = self.x[index]
        y = self.y[index]

        return np.array(x), np.array(y)

class ToNumpyArray(object):
    def __call__(self, sample):
        print(sample)
        x, y = sample
        return {'x': np.array(x), 'y': np.array(y)}