import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from itertools import islice, chain, repeat
import torch.utils.data


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, file):
        self.data = pd.read_csv(file, low_memory=False)
        #self.data.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        self.data.fillna(0,inplace=True)
        
        self.x = self.data[['country_y', 'yearmonth', 'campaign_startdate_cor', 
        'campaign_enddate_cor',
        'media_gained_decisiondate', 
        'media_total_decisiondate', 
        'followers_gained_decisiondate', 
        'followers_total_decisiondate', 
        'following_gained_decisiondate', 
        'following_total_decisiondate', 
        'media_gained_decisiondate_7days', 
        'media_total_decisiondate_7days', 
        'followers_gained_decisiondate_7days', 
        'followers_total_decisiondate_7days', 
        'following_gained_decisiondate_7days', 
        'following_total_decisiondate_7days', 
        'media_gained_decisiondate_15days', 
        'media_total_decisiondate_15days', 
        'followers_gained_decisiondate_15days', 
        'followers_total_decisiondate_15days', 
        'following_gained_decisiondate_15days', 
        'following_total_decisiondate_15days', 
        'media_gained_decisiondate_30days', 
        'media_total_decisiondate_30days', 
        'followers_gained_decisiondate_30days', 
        'followers_total_decisiondate_30days', 
        'following_gained_decisiondate_30days', 
        'following_total_decisiondate_30days', 
        'media_gained_decisiondate_60days', 
        'media_total_decisiondate_60days', 
        'followers_gained_decisiondate_60days', 
        'followers_total_decisiondate_60days', 
        'following_gained_decisiondate_60days', 
        'following_total_decisiondate_60days', 
        'media_gained_decisiondate_90days', 
        'media_total_decisiondate_90days', 
        'followers_gained_decisiondate_90days', 
        'followers_total_decisiondate_90days', 
        'following_gained_decisiondate_90days', 
        'following_total_decisiondate_90days', 
        'followers_rate_decisiondate_7days', 
        'following_rate_decisiondate_7days', 
        'media_rate_decisiondate_7days', 
        'followers_rate_decisiondate_15days', 
        'following_rate_decisiondate_15days', 
        'media_rate_decisiondate_15days', 
        'followers_rate_decisiondate_30days', 
        'following_rate_decisiondate_30days', 
        'media_rate_decisiondate_30days', 
        'followers_rate_decisiondate_60days', 
        'following_rate_decisiondate_60days', 
        'media_rate_decisiondate_60days', 
        'followers_rate_decisiondate_90days', 
        'following_rate_decisiondate_90days', 
        'media_rate_decisiondate_90days',  
        'avg_total_orders', 
#         'avg_revenue', 
        'avg_new_customers', 
        'avg_customer_num', 
        'first_campaign_time', 
        'new_influencer', 
#         'last_revenue', 
        'last_orders_total', 
        'last_new_customers', 
        'last_customer_num', 
        'days_from_last_posting', 
        'days_from_last_sponsored', 
        'days_from_last_org_branded', 
        'days_from_last_org_nonbranded', 
        'days_interval_posting', 
        'days_interval_sponsored', 
        'days_interval_org_branded', 
        'days_interval_org_nonbranded', 
        '90days_comment_count', 
        '90days_like_count', 
        'num_sponsored_posts_90days', 
        'num_organic_branded_90days', 
        'num_organic_nonbranded_90days', 
        'num_posts_90days', 
        '60days_comment_count', 
        '60days_like_count', 
        'num_sponsored_posts_60days', 
        'num_organic_branded_60days', 
        'num_organic_nonbranded_60days', 
        'num_posts_60days', 
        '30days_comment_count', 
        '30days_like_count', 
        'num_sponsored_posts_30days', 
        'num_organic_branded_30days', 
        'num_organic_nonbranded_30days', 
        'num_posts_30days', 
        '15days_comment_count', 
        '15days_like_count', 
        'num_sponsored_posts_15days', 
        'num_organic_branded_15days', 
        'num_organic_nonbranded_15days', 
        'num_posts_15days', 
        '7days_comment_count', 
        '7days_like_count', 
        'num_sponsored_posts_7days', 
        'num_organic_branded_7days', 
        'num_organic_nonbranded_7days', 
        'num_posts_7days'
        ]]
        self.y1 = self.data[['accept']]
        self.y2 = self.data[['Revenue_reg']]
        self.y3 = self.data[['reputation_change_reg']]
        
        #self.category_feature_names = [, 'year', 'month']
        self.category_feature_names = ['country_y', 'yearmonth']
        self.numerical_feature_names = ['media_gained_decisiondate', 
        'media_total_decisiondate', 
        'followers_gained_decisiondate', 
        'followers_total_decisiondate', 
        'following_gained_decisiondate', 
        'following_total_decisiondate', 
        'media_gained_decisiondate_7days', 
        'media_total_decisiondate_7days', 
        'followers_gained_decisiondate_7days', 
        'followers_total_decisiondate_7days', 
        'following_gained_decisiondate_7days', 
        'following_total_decisiondate_7days', 
        'media_gained_decisiondate_15days', 
        'media_total_decisiondate_15days', 
        'followers_gained_decisiondate_15days', 
        'followers_total_decisiondate_15days', 
        'following_gained_decisiondate_15days', 
        'following_total_decisiondate_15days', 
        'media_gained_decisiondate_30days', 
        'media_total_decisiondate_30days', 
        'followers_gained_decisiondate_30days', 
        'followers_total_decisiondate_30days', 
        'following_gained_decisiondate_30days', 
        'following_total_decisiondate_30days', 
        'media_gained_decisiondate_60days', 
        'media_total_decisiondate_60days', 
        'followers_gained_decisiondate_60days', 
        'followers_total_decisiondate_60days', 
        'following_gained_decisiondate_60days', 
        'following_total_decisiondate_60days', 
        'media_gained_decisiondate_90days', 
        'media_total_decisiondate_90days', 
        'followers_gained_decisiondate_90days', 
        'followers_total_decisiondate_90days', 
        'following_gained_decisiondate_90days', 
        'following_total_decisiondate_90days', 
        'followers_rate_decisiondate_7days', 
        'following_rate_decisiondate_7days', 
        'media_rate_decisiondate_7days', 
        'followers_rate_decisiondate_15days', 
        'following_rate_decisiondate_15days', 
        'media_rate_decisiondate_15days', 
        'followers_rate_decisiondate_30days', 
        'following_rate_decisiondate_30days', 
        'media_rate_decisiondate_30days', 
        'followers_rate_decisiondate_60days', 
        'following_rate_decisiondate_60days', 
        'media_rate_decisiondate_60days', 
        'followers_rate_decisiondate_90days', 
        'following_rate_decisiondate_90days', 
        'media_rate_decisiondate_90days', 
        'avg_total_orders', 
#         'avg_revenue', 
        'avg_new_customers', 
        'avg_customer_num', 
        'new_influencer', 
#         'last_revenue', 
        'last_orders_total', 
        'last_new_customers', 
        'last_customer_num', 
        'days_from_last_posting', 
        'days_from_last_sponsored', 
        'days_from_last_org_branded', 
        'days_from_last_org_nonbranded', 
        'days_interval_posting', 
        'days_interval_sponsored', 
        'days_interval_org_branded', 
        'days_interval_org_nonbranded', 
        '90days_comment_count', 
        '90days_like_count', 
        'num_sponsored_posts_90days', 
        'num_organic_branded_90days', 
        'num_organic_nonbranded_90days', 
        'num_posts_90days', 
        '60days_comment_count', 
        '60days_like_count', 
        'num_sponsored_posts_60days', 
        'num_organic_branded_60days', 
        'num_organic_nonbranded_60days', 
        'num_posts_60days', 
        '30days_comment_count', 
        '30days_like_count', 
        'num_sponsored_posts_30days', 
        'num_organic_branded_30days', 
        'num_organic_nonbranded_30days', 
        'num_posts_30days', 
        '15days_comment_count', 
        '15days_like_count', 
        'num_sponsored_posts_15days', 
        'num_organic_branded_15days', 
        'num_organic_nonbranded_15days', 
        'num_posts_15days', 
        '7days_comment_count', 
        '7days_like_count', 
        'num_sponsored_posts_7days', 
        'num_organic_branded_7days', 
        'num_organic_nonbranded_7days', 
        'num_posts_7days'
        ]
        category_features_list = []
        for feat in self.category_feature_names:
            category_features = pd.get_dummies(self.x[feat]).values
            category_features_list.append(category_features)

        # Normalize numerical features
        numerical_features_list = []
        for feat in self.numerical_feature_names:
            numerical_features = self.x[[feat]].values.astype(np.float32)
            numerical_features = (numerical_features - numerical_features.mean(axis=0)) / numerical_features.std(axis=0)
            numerical_features_list.append(numerical_features)

        # Concatenate categorical and numerical features
        features = np.concatenate(category_features_list + numerical_features_list, axis=1)
        
        # Here to See Feature dims
        print('See Feature Shape:', features.shape)
        self.x = features
        self.y1 = self.y1.to_numpy()
        self.y2 = self.y2.to_numpy()
        self.y3 = self.y3.to_numpy()
        
    def __len__(self):
        return len(self.x)
    

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()

        x = self.x[index]
        y1 = self.y1[index]
        y2 = self.y2[index]
        y3 = self.y3[index]
        return np.array(x, dtype = np.float32), np.array(y1, dtype = np.float32), np.array(y2, dtype = np.float32), np.array(y3, dtype = np.float32)

class ToNumpyArray(object):
    def __call__(self, sample):
        print(sample)
        x, y = sample
        return {'x': np.array(x), 'y': np.array(y)}
    
    

