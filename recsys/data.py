import torch
import torch.nn as nn
from argparse import Namespace


class RCDataset(torch.utils.data.Dataset):
    def __init__(self, user_list, item_list, rating_list):
        super(RCDataset, self).__init__()
        self.user_list = user_list
        self.item_list = item_list
        self.rating_list = rating_list
    
    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        user = self.user_list[idx]
        item = self.item_list[idx]
        rating = self.rating_list[idx]
    
        return (
            torch.tensor(user, dtype=torch.long),
            torch.tensor(item, dtype=torch.long),
            torch.tensor(rating, dtype=torch.float)
        )

class RCDataloader(object):
    def __init__(self, 
        params: Namespace, 
        ratings):
        self.ratings = ratings
        self.params = params
        self.batch_size = self.params.batch_size
        self.train_ratings, self.test_ratings = self._leave_one_out(self.ratings)

    def _leave_one_out(self, ratings):
        """
        Leave one out cross-validation protocols
        """
        ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
        test = ratings.loc[ratings['rank_latest']==1]
        train = ratings.loc[ratings['rank_latest']>1]
        return train[['user_id', 'item_id', 'rating']], test[['user_id', 'item_id', 'rating']]

    def get_train_set(self):
        users, items, ratings = [], [], []
        train_ratings = self.train_ratings

        for row in train_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
        
        dataset = RCDataset(
            user_list = users,
            item_list = items,
            rating_list = ratings
        )

        return torch.utils.data.DataLoader(dataset, batch_size = self.batch_size, shuffle=True, num_workers= self.params.num_workers)
    
    def get_test_set(self):
        users, items, ratings= [], [], []
        test_ratings = self.test_ratings

        for row in test_ratings.itertuples():
            users.append(int(row.user_id))
            items.append(int(row.item_id))
            ratings.append(float(row.rating))
        
        dataset = RCDataset(
            user_list = users,
            item_list = items,
            rating_list = ratings
        )

        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.params.num_workers)


class BPRData(torch.utils.data.Dataset):
    def __init__(self,
            features,
            num_item,
            train_mat=None,
            num_ng=0,
            is_training=None):
        self.features = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training

    def ng_sample(self):

        self.features_fill = []
        for x in self.features:
            u, i = x[0], x[1]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_items)
                while (u,j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_fill.append([u,i,j])

    def __len__(self):
        return self.num_ng * len(self.features) if self.is_training else len(self.features)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training else self.features

        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2] if self.is_training else features[idx][1]

        return user, item_i, item_j