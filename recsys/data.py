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
            torch.tensor(rating, dtype=torch.long)
        )


class DataLoader(object):
    def __init__(self,
        params: Namespace,
        ratings):
    
        self.ratings = ratings 
        self.params = params 
        self.batch_size = self.params.batch_size 
        self.train_ratings, self.test_ratings = self._leave_one_out(self, ratings)

    def _leave_one_out(self, ratings): 
        ratings["rank_latest"] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
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

        return torch.utils.data.Dataloader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.params.num_workers)

    def get_test_set(self):
        users, items, ratings = [], [], []
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

        return torch.utils.data.Dataloader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.params.num_workers)
