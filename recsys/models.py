from pathlib import Path 
from typing import Dict, List  

import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F  

from argparse import Namespace 
#from utils import *
from recsys import utils, config 

class mfpt(nn.Module):
    def __init__(self,
        n_users: int,
        n_items: int,
        n_factors: int,
        dropout_p: float)->None:

        super(mfpt, self).__init__()

        self.n_users = n_users
        self.n_items =  n_items 
        self.n_factors = n_factors 
        self.dropout_p = dropout_p

        self.dropout = nn.Dropout(p=self.dropout_p)
        self.users_biases = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=1, sparse=True)
        self.items_biases = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=1, sparse=True) 
        self.user_factor = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=n_factors, sparse=True) 
        self.item_factor = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=n_factors, sparse=True)
    
    def forward(self, user: int, item: int)->torch.Tensor:
        user_embedding = self.user_factor(user).float()
        item_embedding = self.item_factor(item).float()
        
        # add bias 
        preds = self.users_biases(user).float()
        preds +=  self.items_biases(item).float()

        preds += torch.mul(self.dropout(user_embedding), self.dropout(item_embedding)).sum(dim=1, keepdim=True)

        return preds.reshape(-1).squeeze()

    def __call__(self, *args):
      return self.forward(*args)

    def predict(self, user, item):
      return self.forward(user, item)

def initialize_model(
        n_users: int = utils.get_data()['user_id'].nunique() + 1,
        n_items: int = utils.get_data()['item_id'].nunique() + 1,
        params_fp: Path = Path(config.config_dir, "params.json"),
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )->nn.Module:
    params = Namespace(**utils.load_dict(params_fp))
    
    model = mfpt(
        n_users = n_users,
        n_items = n_items,
        n_factors = params.n_factors,
        dropout_p = params.dropout_p
    )

    model = model.to(device)
    return model 
