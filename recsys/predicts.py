from typing import Dict 
from recsys import models, data, config, utils 

import numpy as np 
import torch 

from argparse import Namespace
from pathlib import Path 

def predict(
    artifacts: Dict,
    dataset: torch.utils.data.Dataset = utils.get_data()
    ):

    params = artifacts["params"]
    model = artifacts["model"]

    dataloader = train.Trainer(model, device)
    dataloader = dataloader.get_test_set()

    trainer = train.Trainer(model, device)
    y_true, y_pred = trainer.predict_step(dataloader=dataloader)

    y_true = eval.binary_feedback(y_true, params.threshold)
    y_pred = eval.binary_feedback(y_pred, params.threshold)

    performance = {}
    performance = eval.get_metrics(model, dataloader, params.top_k, y_true, y_pred, device)
    return y_true, y_pred, performance 

def item_recommendations(
    item_id: int,
    top_k: int,
    artifacts: Dict )->Dict:
    
    best_model = artifacts['model']
    dataset = utils.get_data()

    # should use real-time data 
    params_fp = Path(config.config_dir, "params.json")
    params = Namespace(**utils.load_dict(params_fp))
    dataloader = data.RCDataloader(params, dataset)
    dataloader = dataloader.get_test_set()

    items_id = []
    users_id = []

    for user, item, _ in dataloader:
        items_id.append(item)
        users_id.append(user)

    items_id = torch.cat(items_id)
    users_id = torch.cat(users_id)
    """
    print(users_id)
    item_id_index = (items_id==item_id).nonzero(as_tuple=False)
    print(item_id_index.detach().numpy().tolist())
    #user_id = users_id(item_id_index.detach().numpy().tolist())
    user_id = item_id_index.detach().numpy().tolist()  
    print(user_id)
    """
    dataset = utils.get_data()
    items = torch.tensor(dataset['item_id'])

    predictions = best_model(users_id, torch.tensor(item_id))
    _, indices = torch.topk(predictions, top_k)
    recommends = torch.take(items, indices)
    print(dataset['title'][item_id])
    return (dataset['title'][recommends.cpu().detach().numpy().tolist()])

