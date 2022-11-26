from pathlib import Path 
from argparse import Namespace 
from typing import Dict, List, Tuple 
import numpy as np 

import torch 

from sklearn.metrics import precision_recall_fscore_support 
from recsys import train, data, config, utils 

def hit(ng_item, pred_item):
    if ng_item in pred_item:
        return 1
    return 0

def ndcg(ng_item, pred_item):
    if ng_item in pred_item:
        index = pred_item.index(ng_item)
        return np.reciprocal(np.log2(index + 2))
    return 0 

def rec_metrics(model, test_loader, top_k, device):
    HR, NDCG = [], [] 

    for user, item, label in test_loader:
        user = user.to(device)
        item = item.to(device)

        predictions = model(user, item)
        _, indices = torch.topk(predictions, top_k)
        recommends = torch.take(item, indices).cpu().numpy().tolist()

        ng_item = item[0].item()
        HR.append(hit(ng_item, recommends))
        NDCG.append(ndcg(ng_item, recommends))

    return np.mean(HR), np.mean(NDCG)


def binary_feedback(ratings, threshold):
    ratings = np.array(ratings).flatten()
    normalize = ratings - threshold 
    normalize2 = [] 

    for i in range(len(ratings)):
        normalize2.append(0 if normalize[i] <0 else 1)

    return normalize2 


def get_metrics(
    model,
    dataloader: torch.utils.data.Dataloader,
    top_k: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    device: torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
)->Dict:


    metrics = {"overall":{}}
    HR, NDCG = rec_metrics(model, dataloader, top_k, device)

    overall_metrics = precision_recall_fscore_support(y_true, y_pred, average='weighted') 
    metrics['overall']['f1'] = overall_metrics[0]
    metrics['overall']['recall'] = overall_metrics[1]
    metrics['overall']['precision'] = overall_metrics[2]
    
    metrics['overall']['HR'] = HR 
    metrics['overall']['NDCG'] = NDCG

    return metrics

def evaluate(
    model,
    dataloader : torch.utils.data.Dataloader,
    params_fp: Path = Path(config.config_dir, "params.json"),
    device: torch.device = torch.device('cpu'),
    )->Tuple:

    params = Namespace(**utils.load_dict(params_fp))
    metrics = {"overall":{}}
    
    trainer = train.Trainer(model, device)
    y_true, y_pred = trainer.predict_step(dataloader=dataloader)
    
    y_true = binary_feedback(y_true, params.threshold)
    y_pred = binary_feedback(y_pred, params.threshold)

    performance = {}
    performance = get_metrics(mdoel, dataloader, params.top_k, y_true, y_pred, device )
