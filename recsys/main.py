import torch 
import torch.nn as nn 
import torch.optim as optim 

import mlflow 
import optuna 

from argparse import Namespace 
from pathlib import Path 
from typing import Dict 
from recsys import models, utils, config, data, eval 

from recsys.train import Trainer  

def load_artifacts(
    run_id: str = open(Path(config.model_dir, "run_id.txt")),
    #device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    device: torch.device = torch.device("cpu")
)->Dict:

    #run_id = open(run_id_path).read()
    dataset = utils.get_data()
    n_users = dataset['user_id'].nunique() + 1 
    n_items = dataset['item_id'].nunique() + 1 

    # change format for saved model 
    artifact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri.split("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, 'params.json')))
    model_state = torch.load(Path(artifact_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, 'performance.json'))

    # move config.config_dir to artifact_uri 
    params_fp = Path(config.config_dir, "params.json")
    model = models.initialize_model(
        n_users = n_users,
        n_items = n_items,
        params_fp = params_fp,
        device = device 
    )
    model.load_state_dict(model_state)

    return {
        'params': params,
        'model': model, 
        'performance': performance
    }
    
def objective(
    trial: optuna.trial._trial.Trial,
    params_fp: Path = Path(config.config_dir, "params.json"),
    #device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device: torch.device = torch.device("cpu"),
    )->float:
    
    params = Namespace(**utils.load_dict(params_fp))

    params.dropout_p = trial.suggest_uniform("dropout", 0.3, 0.8)
    params.lr = trial.suggest_loguniform("lr", 1e-5, 1e-4)
    params.threshold = trial.suggest_uniform("threshold", 3.5, 5) 
    
    artifacts = train_model(params_fp=params_fp, device=device, trial=trial)
    
    params = artifacts['params']
    performance = artifacts['performance']

    trial.set_user_attr("f1", performance['overall']['f1'])
    trial.set_user_attr("recall", performance['overall']['recall'])
    trial.set_user_attr("precision", performance['overall']['precision'])
    trial.set_user_attr('HR', performance['overall']['HR'])
    trial.set_user_attr('NDCG', performance['overall']['NDCG']) 
    
    if params.save_model:
        torch.save(artifacts['model'].state_dict(), params.model+"recsys.pkl")

    return performance['overall'][params.eval_metrics]

def train_model(
    params_fp: Path = Path(config.config_dir, "params.json"),
    #device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
    device: torch.device = torch.device("cpu"),
    trial: optuna.trial._trial.Trial = None, 
)->Dict:
    
    params = Namespace(**utils.load_dict(params_fp)) 

    dataset = utils.get_data()

    n_users = dataset['user_id'].nunique() + 1 
    n_items = dataset['item_id'].nunique() + 1 

    print(data.RCDataloader.get_test_set)
    print(data.RCDataloader.get_train_set)
    Dataloader = data.RCDataloader(params, dataset)
    train_dataloader = Dataloader.get_train_set()
    test_dataloader = Dataloader.get_test_set()

    model = models.initialize_model(
        n_users = n_users,
        n_items = n_items,
        params_fp = params_fp,
        device = device 
    )

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    trainer = Trainer(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
    best_val_loss, best_model = trainer.train(
        params.n_epochs,
        params.patience,
        train_dataloader,
        test_dataloader
    )

    artifacts = {
        "params": params,
        "model": model,
        "loss": best_val_loss,
    }

    #device = torch.device("cpu")
    performance = eval.evaluate(
        params_fp = params_fp,
        model = best_model,
        dataloader = test_dataloader,
        device = device 
    )

    artifacts['performance'] = performance 

    return artifacts 

