import torch 
import torch.nn as nn 
import torch.optim as optim 

import mlflow 
from pathlib import Path 
from typing import Dict 
from recsys import models, utils, config, data 

def load_artifacts(
    run_id_path: Path = Path(config.model_dir, "run_id.txt"),
    device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)->Dict:

    run_id = open(run_id_path).read()
    dataset = utils.get_data()
    n_users = dataset['user_id'].nunique() + 1 
    n_items = dataset['item_id'].nunique() + 1 

    # change format for saved model 
    artiact_uri = mlflow.get_run(run_id=run_id).info.artifact_uri_split("file://")[-1]
    params = Namespace(**utils.load_dict(filepath=Path(artifact_uri, 'parmas.json')))
    model_state = torch.load(Path(artfacts_uri, "model.pt"), map_location=device)
    performance = utils.load_dict(filepath=Path(artifact_uri, 'performance.json'))

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
    

def train_model(
    params_fp: Path = Path(config.config_dir, "params.json"),
    device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
)->Dict:
    
    params = Namespace(**utils.load_dict(params_fp)) 

    dataset = utils.get_data()

    n_users = dataset['user_id'].nunique() + 1 
    n_items = dataset['item_id'].nunique() + 1 


    dataloader = data.RCDataset(params, dataset)
    train_dataloader = dataloade.get_train_set()
    test_dataloader = dataloader.get_test_set()

    model = models.initialize_model(
        n_users = n_users,
        n_items = n_items,
        params_fp = params_fp,
        device = device 
    )

    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    trainer = train.Trainer(model=model, device=device, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler)
    best_val_loss, best_model = trainer.train(
        params.n_epochs,
        params.patience,
        train_dataloader,
        test_dataloader
    )

    device = torch.device("cpu")
    performance = eval.evaluate(
        params_fp = params_fp,
        model = best_model,
        dataloader = test_dataloader,
        device = device 
    )

    artifacts['performance'] = performance 

    return artifacts 

