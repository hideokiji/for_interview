import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 

from argparse import Namespace 
from pathlib import Path 
import json 
import optuna
import mlflow 
import numpy as np 

from recsys import utils, config 

class Trainer(object):
    def __init__(self,
        model,
        #device: torch.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        device: torch.device = torch.device("cpu"),
        loss_fn=None,
        optimizer=None,
        scheduler=None,
        trial=None):

        self.model= model 
        self.device = device 
        self.loss_fn = loss_fn 
        self.optimizer = optimizer 
        self.scheduler = scheduler 
        self.trial = trial 

    def train_step(self, dataloader):
        self.model.train()
        loss = 0.0 

        for batch, (user, item, label) in enumerate(dataloader):
            user = user.to(self.device)
            item = item.to(self.device)
            label = label.to(self.device)
     
            # forward
            self.optimizer.zero_grad()
            prediction = self.model.predict(user, item)
            
            # backward 
            loss = self.loss_fn(prediction, label)
            loss.backward()
            self.optimizer.step()

            loss += (loss.item() - loss)/(batch + 1) 
        return loss 

    def eval_step(self, dataloader):
        self.model.eval()

        loss = 0.0 
        predictions, labels = [], []
        with torch.no_grad():
            for batch, (user, item, label) in enumerate(dataloader):
                prediction = self.model.predict(user, item)
                J = self.loss_fn(prediction, label).item()

                loss += (J - loss)/(batch + 1)

                prediction = prediction.numpy()
                predictions.extend(prediction)
                labels.extend(label.numpy())
        return loss, np.vstack(labels), np.vstack(predictions)

    def predict_step(self, dataloader):
        self.model.eval()
        predictions, labels = [], []

        with torch.no_grad():
            for batch, (user, item, label) in enumerate(dataloader):
                prediction = self.model.predict(user, item)

                prediction = prediction.numpy()
                predictions.extend(prediction)

                labels.extend(label.numpy())

        return np.vstack(labels), np.vstack(predictions)

    def train(self, num_epochs, patience, train_dataloader, val_dataloader):
        best_val_loss = np.inf
        for epoch in range(num_epochs):
            train_loss = self.train_step(dataloader=train_dataloader)
            val_loss, _, _ = self.eval_step(dataloader=val_dataloader)

            self.scheduler.step(val_loss)

            # early stopping 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = self.model
                _patience = patience 
            else:
                _patience -= 1 
            if not _patience:
                print("Stopping early")
                break 

            # pruning 
            if self.trial:
                self.trial.report(val_loss, epoch)
                if self.trial.report(val_loss, epoch):
                    raise optuna.TrialPruned()

            # Tracking --> bug 
            """
            mlflow.log_metrics(
                {'train_loss': train_loss, 'val_loss':val_loss}, step=epoch 
            )
            """
            print(
              f"Epoch: {epoch + 1}",
              f"train_loss: {train_loss:.5f}",
              f"val_loss: {val_loss:.5f}",
              f"lr: {self.optimizer.param_groups[0]['lr']:.2E}",
              f"patience: {_patience}"
            )

        return best_val_loss, best_model 

def train(
    params_fp: Path=Path(config.config_dir, "params.json"),
    #device: torch.device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu"),
    device: torch.device = torch.device("cpu"),
    trial: optuna.trial._trial.Trial = None 
    ):  

    params = Namespace(**utils.load_dict(params_fp))

    dataset = utils.get_data()
    n_users = dataset['user_id'].nunique() + 1 
    n_items = dataset['item_id'].nunique() + 1 

    Dataloader = data.RCDataloader(params, dataset)
    print(Dataloader)
    train_dataloader = Dataloader.get_train_set()
    test_dataloader = Dataloader.get_test_set()

    model = models.initialize(
        n_users = n_users,
        n_items = n_items,
        params_fp = params_fp,
        device = device 
    )

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=parmas.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode = 'min', factor=0.5, patience=params.patience 
    )

    train = Trainer(
        model = model,
        device = device,
        loss_fn = loss_fn,
        optimizer = optimizer,
        scheduler = scheduler,
        trial = trial, 
    )

    best_val_loss, best_model = trainer.train(
        params.n_epochs, params.patience, train_dataloader, test_dataloader 
    )

    return params, best_model, best_val_loss 
