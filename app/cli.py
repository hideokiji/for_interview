import typer
import json
import tempfile
import warnings
import torch
import pandas as pd


import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback

from numpyencoder import NumpyEncoder
from pathlib import Path
from typing import Dict, Optional
from argparse import Namespace

from recsys import config, eval, main, utils, predicts

warnings.filterwarnings("ignore")
# Typer cli app
app = typer.Typer()

@app.command()
def optimize(
    params_fp: Path = Path(config.config_dir, "params.json"),
    study_name: Optional[str]= 'optimization',
    num_trials: int = 10
)->None:

    params = Namespace(**utils.load_dict(params_fp))

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction='maximize', pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name=params.eval_metrics)

    study.optimize(
        lambda trial: main.objective(trial=trial, params_fp=params_fp),
        n_trials=params.num_trials, 
        callbacks=[mlflow_callback],
    )

    # all trials
    print("Best value "+ str(params.eval_metrics)+":%s"%{study.best_trial.value})
    params = {**params.__dict__, **study.best_trial.params}
    #params['threshold'] = study.best_trial.user_attrs['threshold']
    print(params)
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    json.dumps(params, indent=2, cls=NumpyEncoder)


@app.command()
def train_model_app(
    params_fp: Path = Path(config.config_dir, "params.json"),
    model_dir: Path = Path(config.model_dir),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model"
    )->None:
    """Train a model using the specified parameters

    Args:
        params_fp (Path, optional): Parameters to use for training
        model_dir (Path): location of model artifacts
        experiment_name(str, optional): Name of the experiment to save to run to.
        run_name (str, optional): Name of the run.
    """

    params = Namespace(**utils.load_dict(params_fp))
    # end active mlflow run (if there is one)
    

    # start run
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id

        # train
        # notice that there is no trail. 
        artifacts = main.train_model(params_fp=params_fp)

        # Log metrics
        performance = artifacts['performance']
        json.dumps(performance['overall'], indent=2)
        metrics = {
            "precision": performance['overall']['precision'],
            "recall": performance['overall']['recall'],
            "f1": performance['overall']['f1'],
            "best_val_loss": artifacts['loss'],
            "HR": performance['overall']['HR'],
            "NDCG": performance['overall']['NDCG']
        }

        mlflow.log_metrics(metrics)

        # log artifacts stored inside stores/model
        with tempfile.TemporaryDirectory() as dp:
        #with tempfile.mkstemp() as dp:   
            utils.save_dict(vars(artifacts['params']), Path(dp, "params.json"), cls=NumpyEncoder)
            utils.save_dict(performance, Path(dp, "performance.json"))
            torch.save(artifacts['model'].state_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts['params']))
    

    # save for the repo model
    open(Path(model_dir, "run_id.txt"), 'w').write(run_id)
    utils.save_dict(vars(params), Path(model_dir, "params.json"), cls=NumpyEncoder)
    utils.save_dict((performance), Path(model_dir, "performance.json"))


@app.command()
def recommendation(
    item_id: int,
    top_k: int,
    run_id: str = open(Path(config.model_dir, "run_id.txt")).read()
    )->Dict:
    """Top k item recommendation from the best experiments
    
    Args:
        item_id: item_id from feed_back from user
        artifacts: load from the best exeriments.
    Return:
        return top_k recommended items
    """

    artifacts = main.load_artifacts(run_id=run_id)
    recommends = predicts.item_recommendations(item_id=item_id, top_k=top_k, artifacts=artifacts)

    return recommends

@app.command()
def predict(
    dataset: torch.utils.data.Dataset = utils.get_data(),
    run_id: str = open(Path(config.model_dir, "run_id.txt")).read() 
    )->Dict:
    """Inference with a new dataset using best params in the best experiment

    Args:
        dataset: batch dataset stored in feature_store
        run_id: run_id of the best_experiment
    
    Return:
        binary feed_back of predicted ratings, grouth_truth ratings and performance metrics
    """
    
    artifacts = main.load_artifacts(run_id=run_id)
    prediction = predicts.predict(artifacts=artifacts, dataset=dataset)
    return prediction

