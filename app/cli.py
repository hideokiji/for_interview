from typing import Dict, Optional 
from pathlib import Path 

import typer 
import mlflow  
import optuna 
from optuna.integration.mlflow import MLflowCallback 

import json 

from numpyencoder import NumpyEncoder 

from argparse import Namespace 
from recsys import config, eval, main, utils, predicts 

app = typer.Typer()

@app.command()
def optimize(
    params_fp: Path = Path(config.config_dir, "params.json"),
    study_name: Optional[str] = 'optimization',
    num_trials: int=10)->None:

    print(params_fp)
    params = Namespace(**utils.load_dict(params_fp))
    
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction='maximize', pruner=pruner)
    mlflow_callbacks = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name=params.eval_metrics) 

    study.optimize(
        lambda trial: main.objective(trial=trial, params_fp=params_fp),
        n_trials=params.num_trials,
        callbacks=[mlflow_callbacks],
    )

    print("Best value"+str(params.eval_metrics) +":%s"%{study.best_trial.value})
    params = {**params.__dict__, **study.best_trial.params}
    utils.save_dict(params, params_fp, cls=NumpyEncoder)
    json.dumps(params, indent=2, cls=NumpyEncoder)

@app.command()
def train_model_app(
    params_fp: Path = Path(config.config_dir, "params.json"),
    model_dir: Path = Path(config.model_dir),
    experiment_name: Optional[str] = "best",
    run_name: Optional[str] = "model") -> None:

    params = Namespace(**utils.load_dict(params_fp))

    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id 

        artifacts = main.train_model(params_fp=params_fp)
        performance = artifacts['performance']
        json.dumps(performance['overall']['precision'])
        metrics = {
            "precision": performance['overall']['precision'],
            "recall": performance['overall']['recall'],
            "f1": performance['overall']['f1'],
            "HR": performance['overall']['loss'],
            "NDCG": performance['overall']['NDCG'],
            "best_val_loss": artifacts['loss'],
        }

        mlflow.log_metrics(metrics)

        with tempfile.mkstemp() as dp:
            utils.save_dict(vars(artifacts['params']), Path(dp, "params.json"), cls=NumpyEncoder)
            utils.save_dict(performance, Path(dp, 'performance.json'))
            torch.save(artifacts['model'].save_dict(), Path(dp, "model.pt"))
            mlflow.log_artifacts(dp)
        mlflow.log_params(vars(artifacts['params']))

        open(Path(model_dir, "run_id.txt"), 'w').write(run_id)
        utils.save_dict(vars(params), Path(model_dir, "params.json"), cls=NumpyEncoder)
        utils.save_dict((performance), Path(model_dir, "performance.json"))

@app.command()
def recommendation(
    item_id: int,
    top_k: int,
    run_id: int)->Dict:
    
    artifacts = main.load_artifacts(run_id=run_id)
    recommends = predicts.item_recommedation(item_id=item_id, top_k=top_k, artifacts=artifacts)
    return recommends 
