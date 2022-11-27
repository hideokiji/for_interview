import os 
from pathlib import Path 
import mlflow 

# Directories
base_dir = Path(os.getcwd()).absolute()
config_dir = Path(base_dir, "config")
logs_dir = Path(base_dir, "logs")
data_dir = Path(base_dir, "data")
model_dir = Path(base_dir, "model")
stores_dir = Path(base_dir, "stores")
dataset_dir = Path(base_dir, "ml-100k")

# Local stores
model_registry = Path(stores_dir , "model")

# Create dirs
logs_dir.mkdir(parents=True, exist_ok=True)
data_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
stores_dir.mkdir(parents=True, exist_ok=True)
#feature_store.mkdir(parents=True, exist_ok=True)
model_registry.mkdir(parents=True, exist_ok=True)


# mlflow model registry (for uri/mlflow_id tracking)
mlflow.set_tracking_uri("file://"+str(model_registry.absolute()))

