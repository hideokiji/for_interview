import os 
from pathlib import Path 

base_dir = Path(os.getcwd()).absolute()
dataset_dir = Path(base_dir, 'ml-100k')
config_dir = Path(base_dir, "config")
model_dir = Path(base_dir, "model")
print(base_dir, dataset_dir, config_dir)
