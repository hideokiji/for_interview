import os 
from pathlib import Path 

base_dir = Path(os.getcwd()).absolute()
dataset_dir = Path(base_dir, 'ml-100k')

print(base_dir, dataset_dir)
