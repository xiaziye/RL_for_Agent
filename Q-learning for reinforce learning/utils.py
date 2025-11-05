import os
import yaml
import numpy as np

def load_config(config_path: str = "config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)