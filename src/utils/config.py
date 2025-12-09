import yaml
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "configs")

def load_config(file):
    with open(os.path.join(CONFIG_DIR, file), "r") as f:
        return yaml.safe_load(f)

TRAINING = load_config("training.yaml")
GA_PARAMS = load_config("ga.yaml")

# Load bounds (named)
BOUNDS_DICT = load_config("bounds.yaml")["bounds"]

# This will be used by the genetic algorithm (list of (low, high))
BOUNDS = list(BOUNDS_DICT.values()) 
