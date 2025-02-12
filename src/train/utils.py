import torch
import torch.nn.functional as F
import yaml

def obs2tensor(obs):
    return torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.

def index2onehot(index, batch_size=64, length=4):
    onehot = torch.zeros(batch_size, length)
    onehot[torch.arange(batch_size), index] = 1
    return onehot

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def load_training_params(config):
    training_params = config["training"]
    return training_params