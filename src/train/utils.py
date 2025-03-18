import torch
import torch.nn.functional as F
import yaml

from src.train.scheduler import LinearScheduler, CosineScheduler, ConstantScheduler

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

def scale_onehot(onehot):
    return 2*onehot - 1

def load_schedulable_param(param_config):
    if isinstance(param_config, dict):
        scheduler_type = param_config.get("type", "constant").lower()
        if scheduler_type == "linear":
            return LinearScheduler(
                start_value=param_config["start"],
                end_value=param_config["end"],
                num_steps=param_config["steps"]
            )
        elif scheduler_type == "cosine":
            return CosineScheduler(
                start_value=param_config["start"],
                end_value=param_config["end"],
                num_steps=param_config["steps"]
            )
        elif scheduler_type == "constant":
            # Use a constant scheduler, or you could just return the raw value
            return ConstantScheduler(param_config.get("value", 1.0))
        else:
            raise ValueError(f"Unknown beta scheduler type: {scheduler_type}")
    else:
        # If beta_config is a float, just wrap it in a constant scheduler or use it directly.
        return ConstantScheduler(param_config)