import torch
try:
    import wandb
except ImportError:
    wandb = None
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper
from torch import nn, optim, Tensor
import torch.nn.functional as F
import os
import argparse

import src.model.encoder as encoder
import src.model.decoder as decoder
import src.model.transition as transition
from src.utils.latent_handler import GaussianLatentHandler, CategoricalLatentHandler

from src.model.model_low import ModelLow
from src.train.low_trainer import LowTrainerOnline
from src.train.utils import load_config, load_training_params
from src.model.factory import build_model_from_loaded_config, load_config
import shutil

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="config-low.yaml", help='Path to model config file')
    args = argparser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(project_root, args.config)

    try:
        config = load_config(config_path)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        exit(1)

    training_params = load_training_params(config)
    model = build_model_from_loaded_config(config, device=device)

    model_name = training_params.get('model_name', None)
    batch_size = training_params.get('batch_size', 32)
    trajectory_length = training_params.get('trajectory_length', 5)
    num_epochs = training_params.get('num_epochs', 1000)
    beta = training_params.get('beta', 1.0)

    wandb.init(project='top-down-preferences', config=config)
    if model_name is not None:
        wandb.run.name = model_name
    else:
        model_name = wandb.run.name

    eval_dir = os.path.join(project_root, 'data', 'eval', model_name)
    os.makedirs(eval_dir, exist_ok=True)
    models_dir = os.path.join(project_root, 'models', model_name)
    os.makedirs(models_dir, exist_ok=True)

    config_save_path = os.path.join(models_dir, 'config.yaml')
    shutil.copy2(config_path, config_save_path)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    trainer = LowTrainerOnline(optimizer, batch_size, trajectory_length, beta=beta, device=device, eval_img_root=eval_dir, eval_every_n_epochs=500)
    env = gym.make_vec("MiniGrid-FourRooms-v0", num_envs=batch_size, vectorization_mode="sync", wrappers=[RGBImgPartialObsWrapper])
    trainer.train(model, env, num_epochs, models_dir)

    wandb.finish()