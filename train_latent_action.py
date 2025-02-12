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

from src.data.state_transition_dataset import StateTransitionDataset
from src.train.latent_action_trainer import LatentActionTrainer
from src.train.utils import load_config, load_training_params
from src.model.factory import build_model_from_loaded_config, load_config, load_model
import shutil

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="config-latent-action.yaml", help='Path to model config file')
    argparser.add_argument('--low_model', type=str, default=None, help='Path to low model')
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
    train_ratio = training_params.get('train_ratio', 0.8)
    num_epochs = training_params.get('num_epochs', 1000)
    dataset_name = training_params.get('dataset_name', 'minigrid-5-2000.pt')
    lr = training_params.get('lr', 1e-3)
    beta = training_params.get('beta', 1.0)

    wandb.init(project='top-down-preferences', config=config)
    if model_name is not None:
        wandb.run.name = model_name
    else:
        model_name = wandb.run.name

    eval_dir = os.path.join(project_root, 'data', 'eval', 'latent_action', model_name)
    os.makedirs(eval_dir, exist_ok=True)
    models_dir = os.path.join(project_root, 'models', 'latent_action', model_name)
    os.makedirs(models_dir, exist_ok=True)
    data_root = os.path.join(project_root, 'data', 'state_pairs')

    config_save_path = os.path.join(models_dir, 'config.yaml')
    shutil.copy2(config_path, config_save_path)

    dataset = StateTransitionDataset(os.path.join(data_root, dataset_name))
    print(f'Dataset size: {len(dataset)}')

    if args.low_model is not None:
        low_model = load_model(os.path.join(project_root, 'models', args.low_model),  device=device)
        print(f'Loaded low model from {args.low_model}')
    else:
        low_model = None
        print('No low model provided, eval images will not be generated')

    optimizer = optim.Adam(model.parameters(), lr=lr)
    trainer = LatentActionTrainer(optimizer, batch_size, beta=beta, device=device, eval_img_root=eval_dir, eval_every_n_epochs=50)
    trainer.train(model, dataset, num_epochs, models_dir, train_ratio=train_ratio, low_model=low_model)