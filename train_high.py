import torch
try:
    import wandb
except ImportError:
    wandb = None
import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper
from src.env.custom_sync_vector_env import CustomSyncVectorEnv, make_custom_vec
import os
import argparse
from torch import optim

from src.train.high_trainer import HighTrainerOnline, HighTrainerReplay
from src.train.utils import load_config, load_training_params, load_schedulable_param
from src.model.factory import build_model_from_loaded_config, load_config, load_model
from src.data.state_transition_dataset import StateTransitionsActionsDataset
import shutil

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, default="config-high.yaml", help='Path to model config file')
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

    model_name = training_params.get('model_name', None)
    dataset_name = training_params.get('dataset', None)
    model_low_name = training_params.get('model_low', None)
    model_action_name = training_params.get('model_action', None)
    batch_size = training_params.get('batch_size', 32)
    trajectory_length = training_params.get('trajectory_length', 5)
    num_epochs = training_params.get('num_epochs', 1000)
    beta_config = training_params.get('beta', 1.0)
    beta = load_schedulable_param(beta_config)
    bptt_truncate = training_params.get('bptt_truncate', None)

    model_high = build_model_from_loaded_config(config, device=device)
    model_low_path = os.path.join(project_root, 'models', model_low_name)
    model_low = load_model(model_low_path, device=device)
    model_action_path = os.path.join(project_root, 'models', 'latent_action', model_action_name)
    model_action = load_model(model_action_path, device=device)

    wandb.init(project='top-down-preferences', config=config)
    if model_name is not None:
        wandb.run.name = model_name
    else:
        model_name = wandb.run.name

    eval_dir = os.path.join(project_root, 'data', 'eval', 'high', model_name)
    os.makedirs(eval_dir, exist_ok=True)
    models_dir = os.path.join(project_root, 'models', 'high', model_name)
    os.makedirs(models_dir, exist_ok=True)
    
    if dataset_name is not None:
        data_root = os.path.join(project_root, 'data', 'state_pairs')
        dataset = StateTransitionsActionsDataset(os.path.join(data_root, dataset_name))
        dataset = dataset
    else:
        dataset = None

    config_save_path = os.path.join(models_dir, 'config.yaml')
    shutil.copy2(config_path, config_save_path)

    optimizer = optim.Adam(model_high.parameters(), lr=1e-3)
    
    if dataset is None:
        trainer = HighTrainerOnline(optimizer, batch_size, trajectory_length, beta=beta, device=device, bptt_truncate=bptt_truncate, eval_every_n_epochs=100, eval_img_root=eval_dir)
        #env = gym.make_vec("MiniGrid-FourRooms-v0", num_envs=batch_size, vectorization_mode="sync", wrappers=[RGBImgPartialObsWrapper])
        env = make_custom_vec("MiniGrid-FourRooms-v0", num_envs=batch_size, wrappers=[RGBImgPartialObsWrapper])
        trainer.train(model_high, model_low, model_action, env, num_epochs, models_dir)
    else:
        trainer = HighTrainerReplay(optimizer, batch_size, trajectory_length, beta=beta, device=device, bptt_truncate=bptt_truncate, eval_every_n_epochs=10, eval_img_root=eval_dir)
        trainer.train(model_high, model_low, model_action, dataset, num_epochs, models_dir)

    wandb.finish()