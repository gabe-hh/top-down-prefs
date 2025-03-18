import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper
from src.env.custom_sync_vector_env import make_custom_vec
from src.data.state_transition_dataset import collect_states, collect_states_and_actions, save_state_transitions
from src.model.factory import load_model
import argparse
import torch
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect state pairs')
    parser.add_argument('--model', type=str, required=True, help='Model name')
    parser.add_argument('--action_model', type=str, default=None, help='Action model name')
    parser.add_argument('--env', type=str, default='MiniGrid-FourRooms-v0', help='Environment')
    parser.add_argument('--length', type=int, default=5, help='Length of trajectory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_trajectories', type=int, default=10, help='Number of trajectories')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__)) 

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    #env = gym.make_vec(args.env, num_envs=args.batch_size, vectorization_mode="sync", wrappers=[RGBImgPartialObsWrapper])
    env = make_custom_vec("MiniGrid-FourRooms-v0", num_envs=args.batch_size, wrappers=[RGBImgPartialObsWrapper])

    model_dir = os.path.join(project_root, 'models', args.model)
    model = load_model(model_dir, device=device)

    if args.action_model is not None:
        action_model_dir = os.path.join(project_root, 'models', 'latent_action', args.action_model)
        action_model = load_model(action_model_dir, device=device)

        data = collect_states_and_actions(model, action_model, env, model.action_dim, args.length, args.batch_size, args.num_trajectories, device=device)
        save_path = os.path.join(project_root, 'data', 'state_pairs', f'minigrid-{args.length}-{args.num_trajectories}-actions.pt')
    else:
        data = collect_states(model, env, model.action_dim, args.length, args.batch_size, args.num_trajectories, device=device)

        save_path = os.path.join(project_root, 'data', 'state_pairs', f'minigrid-{args.length}-{args.num_trajectories}.pt')
    save_state_transitions(data, save_path)