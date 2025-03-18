from torch.utils.data import Dataset
import torch
import numpy as np
from src.model.world_model import WorldModel
from src.utils.utils import obs2tensor, get_random_action
from src.utils.eval import plot_img_comparison_batch
import tqdm
import os

class StateTransitionDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'initial_z': torch.tensor(item['initial_z']) if isinstance(item['initial_z'], np.ndarray) else item['initial_z'],
            'initial_dist': tuple(torch.tensor(d) if isinstance(d, np.ndarray) else d for d in item['initial_dist']),
            'initial_h': torch.tensor(item['initial_h']) if isinstance(item['initial_h'], np.ndarray) else item['initial_h'],
            'final_z': torch.tensor(item['final_z']) if isinstance(item['final_z'], np.ndarray) else item['final_z'],
            'final_dist': tuple(torch.tensor(d) if isinstance(d, np.ndarray) else d for d in item['final_dist']),
            'final_h': torch.tensor(item['final_h']) if isinstance(item['final_h'], np.ndarray) else item['final_h']
        }

class StateTransitionsActionsDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        return item

def collect_states(model:WorldModel, env, action_dim, trajectory_length, batch_size, num_trajectories, device='cuda', reset_low=True):
    obs, _ = env.reset()
    collected_data = []
    hi_timestep = model.steps
    for _ in tqdm.trange(num_trajectories, desc="Collecting trajectories"):
        h = model.zero_hidden(batch_size).to(device)
        for t in range(trajectory_length):
            with torch.no_grad():
                #print(f"Collecting state {t}")
                obstensor = obs2tensor(obs['image'], device=device)
                terminal = model.is_terminal(t)
                action, action_tensor = get_random_action(action_dim, batch_size, device=device)

                x_hat, z, dist = model(obstensor, h)

                dist = tuple(d.cpu() for d in dist)

                _,_,h = model.transition(z, action_tensor, h)

                if t % hi_timestep == 0:
                    #print(f"Setting initial states at timestep {t}")
                    initial_z = z.clone()
                    initial_dist = dist
                    initial_h = h.clone()
                elif terminal:
                    #print(f"Setting final states at timestep {t}")
                    final_z = z.clone()
                    final_dist = dist
                    final_h = h.clone()
                    for b in range(batch_size):
                        transition = {
                            'initial_z': initial_z[b].cpu().numpy(),
                            'initial_dist': tuple(d[b].numpy() for d in initial_dist),
                            'initial_h': initial_h[b].cpu().numpy(),
                            'final_z': final_z[b].cpu().numpy(), 
                            'final_dist': tuple(d[b].numpy() for d in final_dist),
                            'final_h': final_h[b].cpu().numpy()
                        }
                        collected_data.append(transition)

                if not terminal:
                    obs,_,_,_,_ = env.step(action)
                else:
                    #print(f"Terminal state reached at timestep {t}")
                    h = model.process_hidden_state(t, h, batch_size)

    print(f'Collected {len(collected_data)} state transitions')
    return collected_data

def collect_states_and_actions(model:WorldModel, action_model, env, action_dim, trajectory_length, batch_size, num_trajectories, device='cuda', reset_low=True):
    obs, _ = env.reset()
    all_trajectories = []
    hi_timestep = model.steps

    for _ in tqdm.trange(num_trajectories, desc="Collecting trajectories"):
        h = model.zero_hidden(batch_size).to(device)
        # Create an empty list for each trajectory in the current batch.
        trajectory_batches = [[] for _ in range(batch_size)]
        
        for t in range(trajectory_length):
            with torch.no_grad():
                obstensor = obs2tensor(obs['image'], device=device)
                terminal = model.is_terminal(t)
                action, action_tensor = get_random_action(action_dim, batch_size, device=device)
                x_hat, z, dist = model(obstensor, h)
                dist = tuple(d.cpu() for d in dist)
                _, _, h = model.transition(z, action_tensor, h)
                
                if t % hi_timestep == 0:
                    # Set the start of a new segment.
                    initial_z = z.clone()
                    initial_dist = dist
                    initial_h = h.clone()
                
                elif terminal:
                    # End current segment.
                    final_z = z.clone()
                    final_dist = dist
                    final_h = h.clone()
                    
                    # Get the action information from the action model.
                    _, _, sampled_action, action_dist = action_model(initial_z, final_z, initial_h, final_h)
                    action_dist = tuple(d.cpu() for d in action_dist)
                    
                    # For each element in the batch, append the transition to its trajectory.
                    for b in range(batch_size):
                        transition = {
                            'initial_z': initial_z[b].cpu().numpy(),
                            'initial_dist': tuple(d[b].numpy() for d in initial_dist),
                            'initial_h': initial_h[b].cpu().numpy(),
                            'final_z': final_z[b].cpu().numpy(),
                            'final_dist': tuple(d[b].numpy() for d in final_dist),
                            'final_h': final_h[b].cpu().numpy(),
                            'action_dist': tuple(d[b].numpy() for d in action_dist),
                            'sampled_action': sampled_action[b].cpu().numpy()
                        }
                        trajectory_batches[b].append(transition)
                
                if not terminal:
                    obs, _, _, _, _ = env.step(action)
                else:
                    h = model.process_hidden_state(t, h, batch_size)
        
        # Add each batch trajectory (which is a list of transitions) to all_trajectories.
        all_trajectories.extend(trajectory_batches)
    
    print(f'Collected {len(all_trajectories)} trajectories with latent actions')
    return all_trajectories

def save_state_transitions(data, path):
    print(f'Saving state transitions to {path}')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)

if __name__ == '__main__':
    dataset = StateTransitionDataset('data/state_pairs/mini-5-2000.pt')
    print(len(dataset))