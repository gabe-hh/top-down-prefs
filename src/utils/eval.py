import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import wandb
import os
import math
import dill
from src.model.world_model import high_tick
from src.train.utils import obs2tensor

# Plotting functions
def plot_img_comparison(img1, img2, title1='Image 1', title2='Image 2', name='Image Comparison', root='./', wandb_log=True):
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy().transpose(1, 2, 0)
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy().transpose(1, 2, 0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img1)
    ax1.set_title(title1)
    ax2.imshow(img2)
    ax2.set_title(title2)

    path = os.path.join(root, name + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)

    plt.savefig(path)
    plt.close()
    if wandb_log and wandb.run is not None:
        wandb.log({name: wandb.Image(path)})

def plot_img_comparison_batch(img1, img2, title1='Image 1', title2='Image 2', name='Image Comparison', root='./', wandb_log=True, limit=None):
    batch_size = img1.size(0)
    if limit is not None:
        batch_size = min(batch_size, limit)
    for i in range(batch_size):
        plot_img_comparison(img1[i], img2[i], title1, title2, name + f'_{i}', root, wandb_log)

def plot_distribution_comparison(logits1, logits2, 
                                 title1='Distribution 1', title2='Distribution 2', 
                                 name='Distribution Comparison', root='./', wandb_log=True, limit=None, max_cols=6):
    """
    Compare non-temporal distributions across dimensions.
    
    Each dimension is shown as a tile with two vertically stacked axes:
      - Top: true distribution
      - Bottom: predicted distribution
      
    Within each tile the two axes are drawn very close together, while the tiles are well separated.
    """
    batch_size, size, num_classes = logits1.shape
    softmax_probs1 = torch.nn.functional.softmax(logits1, dim=-1)
    softmax_probs2 = torch.nn.functional.softmax(logits2, dim=-1)
    
    if limit is not None:
        batch_size = min(batch_size, limit)
    
    # Determine grid dimensions for the tiles.
    n_cols = min(size, max_cols)
    n_rows = math.ceil(size / n_cols)
    
    for i in range(batch_size):
        # Set up the overall figure.
        # Adjust the figure size as desired (width per column, height per row)
        fig = plt.figure(figsize=(4 * n_cols, 3 * n_rows))
        # Outer GridSpec: controls spacing between different tiles.
        outer_gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.7, wspace=0.4)
        
        for dim in range(size):
            # Determine tile (row, col) for this dimension.
            row = dim // n_cols
            col = dim % n_cols
            # Create a nested GridSpec for the tile: 2 rows, 1 column.
            # The hspace here is small so that the two axes (true & predicted) are very close.
            inner_gs = outer_gs[row, col].subgridspec(2, 1, hspace=0.6)
            
            # Top axis: True distribution
            ax_top = fig.add_subplot(inner_gs[0])
            true_dist = softmax_probs1[i, dim].cpu().detach().numpy().reshape(1, -1)
            ax_top.imshow(true_dist, aspect='auto', interpolation='nearest', cmap='gray')
            ax_top.set_title(f'{title1} - Dim {dim}', fontsize=10)
            ax_top.set_yticks([])
            ax_top.set_xticks([])
            ax_top.set_xlabel('Classes', fontsize=8)
            
            # Bottom axis: Predicted distribution
            ax_bot = fig.add_subplot(inner_gs[1])
            pred_dist = softmax_probs2[i, dim].cpu().detach().numpy().reshape(1, -1)
            ax_bot.imshow(pred_dist, aspect='auto', interpolation='nearest', cmap='gray')
            ax_bot.set_title(f'{title2} - Dim {dim}', fontsize=10)
            ax_bot.set_yticks([])
            ax_bot.set_xticks([])
            ax_bot.set_xlabel('Classes', fontsize=8)

        plt.tight_layout()
        path = os.path.join(root, name + f'_{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        
        if wandb_log and wandb.run is not None:
            wandb.log({f"{name}_{i}": wandb.Image(path)})

            
def plot_temporal_distribution_comparison(logits1_list, logits2_list, 
                                          title1='Distribution 1', title2='Distribution 2', 
                                          name='Temporal Distribution Comparison', root='./', wandb_log=True, limit=None, max_cols=4):
    """
    Compare temporal distributions across dimensions.
    
    For each dimension, two vertically stacked axes are created:
      - Top: temporal evolution of Distribution 1 (time vs. classes)
      - Bottom: temporal evolution of Distribution 2
      
    Within each tile the two plots are tightly stacked, while the overall grid separates the tiles.
    """
    # Stack the list of tensors into a tensor with a time dimension.
    logits1 = torch.stack(logits1_list, dim=1)  # [batch_size, time_steps, size, num_classes]
    logits2 = torch.stack(logits2_list, dim=1)
    
    batch_size, time_steps, size, num_classes = logits1.shape
    softmax_probs1 = torch.nn.functional.softmax(logits1, dim=-1)
    softmax_probs2 = torch.nn.functional.softmax(logits2, dim=-1)
    
    if limit is not None:
        batch_size = min(batch_size, limit)
    
    n_cols = min(size, max_cols)
    n_rows = math.ceil(size / n_cols)
    
    for i in range(batch_size):
        fig = plt.figure(figsize=(3 * n_cols, 4 * n_rows))
        outer_gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.7, wspace=0.4)
        
        for dim in range(size):
            row = dim // n_cols
            col = dim % n_cols
            inner_gs = outer_gs[row, col].subgridspec(2, 1, hspace=0.5)
            
            # For the temporal distributions, we transpose the array so that:
            # - x-axis represents time steps
            # - y-axis represents classes
            ax_top = fig.add_subplot(inner_gs[0])
            true_dist = softmax_probs1[i, :, dim].cpu().detach().numpy().T  # shape: (num_classes, time_steps)
            ax_top.imshow(true_dist, aspect='auto', interpolation='nearest', cmap='gray')
            ax_top.set_title(f'{title1} - Dim {dim}', fontsize=10)
            ax_top.set_xticks([])
            ax_top.set_yticks([])
            ax_top.set_xlabel('Time Steps', fontsize=8)
            ax_top.set_ylabel('Classes', fontsize=8)
            
            ax_bot = fig.add_subplot(inner_gs[1])
            pred_dist = softmax_probs2[i, :, dim].cpu().detach().numpy().T  # shape: (num_classes, time_steps)
            ax_bot.imshow(pred_dist, aspect='auto', interpolation='nearest', cmap='gray')
            ax_bot.set_title(f'{title2} - Dim {dim}', fontsize=10)
            ax_bot.set_xticks([])
            ax_bot.set_yticks([])
            ax_bot.set_xlabel('Time Steps', fontsize=8)
            ax_bot.set_ylabel('Classes', fontsize=8)
            
        plt.tight_layout()
        path = os.path.join(root, name + f'_{i}.png')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.savefig(path)
        plt.close()
        
        if wandb_log and wandb.run is not None:
            wandb.log({f"{name}_{i}": wandb.Image(path)})

def plot_low_transition_model_predictions(true_obs, predictions, batch_idx=0, 
                                          name='Low-level transition predictions', 
                                          root='./', wandb_log=True):
    """
    Create a triangular layout plot for a specific batch element.
    
    Layout:
      - Bottom row: true observations (blue border).
      - Row below: step indicators (condensed height).
      - Rows above: predictions.
         For each prediction row, the first image is labeled as "Posterior Reconstruction" (red border)
         and subsequent images are labeled as "Transition Prediction" (orange border).
    
    Both `true_obs` and `predictions` are lists where each element is either:
      - a single image of shape (H, W, C), or
      - a batched array of images of shape (B, H, W, C).
      
    Parameters
    ----------
    true_obs : list
        A list of true observation images. For batched data, each element should be of shape (B, H, W, C).
    predictions : list of lists
        Each element predictions[j] is a list of predicted images starting from true_obs[j].
        For batched data, each predicted image should be of shape (B, H, W, C).
    batch_idx : int, optional
        The batch index to visualize (default is 0).
    name : str, optional
        The name used to save the image.
    root : str, optional
        The root directory where the image will be saved.
    wandb_log : bool, optional
        Whether to log the image to wandb.
    """
    T = len(true_obs)            # Number of true observations (time steps)
    total_cols = T               # Each column corresponds to a time step
    # We have T-1 prediction rows (one for each starting point except the final observation),
    # plus 1 row for true obs and 1 for step labels.
    total_rows = T + 2           

    # Set height ratios: use a lower ratio for the label row.
    # For example, each image row gets a ratio of 1 and the label row gets 0.3.
    height_ratios = [1] * (T + 1) + [0.3]

    fig = plt.figure(figsize=(1.8 * total_cols, 1.8 * total_rows))
    gs = gridspec.GridSpec(total_rows, total_cols, height_ratios=height_ratios, 
                           wspace=0.1, hspace=0.1)

    # Define border colours.
    immediate_color = 'green'      # Immediate (posterior) reconstruction
    transition_color = 'red'       # Transition prediction
    observation_color = 'blue'  # True observation

    # -----------------------------
    # Plot the prediction rows.
    # -----------------------------
    # Predictions are only for the first T-1 true observations.
    true_obs_row_index = T  # True observations row will be at grid row T (0-indexed)
    for j in range(T-1):
        grid_row = true_obs_row_index - 1 - j  # rows counting downward from the top
        row_preds = predictions[j]  # Expected length: T - j.
        for k, pred_img in enumerate(row_preds):
            grid_col = j + k  # align predictions so that the prediction for time step j falls in column j
            ax = fig.add_subplot(gs[grid_row, grid_col])

            # If the image has a batch dimension (first axis), select the desired batch element.
            if pred_img.ndim == 4:
                img_to_plot = pred_img[batch_idx]
            else:
                img_to_plot = pred_img

            # Convert torch Tensors to numpy (and adjust channel order) if necessary.
            if isinstance(img_to_plot, torch.Tensor):
                img_to_plot = img_to_plot.cpu().detach().numpy().transpose(1, 2, 0)

            ax.imshow(img_to_plot, interpolation='none')
            ax.set_xticks([])
            ax.set_yticks([])

            # Set border: first image is the immediate reconstruction, others are transitions.
            border_color = immediate_color if k == 0 else transition_color
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)

            # Add a text label inside the axes.
            label_text = f"Posterior t={j+1}" if k == 0 else f"Transition t={j+k+1}"
            # Place the label at the bottom center of the image (inside the axes).
            ax.text(0.5, -0.15, label_text, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=11, color=border_color) 

    # -----------------------------
    # Plot the true observations row.
    # -----------------------------
    for i in range(T):
        ax = fig.add_subplot(gs[true_obs_row_index, i])
        obs_img = true_obs[i]
        if obs_img.ndim == 4:
            img_to_plot = obs_img[batch_idx]
        else:
            img_to_plot = obs_img

        if isinstance(img_to_plot, torch.Tensor):
            img_to_plot = img_to_plot.cpu().detach().numpy().transpose(1, 2, 0)
                                                                       
        ax.imshow(img_to_plot, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor(observation_color)
            spine.set_linewidth(3)
        
        label_text = f"Observation t={i+1}"
        # Place the label at the bottom center of the image (inside the axes).
        ax.text(0.5, -0.15, label_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=11, color=observation_color)

    # -----------------------------
    # Plot the step indicator row (below true observations) with condensed height.
    # -----------------------------
    label_row_index = true_obs_row_index + 1
    for i in range(T):
        ax = fig.add_subplot(gs[label_row_index, i])
        label = f"t={i+1}"
        ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=12)
        ax.axis('off')

    # Save the figure.
    path = os.path.join(root, name + '.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()

    if wandb_log:
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({name: wandb.Image(path)})
        except ImportError:
            pass  # wandb not available; skip logging.

def eval_high_trajectory(model_high, model_low, model_action, env, device='cuda', trajectory_length=50):
    probs = []
    predictions = []

    posteriors = []
    priors = []

    obs,_ = env.reset()
    obs_tensor = obs2tensor(obs['image']).to(device)
    batch_size = obs_tensor.shape[0]
    
    h_low = model_low.zero_hidden(batch_size, device)
    h_high = model_high.zero_hidden(batch_size, device)
    dist_high_next = None
    for T in range(trajectory_length):
        low_data = high_tick(model_low, env, h_low, obs, device, batch_size)
        initial_z_low = low_data['initial_z']
        initial_dist_low = low_data['initial_dist']
        initial_h_low = low_data['initial_h']
        final_z_low = low_data['final_z']
        final_dist_low = low_data['final_dist']
        final_h_low = low_data['final_h']
        h_low = low_data['h']
        obs = low_data['obs']

        combined_low_state = torch.cat([initial_z_low.view(batch_size, -1), initial_h_low], dim=-1)
        x_hat, z_high, dist_high = model_high(combined_low_state, h_high)
        x_hat = model_low.latent_handler.reshape_latent(x_hat, model_low.latent_size)
        with torch.no_grad():
            _,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
        
        if dist_high_next is not None:
            prior_logits,_ = dist_high_next
        else:
            prior_logits,_ = model_high.zero_prior(batch_size, device)

        z_high_next, dist_high_next, h_high = model_high.transition(z_high, a_high, h_high)
        
        logits_low,_ = initial_dist_low
        
        # TODO: All of this assumes categorical, it needs to be generalized
        logits_high,_ = dist_high
        posteriors.append(logits_high)
        priors.append(prior_logits)
        probs.append(logits_low)
        predictions.append(x_hat)

    return posteriors, priors, probs, predictions

def rollout_high_transition_predictions(model_high, model_low, model_action, env, a_indices, a_onehot, device='cuda', context_length=20):
    obs,_ = env.reset()
    obs_tensor = obs2tensor(obs['image']).to(device)
    batch_size = obs_tensor.shape[0]
    h_low = model_low.zero_hidden(batch_size, device)
    h_high = model_high.zero_hidden(batch_size, device)

    # Generate context
    for i in range(context_length):
        low_data = high_tick(model_low, env, h_low, obs, device, batch_size)
        initial_z_low = low_data['initial_z']
        initial_dist_low = low_data['initial_dist']
        initial_h_low = low_data['initial_h']
        final_z_low = low_data['final_z']
        final_dist_low = low_data['final_dist']
        final_h_low = low_data['final_h']
        h_low = low_data['h']
        obs = low_data['obs']

        combined_low_state = torch.cat([initial_z_low.view(batch_size, -1), initial_h_low], dim=-1)
        x_hat, z_high, dist_high = model_high(combined_low_state, h_high)
        x_hat = model_low.latent_handler.reshape_latent(x_hat, model_low.latent_size)
        with torch.no_grad():
            _,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
        
        z_high_next, _, h_high = model_high.transition(z_high, a_high, h_high)
    
    # Save copies of the hidden states after context
    h_high_copy = h_high.clone()
    h_low_copy = h_low.clone()

    # Make a low level trajectory in a copied env
    chunk_size = model_low.steps - 1
    num_complete_chunks = len(a_onehot) // chunk_size
    a_indices_chunks = [a_indices[i*chunk_size:(i+1)*chunk_size] for i in range(num_complete_chunks)]
    a_onehot_chunks = [a_onehot[i*chunk_size:(i+1)*chunk_size] for i in range(num_complete_chunks)]

    env_copy = env.clone()

    low_trajectory_data = []
    for a_indices_chunk, a_onehot_chunk in zip(a_indices_chunks, a_onehot_chunks):
        low_data = high_tick(model_low, env_copy, h_low, obs, device, batch_size, 
                            a_indices=a_indices_chunk, a_onehot=a_onehot_chunk)
        low_trajectory_data.append(low_data)
    
    # Infer the corresponding high-level actions
    high_actions = []
    for low_data in low_trajectory_data:
        initial_z_low = low_data['initial_z']
        final_z_low = low_data['final_z']
        initial_h_low = low_data['initial_h']
        final_h_low = low_data['final_h']
        with torch.no_grad():
            _,_,a_high, dist_action = model_action(initial_z_low, final_z_low, initial_h_low, final_h_low)
        high_actions.append(a_high)
    # Rollout the high-level model
    h_high = h_high_copy
    h_low = h_low_copy
    dist_high_next = None
    priors = []
    posteriors = []
    
    for i,a_indices_chunk in enumerate(a_indices_chunks):
        a_onehot_chunk = a_onehot_chunks[i]
        #a_high = high_actions[i]
        low_data = high_tick(model_low, env, h_low, obs, device, batch_size, a_indices=a_indices_chunk, a_onehot=a_onehot_chunk)
        combined_low_state = torch.cat([low_data['initial_z'].view(batch_size, -1), low_data['initial_h']], dim=-1)
        remaining_high_actions = torch.stack(high_actions[i:], dim=0)  # [seq_len, batch_size, action_dim]
        first_step, z_list, dist_list = model_high.rollout(combined_low_state, remaining_high_actions, h_high)
        _,_,h_high = model_high.transition(z_list[0], remaining_high_actions[0], h_high)
        _,posterior = first_step
        logits,_ = posterior
        priors.append(list(logits for logits,_ in dist_list)) 
        posteriors.append(logits)

    return posteriors, priors

# Evaluation functions
def rollout_low_transition_predictions(model, env, a_indices, a_onehot, device='cuda'):
    predictions = []
    obs,_ = env.reset()
    obs = obs['image']
    true_obs = [obs]
    batch_size = obs.shape[0]
    h = model.zero_hidden(batch_size).to(device)
    # For each starting point
    for i in range(len(a_onehot)):
        current_obs = true_obs[i]
        obs_tensor = obs2tensor(current_obs).to(device)        
        # Use actions from index i onwards
        remaining_a_sq =  a_onehot[i:]
        with torch.no_grad():
            z_list, dist_list, recon_list = model.rollout_imagination(obs_tensor, remaining_a_sq, h)
            _,_,h = model.transition(z_list[0], remaining_a_sq[0], h)
        predictions.append(recon_list)
        # Get next true observation if not at end
        if i < len(a_onehot):
            obs, _, _, _, _ = env.step(a_indices[i])
            obs = obs['image']
            true_obs.append(obs)

    return true_obs, predictions

def generate_high_trajectory_evaluation(model_high, model_low, model_action, env, device='cuda', name='High-level trajectory', root='./', wandb_log=True, qty=1):
    posteriors, priors, probs, predictions = eval_high_trajectory(model_high, model_low, model_action, env, device)
    batch_size = posteriors[0].shape[0]
    qty = min(batch_size, qty)
    plot_temporal_distribution_comparison(posteriors, priors, 'Posterior', 'Prior', name + '_traj_transitions', root, wandb_log, limit=qty)
    plot_temporal_distribution_comparison(probs, predictions, 'Low Posterior', 'High Reconstruction', name + '_traj_recon', root, wandb_log, limit=qty)

def generate_high_transition_predictions(model_high, model_low, model_action, env, a_indices, a_onehot, device='cuda', name='High-level transition predictions', root='./', wandb_log=True, qty=1):
    posteriors, priors = rollout_high_transition_predictions(model_high, model_low, model_action, env, a_indices, a_onehot, device)
    batch_size = posteriors[0].shape[0]
    qty = min(batch_size, qty)
    plot_temporal_distribution_comparison(posteriors, priors[0], 'Posterior', 'Prior Prediction', name, root, wandb_log, limit=qty)

def generate_low_transition_predictions(model, env, a_indices, a_onehot, device='cuda', name='Low-level transition predictions', root='./', wandb_log=True, qty=1):
    true_obs, predictions = rollout_low_transition_predictions(model, env, a_indices, a_onehot, device)
    batch_size = true_obs[0].shape[0]
    qty = min(batch_size, qty)
    for i in range(qty):
        plot_low_transition_model_predictions(true_obs, predictions, i, name + f'_{i}', root, wandb_log)

    return true_obs, predictions
