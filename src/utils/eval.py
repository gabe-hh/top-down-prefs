import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import wandb
import os

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

def generate_low_transition_predictions(model, env, a_indices, a_onehot, device='cuda', name='Low-level transition predictions', root='./', wandb_log=True, qty=1):
    true_obs, predictions = rollout_low_transition_predictions(model, env, a_indices, a_onehot, device)
    batch_size = true_obs[0].shape[0]
    qty = min(batch_size, qty)
    for i in range(qty):
        plot_low_transition_model_predictions(true_obs, predictions, i, name + f'_{i}', root, wandb_log)

    return true_obs, predictions
