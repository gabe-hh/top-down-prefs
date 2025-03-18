import torch
import torch.distributions as D
import torch.nn.functional as F

import numpy as np

from src.utils.utils import logits2categorical

def compute_efe(model, z, a, h, o_goal, samples=10):
    # Add batch dimension to z if it doesn't exist
    # if len(z.shape) == 2:  # [latent_dim, categories]
    #     z = z.unsqueeze(0).expand(samples, -1, -1)
    # elif len(z.shape) == 3 and z.shape[0] == 1:  # [1, latent_dim, categories]
    #     z = z.expand(samples, -1, -1)
    z_next, dist, h_next = model.transition(z, a, h)
    p_z_logits, _ = dist
    p_z = logits2categorical(p_z_logits)
    #z_next = model.reparameterize(p_z_logits)
    p_o = model.decode(z_next, h_next)
    _, z_next, dist = model(p_o, h_next)
    q_z_logits, _ = dist
    q_z = logits2categorical(q_z_logits)

    H_p_z = p_z.entropy().sum(dim=1).mean()
    H_q_z = q_z.entropy().sum(dim=1).mean()

    # Calculate log p(o|z,a)
    log_p_o = F.mse_loss(o_goal, p_o, reduction='none').sum(dim=[1, 2, 3]).mean()

    efe = ( H_q_z - H_p_z ) + log_p_o

    return efe, H_p_z, H_q_z, log_p_o

import torch
import torch.nn.functional as F

def cem_planning(model, initial_state, hidden_state, o_goal,
                 planning_horizon=9, num_samples=1000, num_elites=100, num_iterations=10,
                 device='cpu'):
    """
    CEM planning for categorical actions with batch size = 1.
    
    Args:
        model: Your world model.
        initial_state: Initial latent state with shape [1, latent_dim].
        hidden_state: Initial hidden state (if applicable) with shape [1, ...].
        o_goal: Goal observation tensor with shape [1, ...].
        planning_horizon: Number of time steps to plan over.
        num_samples: Number of candidate sequences to sample per iteration.
        num_elites: Number of top sequences used to update the distribution.
        num_iterations: Number of CEM iterations.
        device: The device on which to run the planning ('cpu' or 'cuda').
        
    Returns:
        planned_action: The first action (as an integer) of the best candidate sequence.
        best_sequence: The full candidate sequence (as a tensor of action indices).
    """
    # Move model and inputs to the desired device.
    model.to(device)
    initial_state = initial_state.to(device)
    if hidden_state is not None:
        hidden_state = hidden_state.to(device)
    o_goal = o_goal.to(device)
    
    action_dim = model.action_dim

    # Initialize a categorical distribution for each time step with uniform probabilities.
    # Shape: [planning_horizon, action_dim]
    action_probs = torch.ones(planning_horizon, action_dim, device=device) / action_dim

    for iteration in range(num_iterations):
        candidate_sequences = []
        candidate_efes = []

        # Sample candidate sequences from the current distribution.
        for _ in range(num_samples):
            seq = []
            for t in range(planning_horizon):
                # Sample a single action index from the distribution at time step t.
                action = torch.multinomial(action_probs[t], num_samples=1).item()
                seq.append(action)
            candidate_sequences.append(torch.tensor(seq, device=device))
        candidate_sequences = torch.stack(candidate_sequences)  # [num_samples, planning_horizon]

        # Evaluate each candidate sequence.
        for candidate in candidate_sequences:
            # Clone the initial latent state and hidden state for this rollout.
            z = initial_state.clone()  # Shape: [1, latent_dim]
            h = hidden_state.clone() if hidden_state is not None else None
            total_efe = torch.tensor(0.0, device=device)

            # Roll out the candidate sequence.
            for t, action_index in enumerate(candidate):
                # Create a one-hot vector for the current action.
                a = torch.zeros(action_dim, device=device)
                a[action_index] = 1.0

                # Compute one-step expected free energy.
                efe, _, _, _ = compute_efe(model, z, a, h, o_goal)
                total_efe += efe

                # Update the latent state and hidden state via the model's transition.
                z, _, h = model.transition(z, a, h)
            candidate_efes.append(total_efe)
        candidate_efes = torch.stack(candidate_efes)  # [num_samples]

        # Select elite candidates (lowest cumulative EFE).
        elite_indices = candidate_efes.topk(num_elites, largest=False).indices
        elite_sequences = candidate_sequences[elite_indices]

        # Update the action distribution at each time step based on elite candidates.
        new_action_probs = []
        for t in range(planning_horizon):
            counts = torch.zeros(action_dim, device=device)
            for seq in elite_sequences:
                counts[seq[t]] += 1
            if counts.sum() > 0:
                new_probs = counts / counts.sum()
            else:
                new_probs = torch.ones(action_dim, device=device) / action_dim
            new_action_probs.append(new_probs)
        action_probs = torch.stack(new_action_probs)  # [planning_horizon, action_dim]

        print(f"Iteration {iteration+1}, average elite EFE: {candidate_efes[elite_indices].mean().item():.4f}")

    # Select the best candidate sequence from the final iteration.
    best_index = candidate_efes.argmin().item()
    best_sequence = candidate_sequences[best_index]
    planned_action = best_sequence[0].item()  # Execute the first action.

    return planned_action, best_sequence

# Example usage:
# initial_z = model.zero_latent(batch_size=1).squeeze(0)
# hidden = model.zero_hidden(batch_size=1)
# o_goal = some_goal_observation_tensor  # Make sure its dimensions match what your model expects
# action, sequence = cem_planning(model, initial_z, hidden, o_goal)
