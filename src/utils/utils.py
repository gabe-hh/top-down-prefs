import torch
import torch.distributions as D
import torch.nn.functional as F

def logits2categorical(logits):
    return D.OneHotCategorical(logits=logits)

def scale_onehot(onehot):
    return 2*onehot - 1

def obs2tensor(obs, device='cuda'):
    tensor = torch.tensor(obs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.
    return tensor.to(device)

def indicies2onehot(indicies, batch_size=64, indicies_dim=4):
    onehot = torch.zeros(batch_size, indicies_dim)
    onehot[torch.arange(batch_size), indicies] = 1
    return onehot

def get_random_action(action_dim, batch_size, device='cuda'):
    action = torch.randint(0, action_dim-1, (batch_size,))
    action_tensor = F.one_hot(action, action_dim).float().to(device)
    return action, action_tensor

def get_random_action_sequence(length, action_dim, batch_size, device='cuda'):
    actions = torch.randint(0, action_dim-1, (length, batch_size))
    action_tensors = F.one_hot(actions, action_dim).float().to(device)
    return actions, action_tensors