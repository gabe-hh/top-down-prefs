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

def add_goal_mask(img, green_threshold=0.5, diff=0.1):
    """
    Adds a goal mask to the image by detecting green pixels based on a threshold and relative difference.
    Supports both non-batched ([C, H, W]) and batched ([B, C, H, W]) images.
    
    The mask is computed by checking if the green channel is above a threshold and significantly higher
    than both the red and blue channels.
    
    Parameters:
      img (Tensor): Input image tensor with channels in the first dimension.
      green_threshold (float): Minimum value for the green channel to be considered as goal.
      diff (float): Minimum difference by which the green channel must exceed red and blue channels.
      
    Returns:
      Tensor: Image tensor concatenated with the newly computed mask.
    """
    if img.dim() == 3:
        # Non-batched image: shape [C, H, W]
        red = img[0:1]
        green = img[1:2]
        blue = img[2:3]
        mask = ((green > green_threshold) & (green > red + diff) & (green > blue + diff)).to(img.dtype)
        img_with_mask = torch.cat([img, mask], dim=0)
    elif img.dim() == 4:
        # Batched image: shape [B, C, H, W]
        red = img[:, 0:1]
        green = img[:, 1:2]
        blue = img[:, 2:3]
        mask = ((green > green_threshold) & (green > red + diff) & (green > blue + diff)).to(img.dtype)
        img_with_mask = torch.cat([img, mask], dim=1)
    else:
        raise ValueError("Unsupported image shape")
    return img_with_mask

def get_goal_mask(img):
    """
    Returns the goal mask from the image. Supports both non-batched ([C, H, W])
    and batched ([B, C, H, W]) images.
    """
    if img.dim() == 3:
        # Non-batched image
        mask = img[-1:]
    elif img.dim() == 4:
        # Batched image
        mask = img[:, -1:]
    else:
        raise ValueError("Unsupported image shape")
    return mask

def generate_goal_mask(img, green_threshold=0.5, diff=0.1):
    """
    Generates the goal mask from an unmasked image by computing the mask and returning only the mask.
    
    Parameters:
        img (Tensor): Input image tensor with channels in the first dimension for non-batched images
                    or B, C, H, W for batched images.
        green_threshold (float): Threshold for the green channel.
        diff (float): Minimum difference for detecting a goal.
        
    Returns:
        Tensor: The computed goal mask.
    """
    return get_goal_mask(add_goal_mask(img, green_threshold, diff))

def separate_goal_mask(img):
    """
    Separates the goal mask from the image. Supports both non-batched ([C, H, W])
    and batched ([B, C, H, W]) images.
    """
    if img.dim() == 3:
        # Non-batched image
        mask = img[-1:]
        img_without_mask = img[:-1]
    elif img.dim() == 4:
        # Batched image
        mask = img[:, -1:]
        img_without_mask = img[:, :-1]
    else:
        raise ValueError("Unsupported image shape")
    return img_without_mask, mask

def remove_goal_mask(img):
    """
    Removes the goal mask from the image. Supports both non-batched ([C, H, W])
    and batched ([B, C, H, W]) images.
    """
    if img.dim() == 3:
        # Non-batched image
        img_without_mask = img[:-1]
    elif img.dim() == 4:
        # Batched image
        img_without_mask = img[:, :-1]
    else:
        raise ValueError("Unsupported image shape")
    return img_without_mask