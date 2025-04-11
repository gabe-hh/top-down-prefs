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

def onehot2indicies(onehot):
    return torch.argmax(onehot, dim=-1)

def get_random_action(action_dim, batch_size, device='cuda'):
    action = torch.randint(0, action_dim, (batch_size,))
    action_tensor = F.one_hot(action, action_dim).float().to(device)
    return action, action_tensor

def get_random_action_sequence(length, action_dim, batch_size, device='cuda'):
    actions = torch.randint(0, action_dim, (length, batch_size))
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
    # Handle any tensor shape by working with the channels
    # which are always the third-last dimension
    channel_dim = img.dim() - 3  # Channel dimension index
    
    if channel_dim < 0:
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected at least 3D tensor (C,H,W).")
    
    # Extract RGB channels using dynamic indexing
    # This works for any number of batch dimensions
    red = img.select(channel_dim, 0).unsqueeze(channel_dim)
    green = img.select(channel_dim, 1).unsqueeze(channel_dim)
    blue = img.select(channel_dim, 2).unsqueeze(channel_dim)
    
    # Create the mask (goal is green areas)
    mask = ((green > green_threshold) & 
            (green > red + diff) & 
            (green > blue + diff)).to(img.dtype)
    
    # Concatenate along the channel dimension
    img_with_mask = torch.cat([img, mask], dim=channel_dim)
    return img_with_mask

def get_goal_mask(img):
    """
    Returns the goal mask from the image. Supports images with any number of batch dimensions.
    The mask is always the last channel.
    
    Parameters:
      img (Tensor): Input image tensor with channels in the third-last dimension.
      
    Returns:
      Tensor: The mask tensor.
    """
    if img.dim() < 3:
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected at least 3D tensor.")
    
    channel_dim = img.dim() - 3
    # Select the last channel
    return img.select(channel_dim, img.size(channel_dim)-1).unsqueeze(channel_dim)

def generate_goal_mask(img, green_threshold=0.5, diff=0.1):
    """
    Generates the goal mask from an unmasked image by computing the mask and returning only the mask.
    
    Parameters:
        img (Tensor): Input image tensor with channels in the third-last dimension.
        green_threshold (float): Threshold for the green channel.
        diff (float): Minimum difference for detecting a goal.
        
    Returns:
        Tensor: The computed goal mask.
    """
    return get_goal_mask(add_goal_mask(img, green_threshold, diff))

def separate_goal_mask(img):
    """
    Separates the goal mask from the image. Supports images with any number of batch dimensions.
    
    Parameters:
      img (Tensor): Input image tensor with channels in the third-last dimension.
      
    Returns:
      tuple: (img_without_mask, mask)
    """
    if img.dim() < 3:
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected at least 3D tensor.")
    
    channel_dim = img.dim() - 3
    mask = img.select(channel_dim, img.size(channel_dim)-1).unsqueeze(channel_dim)
    img_without_mask = img.narrow(channel_dim, 0, img.size(channel_dim)-1)
    
    return img_without_mask, mask

def attach_goal_mask(img, mask):
    """
    Attaches the goal mask to the image. Supports tensors of any shape by always concatenating
    along the channel dimension (third from last dimension).
    
    Parameters:
      img (Tensor): Input image tensor.
      mask (Tensor): Mask tensor to attach.
      
    Returns:
      Tensor: Image tensor with mask attached.
    """
    if img.dim() < 3:
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected at least 3D tensor.")
    
    # Always concatenate along the channel dimension (third from last)
    channel_dim = img.dim() - 3
    return torch.cat([img, mask], dim=channel_dim)

def remove_goal_mask(img):
    """
    Removes the goal mask from the image. Supports images with any number of batch dimensions.
    
    Parameters:
      img (Tensor): Input image tensor with channels in the third-last dimension.
      
    Returns:
      Tensor: Image tensor without the mask channel.
    """
    if img.dim() < 3:
        raise ValueError(f"Unsupported image shape: {img.shape}. Expected at least 3D tensor.")
    
    channel_dim = img.dim() - 3
    return img.narrow(channel_dim, 0, img.size(channel_dim)-1)