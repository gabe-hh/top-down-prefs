import torch
import torch.distributions as D
import torch.nn.functional as F

import numpy as np

from src.utils.utils import logits2categorical
from src.utils.loss import kld_categorical

def compute_ig_sample(model, z, a, h):
    z_next, dist, h_next = model.transition(z, a, h)
    p_z_logits, _ = dist
    p_z = logits2categorical(p_z_logits)
    #z_next = model.reparameterize(p_z_logits)
    p_o,p_o_mask = model.decode(z_next, h_next)
    p_o_concat = torch.cat([p_o, p_o_mask], dim=1)
    _, z_next, dist = model(p_o_concat, h_next)
    q_z_logits, _ = dist
    q_z = logits2categorical(q_z_logits)

    ig = -kld_categorical(q_z_logits, p_z_logits) # TODO: This assumes categorical, make it general

    return ig

def approximate_ig_mc(model, z, a, h, samples=10):
    ig = 0
    with torch.no_grad():
        for _ in range(samples):
            ig += compute_ig_sample(model, z, a, h)
        ig /= samples
    return ig

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

def compute_log_p_o(recon, preference, goal_is_bernoilli=False, reduce=True, average_over_pixels=False):
    if goal_is_bernoilli:
        # First ensure that the preference is the same shape as the reconstructions
        preference = preference.expand_as(recon)
        # Compute the binary cross entropy loss
        log_p_o = F.binary_cross_entropy(recon, preference, reduction='none')
        # print(f"log_p_o shape: {log_p_o.shape}, goal shape: {preference.shape}, recon shape: {recon.shape}")
        # print(f"recon max: {recon.max()}, recon min: {recon.min()}")
        # print(f"preference max: {preference.max()}, preference min: {preference.min()}")
        # Sum over the last dimensions (spatial and channel dimensions) and mean over batch dimensions
        if average_over_pixels:
            log_p_o = log_p_o.mean(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        else:
            log_p_o = log_p_o.sum(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        if reduce:
            log_p_o = log_p_o.mean()
        #print(f"log_p_o shape: {log_p_o.shape} log_p_o value: {log_p_o}")
    else:
        log_p_o = F.mse_loss(recon, preference, reduction='none')
        # Sum over the last dimensions (spatial and channel dimensions) and mean over batch dimensions
        log_p_o = log_p_o.sum(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        if reduce:
            log_p_o = log_p_o.mean()
    return log_p_o

def compute_EFE_igpv(posterior:D.Distribution, posterior_given_o:D.Distribution, recon, preference, goal_is_bernoilli=False, goal_precision=1., reduce=True, average_over_pixels=False, average_kl=False):
    """
    Compute the EFE in terms of information gain (IG) and pragmatic value (PV).
    Inputs can be batched over an arbitrary number of dimensions.
    """
    
    # Information Gain (states)
    kl = D.kl_divergence(posterior, posterior_given_o)
    # Handle potential inf values in KL divergence
    kl = torch.clamp(kl, max=1e6) 
    if average_kl:
        ig = kl.mean(dim=-1)
    else:
        ig = kl.sum(dim=-1)
    if reduce:
        ig = ig.mean()

    # Pragmatic Value (observations)
    if goal_is_bernoilli:
        # First ensure that the preference is the same shape as the reconstructions
        preference = preference.expand_as(recon)
        # Compute the binary cross entropy loss
        log_p_o = F.binary_cross_entropy(recon, preference, reduction='none')
        # print(f"log_p_o shape: {log_p_o.shape}, goal shape: {preference.shape}, recon shape: {recon.shape}")
        # print(f"recon max: {recon.max()}, recon min: {recon.min()}")
        # print(f"preference max: {preference.max()}, preference min: {preference.min()}")
        # Sum over the last dimensions (spatial and channel dimensions) and mean over batch dimensions
        if average_over_pixels:
            log_p_o = log_p_o.mean(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        else:
            log_p_o = log_p_o.sum(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        if reduce:
            log_p_o = log_p_o.mean()
        #print(f"log_p_o shape: {log_p_o.shape} log_p_o value: {log_p_o}")
    else:
        log_p_o = F.mse_loss(recon, preference, reduction='none')
        # Sum over the last dimensions (spatial and channel dimensions) and mean over batch dimensions
        log_p_o = log_p_o.sum(dim=list(range(log_p_o.dim()-1, log_p_o.dim()-4, -1)))
        if reduce:
            log_p_o = log_p_o.mean()
    
    #log_p_o = 0.
    return - ig + goal_precision * log_p_o, ig, - log_p_o