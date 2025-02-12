import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

def reparameterize_gaussian(mu, logvar):
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std

def reparameterize_categorical(logits, temperature=1.0, straight_through=False):
    if straight_through:
        p = D.OneHotCategoricalStraightThrough(logits=logits)
    else:
        p = D.RelaxedOneHotCategorical(temperature, logits=logits)
    return p.rsample()
