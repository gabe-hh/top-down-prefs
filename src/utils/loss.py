import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from src.utils.utils import logits2categorical, scale_onehot

def kld_gaussian(mu1, logvar1, mu2, logvar2):
    return 0.5 * torch.sum(logvar2 - logvar1 + (logvar1.exp() + (mu1 - mu2).pow(2)) / logvar2.exp() - 1)

def kld_categorical(logits, prior_logits):
    q = F.softmax(logits, dim=-1)
    p = F.softmax(prior_logits, dim=-1)
    return torch.sum(q * (torch.log(q + 1e-8) - torch.log(p + 1e-8)))

def mse_loss(tgt, pred):
    return F.mse_loss(pred, tgt, reduction='sum')

def ce_loss(logits, tgt_onehot):
    logits = logits.permute(0, 2, 1)
    tgt_indicies = logits2categorical(tgt_onehot)
    return F.cross_entropy(logits, tgt_indicies, reduction='sum')