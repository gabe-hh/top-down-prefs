import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.base import MLP
from src.utils.utils import logits2categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, layers, action_dim, action_classes=None, aux_dim=0):
        super(PolicyNetwork, self).__init__()
        self.action_classes = action_classes
        self.input_dim = state_dim + aux_dim
        output_dim = action_dim if action_classes is None else action_dim * action_classes
        self.model = MLP(self.input_dim, layers, output_dim, activation=F.relu, output_activation=None)

    def forward(self, x, *aux):
        if x.size(-1) != self.input_dim:
            batch_shape = x.shape[:-2]
            x = x.reshape(*batch_shape, -1)
        if aux:
            x = torch.cat([x, *aux], dim=-1)
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, x.shape[-1])
        logits = self.model(x)

        logits = logits.view(*batch_shape, -1)
        if self.action_classes is not None:
            logits = logits.view(logits.shape[0], -1, self.action_classes)
        p_a = F.softmax(logits, dim=-1)
        p_a_D = logits2categorical(logits)
        return logits,p_a,p_a_D