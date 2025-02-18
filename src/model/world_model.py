import torch
from torch import nn
import torch.nn.functional as F

from src.model.encoder import ConvEncoder, ConvEncoderCategorical
from src.model.decoder import ConvDecoder
import os
import src.model.transition as transition

class WorldModel(nn.Module):
    def __init__(self, 
                 encoder,
                 decoder,
                 transition,
                 latent_handler,
                 latent_size,
                 steps=5,
                 reset_hidden=True):
        
        super(WorldModel, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.transition_model = transition
        self.latent_handler = latent_handler
        self.latent_size = latent_size
        self.steps = steps
        self.current_step = 0
        self.action_dim = transition.action_dim
        self.reset_hidden = reset_hidden

    def forward(self, x, *h):
        h = torch.cat(h, dim=-1) if h else None
        dist = self.encoder(x, h)
        z = self.latent_handler.reparameterize(dist)
        x_hat = self.decoder(z, h)
        return x_hat, z, dist
    
    def decode(self, z, *h):
        h = torch.cat(h, dim=-1) if h else None
        return self.decoder(z, h)

    def transition(self, z, a, h=None):
        if h is not None:
            dist, h = self.transition_model(z, a, h)
        else:
            dist = self.transition_model(z, a)
        z_next = self.latent_handler.reparameterize(dist)
        return z_next, dist, h
    
    def zero_latent(self, batch_size, device='cpu'):
        return self.latent_handler.zero_latent(batch_size, self.latent_size, device=device)
    
    def zero_hidden(self, batch_size):
        return self.transition_model.zero_hidden(batch_size)
    
    def zero_prior(self, batch_size, device='cpu'):
        return self.latent_handler.zero_prior(batch_size, self.latent_size, device=device)
    
    def save_model(self, root, name):
        path = os.path.join(root, name)
        os.makedirs(root, exist_ok=True)
        torch.save(self.state_dict(), path)

    def rollout(self, x, a_seq, h=None):
        dist = self.encoder(x, h)
        z = self.latent_handler.reparameterize(dist)
        z_list = [z]
        dist_list = [dist]
        for a in a_seq:
            z, dist, h = self.transition(z, a, h)
            z_list.append(z)
            dist_list.append(dist)
        return z_list, dist_list
    
    def rollout_imagination(self, x, a_seq, h=None):
        x_hat, z, dist = self(x, h)
        z_list = [z]
        dist_list = [dist]
        recon_list = [x_hat]
        for a in a_seq:
            z, dist, h = self.transition(z, a, h)
            x_hat = self.decoder(z, h)
            z_list.append(z)
            dist_list.append(dist)
            recon_list.append(x_hat)
        return z_list, dist_list, recon_list

    def is_terminal(self, t):
        return (t+1) % self.steps == 0
    
    def process_hidden_state(self, t, h, batch_size):
        if t % self.steps == 0 and self.reset_hidden:
            return self.zero_hidden(batch_size)
        return h

            