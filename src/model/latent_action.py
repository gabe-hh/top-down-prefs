import torch
from torch import nn
from src.model.encoder import DenseEncoder, DenseEncoderCategorical
from src.model.decoder import DenseDecoder
import src.model.utils as utils
import os

class LatentActionModel(nn.Module):
    def __init__(self,
                 state_dim,
                 action_dim,
                 encoder_layers,
                 decoder_layers,
                 latent_handler,
                 deterministic_dim=0,
                 include_initial_deterministic=False,
                 predict_deterministic=False,
                 action_classes=None,
                 num_classes=None):
        super(LatentActionModel, self).__init__()

        self.predict_deterministic = predict_deterministic
        self.include_initial_deterministic = include_initial_deterministic
        self.state_dim = state_dim
        self.latent_handler = latent_handler

        if include_initial_deterministic:
            initial_deterministic_size = deterministic_dim
            deterministic_input_size = deterministic_dim * 2
        else:
            initial_deterministic_size = 0
            deterministic_input_size = deterministic_dim

        if num_classes is None:
            self.state_type = 'continuous'
            self.input_size = state_dim * 2 + deterministic_input_size
            state_size = state_dim
            self.output_size = state_dim
        else:
            self.state_type = 'categorical'
            self.num_classes = num_classes
            self.input_size = state_dim * num_classes * 2 + deterministic_input_size
            state_size = state_dim * num_classes
            self.output_size = state_dim * num_classes

        if predict_deterministic:
            self.output_size += deterministic_dim

        if action_classes is not None:
            self.encoder = DenseEncoderCategorical(self.input_size, encoder_layers, action_classes, action_dim)
            self.decoder = DenseDecoder(action_dim * action_classes + state_size, decoder_layers, self.output_size)
            self.type = 'categorical'
        else:
            self.encoder = DenseEncoder(self.input_size, encoder_layers, action_dim)
            self.type = 'normal'
            self.decoder = DenseDecoder(action_dim + state_size + initial_deterministic_size, decoder_layers, self.output_size)

        print(f"Encoder input size: {self.input_size}, output size: {self.encoder.output_size}")
        print(f"Decoder input size: {self.decoder.input_dim}, output size: {self.decoder.output_size}")

    def encode_action(self, s_initial, s_final, d_initial=None, d_final=None):
        s_initial = s_initial.view(s_initial.size(0), -1)
        s_final = s_final.view(s_final.size(0), -1)
        x = torch.cat([s_initial, s_final], dim=-1)
        if self.include_initial_deterministic:
            d = torch.cat([d_initial, d_final], dim=-1)
        dist = self.encoder(x, d)
        a = self.latent_handler.reparameterize(dist)
        return a, dist
    
    def decode(self, a, s_initial, d_initial=None):
        s_initial = s_initial.view(s_initial.size(0), -1)
        a = a.view(a.size(0), -1)
        x = torch.cat([a, s_initial], dim=-1)
        out = self.decoder(x, d_initial)
        if self.predict_deterministic:
            s_hat, d_hat = torch.split(out, [self.output_size - s_initial.size(1), s_initial.size(1)], dim=-1)
        else:
            s_hat = out
            d_hat = None
        
        if self.state_type == 'categorical':
            s_hat = s_hat.view(-1, self.state_dim, self.num_classes)
        return s_hat, d_hat
        
    def forward(self, s_initial, s_final, d_initial=None, d_final=None):
        a, dist = self.encode_action(s_initial, s_final, d_initial, d_final)
        s_hat, d_hat = self.decode(a, s_initial, d_initial)
        return s_hat, d_hat, a, dist
    
    def save_model(self, root, name):
        path = os.path.join(root, name)
        os.makedirs(root, exist_ok=True)
        torch.save(self.state_dict(), path)

    def get_reconstructed_state(self, s_hat, sample=True):
        if self.state_type == 'categorical':
            if sample:
                # Sample from the distribution defined by the logits
                dist = torch.distributions.OneHotCategorical(logits=s_hat)
                return dist.sample()
            else:
                # Use the argmax to get a deterministic output
                return torch.argmax(s_hat, dim=-1)
        else:
            return s_hat