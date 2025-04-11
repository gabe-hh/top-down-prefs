import torch
from torch import nn
import torch.nn.functional as F

from src.model.base import Dense, CNN

class ConvEncoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 img_size,
                 conv_layers,
                 fc_layers, 
                 latent_dim, 
                 conv_act=F.relu,
                 fc_act=F.relu,
                 aux_size=0):
        
        super(ConvEncoder, self).__init__()
        self.conv_act = conv_act
        self.fc_act = fc_act
        
        self.cnn = CNN(in_channels, img_size, conv_layers, activation=conv_act)
        self.dense = Dense(self.cnn.output_size + aux_size, fc_layers, activation=fc_act)
        self.output_mu = nn.Linear(self.dense.output_size, latent_dim)
        self.output_logvar = nn.Linear(self.dense.output_size, latent_dim)
        self.aux_size = aux_size
        
    def forward(self, x, aux=None):
        if self.aux_size == 0:
            aux = None
        embedding = self.cnn(x)
        # Flatten only the image dimensions (last 3), preserving batch dimensions
        batch_shape = embedding.shape[:-3]  # All dimensions except the last 3
        image_shape = embedding.shape[-3:]  # The last 3 dimensions
        embedding = embedding.reshape(*batch_shape, -1)  # Flatten image dimensions
        if aux is not None:
            embedding = torch.cat([embedding, aux], dim=-1)
        embedding = self.dense(embedding)
        mu = self.output_mu(embedding)
        logvar = self.output_logvar(embedding)
        return (mu, logvar)
    
class ConvEncoderCategorical(nn.Module):
    def __init__(self, 
                 in_channels, 
                 img_size,
                 conv_layers,
                 fc_layers, 
                 latent_dim,
                 num_classes,
                 conv_act=F.relu,
                 fc_act=F.relu,
                 straight_through=False,
                 temperature=1.0,
                 aux_size=0):
        
        super(ConvEncoderCategorical, self).__init__()
        self.conv_act = conv_act
        self.fc_act = fc_act
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.straight_through = straight_through
        self.temperature = temperature

        self.cnn = CNN(in_channels, img_size, conv_layers, activation=conv_act)
        self.dense = Dense(self.cnn.output_size + aux_size, fc_layers, activation=fc_act)
        self.output = nn.Linear(self.dense.output_size, latent_dim*num_classes)
        self.aux_size = aux_size
        
    def forward(self, x, aux=None):
        if self.aux_size == 0:
            aux = None

        # Store the original batch shape
        original_shape = x.shape[:-3]  # All dimensions except the last 3 (image dimensions)

        # If there are multiple batch dimensions, flatten them
        if len(original_shape) > 1:
            x = x.view(-1, *x.shape[-3:])  # Flatten batch dims, preserve image dims

        embedding = self.cnn(x)
        # Flatten only the image dimensions (last 3), preserving batch dimensions
        batch_shape = embedding.shape[:-3]  # All dimensions except the last 3
        image_shape = embedding.shape[-3:]  # The last 3 dimensions
        embedding = embedding.reshape(*batch_shape, -1)  # Flatten image dimensions
        if aux is not None:
            aux = aux.view(-1, aux.shape[-1])  # Flatten batch dims, preserve aux dims
            embedding = torch.cat([embedding, aux], dim=-1)
        embedding = self.dense(embedding)
        logits = self.output(embedding)
        # Restore the original batch shape if multiple batch dims were flattened
        if len(original_shape) > 1:
            # First reshape to include all original batch dimensions and the latent dimensions
            logits = logits.view(*original_shape, self.latent_dim * self.num_classes)
        # Now reshape to have batch dimensions, latent dimension, and class dimension
        logits = logits.view(*logits.shape[:-1], self.latent_dim, self.num_classes)
        p_z = F.softmax(logits, dim=-1)
        return (logits, p_z)
    
class DenseEncoder(nn.Module):
    def __init__(self, 
                 input_size, 
                 layers, 
                 latent_dim, 
                 activation=F.relu,
                 aux_size=0):
        
        super(DenseEncoder, self).__init__()
        self.activation = activation
        
        self.dense = Dense(input_size + aux_size, layers, activation=activation)
        self.output_mu = nn.Linear(self.dense.output_size, latent_dim)
        self.output_logvar = nn.Linear(self.dense.output_size, latent_dim)
        self.aux_size = aux_size
        self.output_size = latent_dim
        
    def forward(self, x, aux=None):
        # Handle potential multiple batch dimensions by checking last dimension
        if x.size(-1) != self.dense.input_size - self.aux_size:
            # Preserve all batch dimensions and flatten only the feature dimensions
            batch_shape = x.shape[:-1]  # All dimensions except the last one
            x = x.reshape(*batch_shape, -1)  # Flatten to have correct feature size
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        mu = self.output_mu(x)
        logvar = self.output_logvar(x)
        return (mu, logvar)
    
class DenseEncoderCategorical(nn.Module):
    def __init__(self, 
                 input_size, 
                 layers, 
                 latent_dim,
                 num_classes,
                 activation=F.relu,
                 straight_through=False,
                 temperature=1.0,
                 aux_size=0):
        
        super(DenseEncoderCategorical, self).__init__()
        self.activation = activation
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.straight_through = straight_through
        self.temperature = temperature

        self.dense = Dense(input_size + aux_size, layers, activation=activation)
        self.output = nn.Linear(self.dense.output_size, latent_dim*num_classes)

        self.output_size = latent_dim*num_classes
        
    def forward(self, x, aux=None):
        # Handle potential multiple batch dimensions by checking last dimension
        if x.size(-1) != self.dense.input_size - self.aux_size:
            # Preserve all batch dimensions and flatten only the feature dimensions
            batch_shape = x.shape[:-1]  # All dimensions except the last one
            x = x.reshape(*batch_shape, -1)  # Flatten to have correct feature size
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        logits = self.output(x)
        logits = logits.view(-1, self.latent_dim, self.num_classes)
        p_z = F.softmax(logits, dim=-1)
        return (logits, p_z)