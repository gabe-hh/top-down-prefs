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
        embedding = torch.flatten(embedding, start_dim=1)
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
        embedding = self.cnn(x)
        embedding = torch.flatten(embedding, start_dim=1)
        if aux is not None:
            embedding = torch.cat([embedding, aux], dim=-1)
        embedding = self.dense(embedding)
        logits = self.output(embedding)
        logits = logits.view(-1, self.latent_dim, self.num_classes)
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
        x = x.view(x.size(0), -1)
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
        x = x.view(x.size(0), -1)
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        logits = self.output(x)
        logits = logits.view(-1, self.latent_dim, self.num_classes)
        p_z = F.softmax(logits, dim=-1)
        return (logits, p_z)