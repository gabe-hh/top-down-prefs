import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

from src.model.base import Dense

class TransitionBase(nn.Module):
    def __init__(self):
        super(TransitionBase, self).__init__()
        self.is_recurrent = False

    def forward(self, x, a):
        raise NotImplementedError
    
    def zero_hidden(self, batch_size):
        if self.is_recurrent:
            return torch.zeros(batch_size, self.hidden_dim)
        return None

class TransitionNormal(TransitionBase):
    def __init__(self, 
                 latent_dim, 
                 action_dim, 
                 layers,
                 activation=F.relu):
        
        super(TransitionNormal, self).__init__()
        
        self.action_dim = action_dim
        self.latent_dim = latent_dim

        self.dense = Dense(latent_dim+action_dim, layers, activation=activation)
        self.output_mu = nn.Linear(self.dense.output_size, latent_dim)
        self.output_logvar = nn.Linear(self.dense.output_size, latent_dim)

    def forward(self, x, a):
        x = self.dense(torch.cat([x, a], dim=-1))
        mu = self.output_mu(x)
        logvar = self.output_logvar(x)
        return (mu, logvar)
    
class TransitionCategorical(TransitionBase):
    def __init__(self, 
                 latent_dim,
                 num_classes, 
                 action_dim, 
                 layers,
                 activation=F.relu,
                 final_activation=F.linear):
        
        super(TransitionCategorical, self).__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.input_size = latent_dim*num_classes + action_dim
        self.dense = Dense(self.input_size, layers, activation=activation)
        self.output = nn.Linear(self.dense.output_size, latent_dim*num_classes)
        self.final_activation = final_activation

    def forward(self, x, a):
        x = x.view(-1, self.latent_dim*self.num_classes)
        a = a.view(-1, self.action_dim)
        x = self.dense(torch.cat([x, a], dim=-1))
        logits = self.final_activation(self.output(x))
        logits = logits.view(-1, self.latent_dim, self.num_classes)
        p_x = F.softmax(logits, dim=-1)
        return (logits, p_x)
    
class RSSMTransitionNormal(TransitionBase):
    def __init__(self, 
                 latent_dim, 
                 action_dim, 
                 hidden_dim, 
                 layers,
                 cell_type='gru',
                 activation=F.relu):
        super(RSSMTransitionNormal, self).__init__()
        
        self.is_recurrent = True
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.input_size = latent_dim + action_dim

        if cell_type == 'gru':
            self.rnn = nn.GRUCell(latent_dim+action_dim, hidden_dim)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTMCell(latent_dim+action_dim, hidden_dim)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        
        self.dense = Dense(hidden_dim, layers, activation=activation)
        self.output_mu = nn.Linear(self.dense.output_size, latent_dim)
        self.output_logvar = nn.Linear(self.dense.output_size, latent_dim)
    
    def forward(self, x, a, h):
        h = self.rnn(torch.cat([x, a], dim=-1), h)
        x = self.dense(h)
        mu = self.output_mu(x)
        logvar = self.output_logvar(x)
        return (mu, logvar), h
    
class RSSMTransitionCategorical(TransitionBase):
    def __init__(self, 
                 latent_dim, 
                 num_classes, 
                 action_dim, 
                 hidden_dim, 
                 layers,
                 cell_type='gru',
                 activation=F.relu,
                 final_activation=F.linear):
        super(RSSMTransitionCategorical, self).__init__()

        self.is_recurrent = True
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.input_size = latent_dim*num_classes + action_dim

        if cell_type == 'gru':
            self.rnn = nn.GRUCell(self.input_size, hidden_dim)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTMCell(self.input_size, hidden_dim)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
        self.dense = Dense(hidden_dim, layers, activation=activation)
        self.output = nn.Linear(self.dense.output_size, latent_dim*num_classes)
        #self.final_activation = final_activation

    def forward(self, x, a, h):
        x = x.view(-1, self.latent_dim*self.num_classes)
        a = a.view(-1, self.action_dim)
        h = self.rnn(torch.cat([x, a], dim=-1), h)
        x = self.dense(h)
        logits = self.output(x)
        logits = logits.view(-1, self.latent_dim, self.num_classes)
        p_x = F.softmax(logits, dim=-1)
        return (logits, p_x), h
    
class RNNTransitionNormal(TransitionBase):
    def __init__(self, 
                 latent_dim, 
                 action_dim,
                 layers,
                 cell_type='gru',
                 activation=F.relu):
        super(RNNTransitionNormal, self).__init__()

        self.is_recurrent = True
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = latent_dim*2

        self.dense = Dense(latent_dim, layers, activation=activation)

        if cell_type == 'gru':
            self.rnn = nn.GRUCell(self.dense.output_size, self.hidden_dim)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTMCell(self.dense.output_size, self.hidden_dim)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
    
    def forward(self, x, a, h):
        x = self.dense(torch.cat([x, a], dim=-1))
        mu, logvar = torch.split(h, self.latent_dim, dim=-1)
        return (mu, logvar), h
    
class RNNTransitionCategorical(TransitionBase):
    def __init__(self, 
                 latent_dim, 
                 num_classes, 
                 action_dim,
                 layers,
                 cell_type='gru',
                 activation=F.relu):
        super(RNNTransitionCategorical, self).__init__()

        self.is_recurrent = True
        self.latent_dim = latent_dim
        self.input_size = latent_dim + action_dim
        self.num_classes = num_classes
        self.action_dim = action_dim
        self.hidden_dim = latent_dim * num_classes
        self.input_size = latent_dim * num_classes + action_dim

        self.dense = Dense(self.input_size, layers, activation=activation)

        if cell_type == 'gru':
            self.rnn = nn.GRUCell(self.dense.output_size, self.hidden_dim)
        elif cell_type == 'lstm':
            self.rnn = nn.LSTMCell(self.dense.output_size, self.hidden_dim)
        else:
            raise ValueError(f"Unknown cell type: {cell_type}")
    
    def forward(self, x, a, logits):
        x = x.view(-1, self.latent_dim*self.num_classes)
        a = a.view(-1, self.action_dim)
        logits = logits.view(-1, self.num_classes*self.latent_dim)
        x = self.dense(torch.cat([x, a], dim=-1))
        h = self.rnn(x, logits)

        logits = h.view(-1, self.latent_dim, self.num_classes)
        p_x = F.softmax(logits, dim=-1)
        return (logits, p_x), h