import torch
from torch import nn
import torch.nn.functional as F

from src.model.base import Dense, DCNN

class ConvDecoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 in_shape,
                 deconv_layers,
                 fc_layers, 
                 conv_act=F.relu,
                 fc_act=F.relu,
                 final_act=F.sigmoid,
                 last_layer_conv=False,
                 aux_size=0):
        
        super(ConvDecoder, self).__init__()
        self.input_dim = input_dim
        self.conv_act = conv_act
        self.fc_act = fc_act
        self.in_shape = in_shape
        img_size = in_shape[-1]
        in_channels = in_shape[0]
        fc_layers = fc_layers.copy()
        fc_layers.append(in_channels*img_size*img_size)
        self.dense = Dense(input_dim + aux_size, fc_layers, activation=fc_act)
        self.deconv = DCNN(in_channels, img_size, deconv_layers, activation=conv_act, final_activation=final_act, last_layer_conv=last_layer_conv)
        self.output_size = self.deconv.output_size
        self.aux_size = aux_size

        print(f"ConvDecoder output: {self.output_size[0]}x{self.output_size[1]}x{self.output_size[2]}")

    def forward(self, x, aux=None):
        if self.aux_size == 0:
            aux = None
        x = x.view(x.size(0), -1)
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        x = x.view(x.size(0), *self.in_shape)
        x = self.deconv(x)
        return x
    
class DenseDecoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 fc_layers, 
                 output_size,
                 fc_act=F.relu,
                 aux_size=0):
        
        super(DenseDecoder, self).__init__()
        self.input_dim = input_dim
        self.fc_act = fc_act
        self.dense = Dense(input_dim + aux_size, fc_layers, activation=fc_act)
        self.output = nn.Linear(self.dense.output_size, output_size)
        self.aux_size = aux_size
        self.output_size = output_size
        
    def forward(self, x, aux=None):
        if self.aux_size == 0:
            aux = None
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        x = self.output(x)
        return x