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
        return (x, None)
    
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
        x = x.view(x.size(0), -1)
        if self.aux_size == 0:
            aux = None
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        x = self.dense(x)
        x = self.output(x)
        return (x, None)
    
class MultiHeadDecoder(nn.Module):
    def __init__(self, 
                 input_dim,
                 in_shape,
                 deconv_layers_img,
                 deconv_layers_mask,
                 fc_layers, 
                 conv_act=F.relu,
                 fc_act=F.relu,
                 final_act_img=torch.sigmoid,
                 final_act_mask=None,  # We'll output logits for mask (use BCEWithLogitsLoss)
                 aux_size=0,
                 last_layer_conv=False):
        super(MultiHeadDecoder, self).__init__()
        self.in_shape = in_shape  # e.g., [channels, height, width] for the image branch
        self.aux_size = aux_size
        img_size = in_shape[-1]
        in_channels = in_shape[0]
        fc_layers = fc_layers.copy()
        fc_layers.append(in_channels*img_size*img_size)

        # Shared dense block that processes the latent vector (+aux input)
        self.shared_dense = Dense(input_dim + aux_size, fc_layers, activation=fc_act)
        
        # After the dense block, we assume the output is reshaped to a feature map
        # matching in_shape. Both branches will share these features.
        
        # Image branch: deconvolution layers for full image reconstruction
        self.image_deconv = DCNN(
            in_channels=in_shape[0], 
            img_size=in_shape[1], 
            deconv_layers=deconv_layers_img, 
            activation=conv_act, 
            final_activation=final_act_img, 
            last_layer_conv=last_layer_conv
        )
        
        # Goal mask branch: similar structure but designed for 1-channel output.
        self.mask_deconv = DCNN(
            in_channels=1,            # output a single-channel mask
            img_size=in_shape[1], 
            deconv_layers=deconv_layers_mask, 
            activation=conv_act, 
            final_activation=final_act_mask,  # No final activation if using BCEWithLogitsLoss
            last_layer_conv=last_layer_conv
        )

    def forward(self, x, aux=None):
        # x is the latent vector (or feature from latent handler)
        # Pass through dense (shared) layers
        if self.aux_size == 0:
            aux = None
        x = x.view(x.size(0), -1)
        if aux is not None:
            x = torch.cat([x, aux], dim=-1)
        shared_features = self.shared_dense(x)
        
        # Reshape shared features into the shape expected by deconv layers.
        # This assumes self.in_shape is compatible with the dense output.
        shared_features = shared_features.view(shared_features.size(0), *self.in_shape)
        
        # Pass shared features to each branch
        img_out = self.image_deconv(shared_features)
        
        # For the mask branch, you might either:
        # Option A: Pass the shared features directly (if the spatial resolution is compatible)
        # Option B: Process shared features with a small convolutional head.
        # Here, we assume Option A for simplicity.
        # First, we squeeze/adjust channels if needed:
        mask_input = shared_features.mean(dim=1, keepdim=True)  # For example, average over channels
        mask_out = self.mask_deconv(mask_input)
        
        return (img_out, mask_out)
