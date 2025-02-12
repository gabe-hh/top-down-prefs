import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F

ACTIVATION_FN_MAP = {
    "elu": F.elu,
    "relu": F.relu,
    "sigmoid": F.sigmoid,
    "tanh": F.tanh,
    "leaky_relu": F.leaky_relu,
}

class Dense(nn.Module):
    def __init__(self, input_size, layers, activation=F.relu):
        super(Dense, self).__init__()
        self.activation = ACTIVATION_FN_MAP[activation.lower()] if isinstance(activation, str) else activation

        self.layers = nn.ModuleList()
        current_size = input_size

        for size in layers:
            self.layers.append(nn.Linear(current_size, size))
            print(f"layer: {current_size} -> {size}")
            current_size = size
        
        self.output_size = current_size

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))

        return x

class CNN(nn.Module):
    def __init__(self, in_channels, img_size, conv_layers, activation=F.relu):
        super(CNN, self).__init__()
        self.activation = ACTIVATION_FN_MAP[activation.lower()] if isinstance(activation, str) else activation
        
        # Layers: [(channels, kernel_size, stride, padding), ...]
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        
        for channels, kernel_size, stride, padding in conv_layers:
            self.conv_layers.append(nn.Conv2d(current_channels, channels, kernel_size, stride, padding))
            current_channels = channels
            img_size = (img_size - kernel_size + 2*padding) // stride + 1
        
        current_size = current_channels * img_size * img_size
        print(f"CNN output: {channels}x{img_size}x{img_size} = {current_size}")

        self.output_size = current_size

    def forward(self, x):
        # Convolutional forward pass
        for conv in self.conv_layers:
            x = self.activation(conv(x))
        
        return x
    
class DCNN(nn.Module):
    def __init__(self, in_channels, img_size, deconv_layers,
                 activation=F.relu, final_activation=None, last_layer_conv=False):
        """
        Parameters:
            in_channels: Number of input channels.
            img_size: Spatial size of the input (assumed square).
            deconv_layers: List of tuples (channels, kernel_size, stride, padding, output_padding)
            activation: Activation for intermediate layers.
            final_activation: Activation for the final layer (e.g., torch.sigmoid). If None, no activation is applied.
            final_layer_type: 'deconv' to use ConvTranspose2d or 'conv' to use Conv2d for the final layer.
        """
        super(DCNN, self).__init__()
        self.activation = ACTIVATION_FN_MAP[activation.lower()] if isinstance(activation, str) else activation
        self.final_activation = ACTIVATION_FN_MAP[final_activation.lower()] if isinstance(final_activation, str) else final_activation
        self.deconv_layers = nn.ModuleList()
        current_channels = in_channels

        for i, (channels, kernel_size, stride, padding, output_padding) in enumerate(deconv_layers):
            is_last = (i == len(deconv_layers) - 1)
            if is_last and last_layer_conv:
                layer = nn.Conv2d(current_channels, channels, kernel_size, stride, padding)
            else:
                layer = nn.ConvTranspose2d(current_channels, channels, kernel_size, stride, padding, output_padding)
            self.deconv_layers.append(layer)
            current_channels = channels
            img_size = (img_size - 1) * stride - 2*padding + kernel_size + output_padding

        self.output_size = (current_channels, img_size, img_size)
        print(f"DCNN output: {current_channels}x{img_size}x{img_size}")

    def forward(self, x):
        num_layers = len(self.deconv_layers)
        for i, layer in enumerate(self.deconv_layers):
            x = layer(x)

            if i == num_layers - 1:
                if self.final_activation is not None:
                    x = self.final_activation(x)
            else:
                x = self.activation(x)
        return x