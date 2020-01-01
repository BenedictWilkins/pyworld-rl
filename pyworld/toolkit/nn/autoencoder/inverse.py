import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_transpose(layer, share_weights=True):
    convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                       kernel_size=layer.kernel_size, 
                       stride=layer.stride, 
                       padding=layer.padding)
    if share_weights:
        convt2d.weight = layer.weight
    return convt2d

def linear_transpose(layer, share_weights=True):
    lt = nn.Linear(layer.out_features, layer.in_features, layer.bias is not None)
    if share_weights:
        lt.weight = nn.Parameter(layer.weight.t())
    return lt

inverse_map = {
      #nn.Conv1d: not_implemented,
      nn.Conv2d: conv2d_transpose,
      #nn.Conv3d: not_implemented,
      nn.Linear: linear_transpose}

def inverse(*layers, share_weights = True):
    inverse_layers = []
    for layer in reversed(layers):
        inverse_layers.append(inverse_map[type(layer)](layer, share_weights))
    return inverse_layers