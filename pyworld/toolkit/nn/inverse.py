#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 11:39:40

    Build inverse PyTorch layers.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d_transpose(layer, share_weights=True):
    """ Transpose 2D convolutional layer (see torch.nn.ConvTranspose2d).

    Args:
        layer (torch.nn.Conv2d): Convolutional layer.
        share_weights (bool, optional): should the inverse layer share weights?. Defaults to True.

    Returns:
        torch.nn.ConvTranspose2d: inverse layer.
    """
    convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                       kernel_size=layer.kernel_size, 
                       stride=layer.stride, 
                       padding=layer.padding)
    if share_weights:
        convt2d.weight = layer.weight
    return convt2d

def linear_transpose(layer, share_weights=True):
    """ Transpose linear layer.

    Args:
        layer (torch.nn.Linear): Linear layer.
        share_weights (bool, optional): should the inverse layer share weights? (bias will not be shared). Defaults to True.

    Returns:
        torch.nn.Linear: inverse layer.
    """
    lt = nn.Linear(layer.out_features, layer.in_features, layer.bias is not None)
    if share_weights:
        lt.weight = nn.Parameter(layer.weight.t())
    return lt

inverse_map = {
      #nn.Conv1d: not_implemented,
      nn.Conv2d: conv2d_transpose,
      #nn.Conv3d: not_implemented,
      nn.Linear: linear_transpose,
      nn.Identity: lambda *args, **kwargs: nn.Identity()} #TODO fill this out

def inverse(*layers, share_weights = True):
    """ Inverse a sequence of layers. The returned sequence will be in reverse order, i.e. if the input is l1,l2,l3, the output will be the li3,li2,li1.

    Args:
        share_weights (bool, optional): should the inverse layers share the weights of the original?. Defaults to True.

    Returns:
        list: inverse layers (in reverse order).
    """
    inverse_layers = []
    for layer in reversed(layers):
        inverse_layers.append(inverse_map[type(layer)](layer, share_weights))
    return inverse_layers