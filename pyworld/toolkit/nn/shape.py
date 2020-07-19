#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 15:16:33

    Shape utilities.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import math
import torch.nn as nn

def as_shape(shape):
    if isinstance(shape, int):
        return (shape,)
    else:
        return tuple(shape)
       
def conv2d_shape(layer, input_shape, *args, **kwargs):
    """ 
        Get the output shape of a 2D convolution given the input_shape.
        
    Args:
        layer (nn.Conv2d): 2D convolutional layer.
        input_shape (tuple): expected input shape (CHW format)

    Returns:
        tuple: output shape (CHW)
    """
    input_shape = as_shape(input_shape)
    h,w = input_shape[-2:]
    pad, dilation, kernel_size, stride = layer.padding, layer.dilation, layer.kernel_size, layer.stride

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h + (2 * pad[0]) - (dilation[0] * (kernel_size[0] - 1) ) - 1 )/ stride[0]) + 1)
    w = math.floor(((w + (2 * pad[1]) - (dilation[1] * (kernel_size[1] - 1) ) - 1 )/ stride[1]) + 1)
    return layer.out_channels, h, w

def linear_shape(layer, *args, **kwargs):
    """ Get the output shape of a linear layer.

    Args:
        layer (nn.Linear): linear layer.

    Returns:
        tuple: output shape (D,)
    """
    return (layer.weight.shape[0],)


shape_map = {nn.Linear:linear_shape,
             nn.Conv2d:conv2d_shape,
             nn.Identity:lambda layer, input_shape, *args, **kwargs: input_shape}

def shape(layer, *args, **kwargs):
    return shape_map[type(layer)](layer, *args, **kwargs)