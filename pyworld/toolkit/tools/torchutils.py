#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:44:12 2019

@author: ben
"""

try:
    import torch
except:
    print("WARNING: MODULE NOT FOUND: torch")

import numpy as np
import math  

from . import datautils as du 

def as_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise ValueError("Invalid shape argument: {0}".format(str(shape)))

def collect(model, *data, batch_size=128):
    with torch.no_grad():
        iterator = du.batch_iterator(*data, batch_size=batch_size, force_nonsingular=True)
        result = torch.cat([model(*x).cpu() for x in iterator], axis=0)
    return result

def load(model, *args, device = 'cpu', path = None, **kwargs): #see fileutils now...
    model_ = model(*args, **kwargs).to(device)
    if path is not None:
        model_.load_state_dict(torch.load(path))    
    return model_
        
def batch_to_numpy(batch, types, copy=False):
    return [np.array(batch[i], copy=copy, dtype=types[i]) for i in range(len(batch))]

def batch_to_tensor(batch, types, device='cpu'):
        return [types[i](batch[i]).to(device) for i in range(len(batch))]
    
def distance_matrix(x, y):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    dist_mat = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat    

def to_numpy(x):
    '''
        Converts x to a numpy array, detaching gradient information and moving to cpu if neccessary.
    '''
    if isinstance(x, torch.Tensor):
        x = x.cpu()
        if x.requires_grad:
            x = x.detach()
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise TypeError("x must be torch Tensor or numpy array")

def to_numpyf(model):
    '''
        Use: Wraps the output of a function (model) to a numpy array using numpy(x).
    '''
    return lambda *x: to_numpy(model(*x))

def to_torch(x):
    if torch.is_tensor(x):
        return x
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    else:
        return torch.Tensor(x) #???

    
def device(display=True):
    device = None
    if(torch.cuda.is_available()): 
        device = 'cuda'
    else:
        device = 'cpu'
    if display:
        print("USING DEVICE:", device)
    return device

def conv_output_shape(input_shape, out_channels, kernel_size=1, stride=1, pad=0, dilation=1):
    '''
        Get the output shape of a convolution given the input_shape.
        Arguments:
            input_shape: in CHW (or HW) format
            TODO
    '''
    input_shape = as_shape(input_shape)
    h,w = input_shape[-2:]

    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = math.floor(((w + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return out_channels, h, w
