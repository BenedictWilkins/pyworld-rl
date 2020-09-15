#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:44:12 2019

@author: Benedict Wilkins

"""


import torch

import numpy as np
import math  

from . import datautils as du 



def unit_vector(x):
    """ 
        Convert a batch of vectors a batch of unit vectors
    """
    assert len(x.shape) == 2
    return x / torch.norm(x, dim=-1, keepdim=True) 




def as_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, list):
        return tuple(shape)
    elif isinstance(shape, int):
        return (shape,)
    else:
        raise ValueError("Invalid shape argument: {0}".format(str(shape)))

def identity(*args): # there is no identity in torch.nn.F ? use this as a placeholder
    return args

def collect(model, *data, batch_size=128):
    with torch.no_grad():
        iterator = du.batch_iterator(*data, batch_size=batch_size)
        result = torch.cat([model(x).cpu() for x in iterator], axis=0)
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

def from_numpy(x, *y, device='cpu'):
    """ Converts x to a torch tensor.

        Args:
            x (np.ndarray, tuple)
            y ([np.ndarray, tuple])
    """
    def _from_numpy(x):
        if isinstance(x, tuple):
            return tuple([_from_numpy(y) for y in x])
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        else:
            raise TypeError(type(x))
        
    if isinstance(x, tuple):
        return _from_numpy(tuple([*x, *y]))
    elif len(y) > 0:
        return _from_numpy(tuple([x, *y]))
    else:
        return _from_numpy(x)

def to_numpy(x, *y):
    '''
        Converts x to a numpy array, detaching gradient information and moving to cpu if neccessary.
        
        Examples:
            a,b = to_numpy(Tensor(), Tensor())
            a,b = to_numpy((Tensor(), Tensor()))
            (a,b), c = to_numpy((Tensor(), Tensor()), Tensor())
    '''
    def _to_numpy(x):
        if isinstance(x, tuple):
            return tuple([_to_numpy(y) for y in x])

        if isinstance(x, torch.Tensor):
            if x.requires_grad:
                x = x.detach()
            x = x.cpu()
            return x.numpy()
        elif isinstance(x, np.ndarray):
            return x
        else:
            return np.array(x)

    if isinstance(x, tuple):
        return _to_numpy(tuple([*x, *y]))
    elif len(y) > 0:
        return _to_numpy(tuple([x, *y]))
    else:
        return _to_numpy(x)
    

def to_numpyf(model):
    '''
        Wraps the output of a function (model) to a numpy array using to_numpy(model(x)).
    '''
    return lambda *x, **k: to_numpy(model(*x, **k))

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
