#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:44:12 2019

@author: ben
"""

import torch
import torch.nn as nn
import numpy as np    

def load(model, *args, device = 'cpu', path = None, **kwargs):
    model_ = model(*args, **kwargs).to(device)
    if path is not None:
        model_.load_state_dict(torch.load(path))    
    return model_
        
    
def batch_to_numpy(batch, types, copy=False):
    return [np.array(batch[i], copy=copy, dtype=types[i]) for i in range(len(batch))]

def batch_to_tensor(batch, types, device='cpu'):
        return [types[i](batch[i]).to(device) for i in range(len(batch))]
    
def distance_matrix(x,y):
    x = x.view(x.shape[0], -1)
    y = y.view(y.shape[0], -1)
    dist_mat = torch.empty((x.shape[0], y.shape[0]), dtype=x.dtype)
    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        sq_dist = torch.sum((r_v - y) ** 2, 1)
        dist_mat[i] = sq_dist.view(1, -1)
    return dist_mat    

def numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.cpu()
        if x.requires_grad:
            x = x.detach()
        return x.numpy()
    elif isinstance(x, np.array):
        return x
    else:
        raise TypeError

def tonumpy(model):
    return lambda x: numpy(model(x))
    
def device(display=True):
    device = None
    if(torch.cuda.is_available()): 
        device = 'cuda'
    else:
        device = 'cpu'
    if display:
        print("USING DEVICE:", device)
    return device

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w



class __inverse:
    
    def not_implemented(*args):
        raise NotImplementedError
        
    def convtranspose2d(layer, share_weights):
        convt2d = nn.ConvTranspose2d(layer.out_channels, layer.in_channels, 
                           kernel_size=layer.kernel_size, 
                           stride=layer.stride, 
                           padding=layer.padding)
        if share_weights:
            convt2d.weight = layer.weight
        return convt2d
    
    def lineartranspose(layer, share_weights):
        lt = nn.Linear(layer.out_features, layer.in_features, layer.bias is not None)
        if share_weights:
            lt.weight = nn.Parameter(layer.weight.t())
        return lt
    
    il = {
          nn.Conv1d: not_implemented,
          nn.Conv2d: convtranspose2d,
          nn.Conv3d: not_implemented,
          nn.Linear: lineartranspose
         }

def construct_inverse(*layers, share_weights = True):
    inverse_layers = []
    for layer in reversed(layers):
        inverse_layers.append(__inverse.il[type(layer)](layer, share_weights))
    return inverse_layers

def default_conv2d(input_shape):    
    assert len(input_shape) == 2
    s1 = conv_output_shape(input_shape, kernel_size=4, stride=2)
    s2 = conv_output_shape(s1, kernel_size=4, stride=1)
    s3 = conv_output_shape(s2, kernel_size=4, stride=1)
    
    layers = [nn.Conv2d(1, 64, kernel_size=4, stride=2),
              nn.Conv2d(64, 32, kernel_size=4, stride=1),
              nn.Conv2d(32, 16, kernel_size=4, stride=1)]
    
    return layers, [s1, s2, s3]


if __name__ == "__main__":
    x = torch.FloatTensor([[1],[2],[3],[4]])
    z = distance_matrix(x,x)
    print(z)