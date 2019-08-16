#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:44:12 2019

@author: ben
"""

import torch
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
    
def device(display=False):
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