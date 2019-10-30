#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:21:53 2019

@author: Benedict Wilkins
"""

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    class __Parallel(nn.Module):
        
        def __init__(self, *layers):
            self.layers = layers
                
        def __call__(self, *args):
            return tuple([layer(*args) for layer in self.layers])
    
    def __init__(self, *layers):
        super(MLP, self).__init__()
        self.layers = layers
        for i, layer in enumerate(layers):
            if isinstance(layer, nn.Module):
                super(MLP, self).add_module('layer:' + str(i), layer)
                    
    def to(self, device):
        self.device = device
        super(MLP, self).to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        for a in self.layers:
            x = a(x)
        return x

def default(*shapes, output_activation=None, device='cpu'):
    layers = []
    for i in range(len(shapes)-1):
        layers.append(nn.Linear(shapes[i], shapes[i+1]))
        layers.append(F.leaky_relu)
    if output_activation:
        layers.append(output_activation)
    return MLP(layers).to(device)

if __name__ == "__main__":
    default(10,20,2)
    
    