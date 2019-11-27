#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:21:53 2019

@author: Benedict Wilkins
"""

import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, *shapes, output_activation=None):
        super(MLP, self).__init__()
        self.layers = []
        for i in range(len(shapes)-1):
            self.layers.append(nn.Linear(shapes[i], shapes[i+1]))
            self.layers.append(F.leaky_relu)
        if output_activation:
            self.layers.append(output_activation)
            
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                super(MLP, self).add_module('layer:' + str(i), layer)
        print()
        
        
    def to(self, device):
        self.device = device
        return super(MLP, self).to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        for a in self.layers:
            x = a(x)
        return x

def default():
    layers = []
    
    return MLP(layers)

if __name__ == "__main__":
    default(10,20,2)
    
    