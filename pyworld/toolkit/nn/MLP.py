#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:21:53 2019

@author: Benedict Wilkins
"""

import torch.nn as nn
import torch.nn.functional as F

from ..tools import torchutils as tu

class MLP(nn.Module):
    
    def __init__(self, *shapes, output_activation=None):
        super(MLP, self).__init__()
        if len(shapes) < 2:
            raise ValueError("shapes must contain at least 2 elements, the input and output dimension of the network")
        self.device = 'cpu'
        layers = []
        
        for i in range(len(shapes)-2):
            #print("layer", shapes[i], shapes[i+1])
            layers.append(nn.Linear(shapes[i], shapes[i+1]))
            layers.append(nn.LeakyReLU())
        #print("layer", shapes[-2], shapes[-1])
        layers.append(nn.Linear(shapes[-2], shapes[-1]))
        if output_activation is not None:
            self.output_activation = output_activation
        else:
            self.output_activation = nn.Identity()

        '''
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                super(MLP, self).add_module('layer:' + str(i), layer)   
        '''       
        self.layers = nn.Sequential(*layers)        
        
        self.input_shape = tu.as_shape(shapes[0])
        self.output_shape = tu.as_shape(shapes[-1])
        
    def to(self, device):
        self.device = device
        return super(MLP, self).to(self.device)
    
    def forward(self, x):
        x = x.to(self.device)
        return self.output_activation(self.layers(x))

if __name__ == "__main__":
    import torch
    mlp = MLP(1,2,3,4)
    print(mlp(torch.FloatTensor([[1.]])))