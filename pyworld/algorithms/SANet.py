#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 11:36:03 2019

@author: ben
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyworld.toolkit.tools.torchutils as tu


class SANet(nn.Module):
    
    '''
        A convolutional network that takes as input a state of dimension state_shape = (C,H,W) 
        and a 1-hot representation of a discrete action of dimension action_shape = (N,). 
    '''
    
    def __init__(self, state_shape, action_shape):
        super(SANet, self).__init__()
        self.state_shape = state_shape
        self.action_shape = action_shape

        
        self.conv1 = nn.Conv2d(state_shape[0], 64, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.conv3 =  nn.Conv2d(32, 16, kernel_size=4, stride=1)
        
        self.action_layer1 = nn.Linear(action_shape, 256)
        self.action_layer2 = nn.Linear(256, 128)
                
        s1 = tu.conv_output_shape(state_shape[1:], kernel_size=4, stride=2)
        s2 = tu.conv_output_shape(s1, kernel_size=4, stride=1)
        s3 = tu.conv_output_shape(s2, kernel_size=4, stride=1)
        
        self.output_shape = np.prod(s3) * 16 + 128
        
    def inverse(self, share_weights=True):
        return tu.construct_inverse(self.conv1, self.conv2, self.conv3, share_weights=share_weights)
    
    def to(self, device):
        self.device = device
        return super(SANet, self).to(device)
    
    def forward(self, s, a):
            a_ = F.leaky_relu(self.action_layer1(a))
            a_ = F.leaky_relu(self.action_layer2(a_))
            s_ = F.leaky_relu(self.conv1(s))
            s_ = F.leaky_relu(self.conv2(s_))
            s_ = F.leaky_relu(self.conv3(s_)).view(s.shape[0], -1)
            x_ = torch.cat((s_, a_), 1)
            return x_
        
class DSANet(SANet):
    
    def __init__(self, state_shape, action_shape, output_shape, output_activation=lambda x: x):
        super(DSANet, self).__init__(state_shape, action_shape)
        self.output_layer = nn.Linear(self.output_shape, output_shape)
        self.output_activation = output_activation
        self.output_shape = output_shape
        
    def forward(self, s, a):
        x_ = F.leaky_relu(super(DSANet,self).forward(s, a))
        return self.output_activation(self.output_layer(x_))
