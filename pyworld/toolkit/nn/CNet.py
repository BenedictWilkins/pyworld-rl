#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:56:00 2019

@author: ben
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import pyworld.toolkit.tools.torchutils as tu

class CNet(nn.Module):
    
    '''
        A convolutional network that takes as input an image of dimension input_shape = (C,H,W).
    '''
    def __init__(self, input_shape, device='cpu'):
        super(CNet, self).__init__() 
        self.input_shape = tu.as_shape(input_shape)
        
        self.conv1 = nn.Conv2d(self.input_shape[0], 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.conv3 =  nn.Conv2d(32, 64, kernel_size=4, stride=1)
        if device != 'cpu':
            self.to(device)

        s1 = tu.conv_output_shape(self.input_shape[1:], kernel_size=4, stride=2)
        s2 = tu.conv_output_shape(s1, kernel_size=4, stride=1)
        s3 = tu.conv_output_shape(s2, kernel_size=4, stride=1)
        
        self.output_shape = tu.as_shape(int(np.prod(s3) * 64))
    
    def to(self, device):
        self.device = device
        return super(CNet, self).to(device)
        
    def inverse(self, share_weights=True):
        return tu.construct_inverse(self.conv1, self.conv2, self.conv3, share_weights=share_weights)
        
    def forward(self, x_):
        x_ = x_.to(self.device)
        y_ = F.leaky_relu(self.conv1(x_))
        y_ = F.leaky_relu(self.conv2(y_))
        y_ = F.leaky_relu(self.conv3(y_)).view(x_.shape[0], -1)
        return y_
    
    
class CNet2(CNet):
    
    '''
        A convolutional network based on CNet with a fully connected output layer of given dimension.
    '''
    
    def __init__(self, input_shape, output_shape, output_activation=nn.Identity()):
        super(CNet2, self).__init__(input_shape)
        output_shape = tu.as_shape(output_shape)
        self.out_layer = nn.Linear(self.output_shape[0], output_shape[0])
        self.output_shape = output_shape
        self.output_activation = output_activation
        
    def forward(self, x_):
        x_ = super(CNet2, self).forward(x_)
        y_ = self.output_activation(self.out_layer(x_))
        return y_
        
        
        
        
        
        
        