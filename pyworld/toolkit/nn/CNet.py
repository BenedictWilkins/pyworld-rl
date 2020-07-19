#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 14:56:00 2019

@author: ben
"""

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..tools import torchutils as tu
from .inverse import inverse

from . import Sequential


class CNet(Sequential.Sequential):
    
    '''
        A convolutional network that takes as input an image of dimension input_shape = (C,H,W).
    '''
    def __init__(self, input_shape):

        self.input_shape = tu.as_shape(input_shape)
  
        s1 = tu.conv_output_shape(input_shape, 16, kernel_size=4, stride=2)
        s2 = tu.conv_output_shape(s1, 32, kernel_size=4, stride=1)
        s3 = tu.conv_output_shape(s2, 64, kernel_size=4, stride=1)
    
        layers = dict(conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2), r1 = F.leaky_relu,
                      conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1),             r2 = F.leaky_relu,
                      conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1),             r3 = F.leaky_relu)

        super(CNet, self).__init__(**layers) 
        
        self.output_shape = self.shape(input_shape)['conv3']

class CNet2(CNet):
    
    '''
        A convolutional network based on CNet with a fully connected output layer of given dimension.
    '''
    
    def __init__(self, input_shape, output_shape, output_activation=nn.Identity()):
        super(CNet2, self).__init__(input_shape)
        output_shape = tu.as_shape(output_shape)
        self.layers['view'] = Sequential.view(-1)
        self.layers['out_layer'] = nn.Linear(np.prod(self.output_shape), output_shape[0])
        self.layers['out_activation'] = output_activation
        self.output_shape = output_shape

    def inverse(self, **kwargs):
        inet = super(CNet2, self).inverse(**kwargs)
        inet.layers['view'] = Sequential.view(*self.shape(self.input_shape)['conv3'])
        return inet

"""
# old CNet implementations, TODO remove

class CNet(nn.Module):
    '''
        A convolutional network that takes as input an image of dimension input_shape = (C,H,W).
    '''
    def __init__(self, input_shape):
        super(CNet, self).__init__() 
        self.input_shape = tu.as_shape(input_shape)
        self.device = "cpu"
        
        s1 = tu.conv_output_shape(input_shape, 16, kernel_size=4, stride=2)
        s2 = tu.conv_output_shape(s1, 32, kernel_size=4, stride=1)
        s3 = tu.conv_output_shape(s2, 64, kernel_size=4, stride=1)
    
        self.conv1 = nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1)

        self.output_shape = tu.as_shape(int(np.prod(s3)))

    def to(self, device):
        self.device = device
        return super(CNet, self).to(device)

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

"""