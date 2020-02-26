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
        self.device = 'cpu'
        self.state_shape = tu.as_shape(state_shape)
        self.action_shape = tu.as_shape(action_shape)

        s1 = tu.conv_output_shape(state_shape, 16, kernel_size=4, stride=2)
        s2 = tu.conv_output_shape(s1, 32, kernel_size=4, stride=1)
        s3 = tu.conv_output_shape(s2, 64, kernel_size=4, stride=1)
    
        self.conv1 = nn.Conv2d(state_shape[0], 16, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=1)

        self.action_layer1 = nn.Linear(self.action_shape[0], 256)
        self.action_layer2 = nn.Linear(256, 256)   
        
        self.output_shape = tu.as_shape(int(np.prod(s3)) + 256)
        
    def inverse(self, share_weights=True):
        return tu.construct_inverse(self.conv1, self.conv2, self.conv3, share_weights=share_weights)
    
    def to(self, device):
        self.device = device
        return super(SANet, self).to(device)
    
    def forward(self, sa):
        s, a = sa # this is easier with the use of an optimiser (otherwise we gotta mess around with *(x,) everywhere!)
        s, a = s.to(self.device), a.to(self.device)
        a_ = F.leaky_relu(self.action_layer1(a))
        a_ = F.leaky_relu(self.action_layer2(a_))
        s_ = F.leaky_relu(self.conv1(s))
        s_ = F.leaky_relu(self.conv2(s_))
        s_ = F.leaky_relu(self.conv3(s_)).view(s.shape[0], -1)
        x_ = torch.cat((s_, a_), 1)
        return x_
        
class SANet2(SANet):
    
    '''
        An extension to SANet that includes an output layer of a given size.
    '''
    
    def __init__(self, state_shape, action_shape, output_shape, output_activation=nn.Identity()):
        super(SANet2, self).__init__(state_shape, action_shape)
        output_shape = tu.as_shape(output_shape)
        self.output_layer = nn.Linear(self.output_shape[0], output_shape[0])
        self.output_shape = output_shape
        self.output_activation = output_activation

    def forward(self, sa):
        x_ = F.leaky_relu(super(SANet2, self).forward(sa))
        return self.output_activation(self.output_layer(x_))
    
class SANet3(nn.Module): #TODO
    
    '''
        Similar to SANet2, but an MLP network that takes in a vectorised state of dimension state_shape = (M,),
        and a 1-hot representation of a discrete action of dimension action_shape = (N,). 
    '''
    
    def __init__(self, state_shape, action_shape, output_shape):
        self.action_shape = tu.as_shape(action_shape)
        self.state_shape = tu.as_shape(state_shape)
        self.output_shape = tu.as_shape(output_shape)
        
        _iaction_shape = self.action_shape[0] * 64
        _istate_shape = self.state_shape[0] * 64
        self.action_layer1 = nn.Linear(self.action_shape[0], _iaction_shape)

        self.state_layer1 = nn.Linear(self.state_shape[0], _istate_shape)

        _icom_shape = (_istate_shape + _iaction_shape) // 2
        self.com_layer1 = nn.Linear(_istate_shape + _iaction_shape, _icom_shape)
        self.com_layer2 = nn.Linear(_icom_shape, self.output_shape[0])
        
    def forward(self, sa):
        s, a = sa
        raise NotImplementedError()
    
