#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 11:13:34 2019

@author: ben
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AE
from ..inverse import inverse
from ...tools import torchutils as tu

class VAE(AE.AE):
    
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__(encoder, decoder)

    def reparam(self, mean, logvar):
        return torch.FloatTensor(mean.size()).normal_().to(self.device) * torch.exp(logvar / 2.) + mean

    def forward(self, *args):
        args = [x.to(self.device) for x in args]
        mu, logvar = self.encoder(*args)
        z = self.reparam(mu, logvar)
        return self.decoder(z), mu, logvar
    
    def encode(self, *args):
        args = [x.to(self.device) for x in args]
        mu, logvar = self.encoder(*args)
        return mu, logvar

def default2D(input_shape, latent_dim, share_weights=True):
    assert len(input_shape) == 2
    s1 = tu.conv_output_shape(input_shape, kernel_size=4, stride=2)
    s2 = tu.conv_output_shape(s1, kernel_size=4, stride=1)
    s3 = tu.conv_output_shape(s2, kernel_size=4, stride=1)
    
    layers = [nn.Conv2d(1, 64, kernel_size=4, stride=2),
              nn.Conv2d(64, 32, kernel_size=4, stride=1),
              nn.Conv2d(32, 16, kernel_size=4, stride=1),
              ]
    l_mu = nn.Linear(s3[0] * s3[1] * 16, latent_dim)
    l_logvar = nn.Linear(s3[0] * s3[1] * 16, latent_dim)
    l2 = nn.Linear(latent_dim, s3[0] * s3[1] * 16)

    inverse_layers = inverse(*layers, share_weights=share_weights)
    
    class Encoder(nn.Module):
        
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv1 = layers[0]
            self.conv2 = layers[1]
            self.conv3 = layers[2]
            self.l_mu = l_mu
            self.l_logvar = l_logvar
            self.latent_dim = latent_dim
   
        def forward(self, x):
            x_ = F.leaky_relu(self.conv1(x))
            x_ = F.leaky_relu(self.conv2(x_))
            x_ = F.leaky_relu(self.conv3(x_)).view(x.shape[0], -1)
            return self.l_mu(x_), self.l_logvar(x_)
        
    class Decoder(nn.Module):
        
        def __init__(self):
            super(Decoder, self).__init__()
            self.linear1 = l2
            self.conv1 = inverse_layers[0]
            self.conv2 = inverse_layers[1]
            self.conv3 = inverse_layers[2]
            self.latent_dim = latent_dim

        def forward(self, x):
            x_ = F.leaky_relu(self.linear1(x)).view(x.shape[0], 16, *s3)
            x_ = F.leaky_relu(self.conv1(x_))
            x_ = F.leaky_relu(self.conv2(x_))
            x_ = self.conv3(x_)
            return x_
        
    return Encoder(), Decoder()
