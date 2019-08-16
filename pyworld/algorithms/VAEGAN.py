#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:07:20 2019

@author: ben
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import VAE

class Discriminator(nn.Module):
    
    def __init__(self, x_dim, h_dim, z_dim, channels, keep_prob):
        super(Discriminator, self).__init__()
         #this could be shared with the encoder?
        self.cin_1_disc = nn.Conv2d(channels, 64, kernel_size=4, stride=2)
        self.cin_2_disc = nn.Conv2d(64, 32, kernel_size=4, stride=1)
        self.cin_3_disc = nn.Conv2d(32, 16, kernel_size=4, stride=1)
        
        self.cout_shape = np.prod(self._conv_shapes(x_dim)[-1])
        
        #this is the layer we will use to compute loss
        self.intermediate = nn.Sequential(
            nn.Dropout(1-keep_prob),
            nn.Linear(self.cout_shape, h_dim),
            nn.ReLU())
        
        self.out = nn.Sequential(
            nn.Dropout(1-keep_prob),
            nn.Linear(h_dim, 1),
            nn.Sigmoid())
        
    def forward(self, x):
        _in = F.relu(self.cin_1_disc(x))
        _in = F.relu(self.cin_2_disc(_in))
        _in = F.relu(self.cin_3_disc(_in)).view(-1, self.cout_shape)
        im = self.intermediate(_in)
        return self.out(im), _in.detach()
        
    def _conv_shapes(self, x_dim):
        shapes = []
        _in = self.cin_1_disc(torch.FloatTensor(np.zeros((1, *x_dim))))
        shapes.append(_in.shape)
        _in = self.cin_2_disc(_in)
        shapes.append(_in.shape)
        _in = self.cin_3_disc(_in)
        shapes.append(_in.shape)
        return shapes
    
    
class VAEGAN(VAE):

    def __init__(self, x_dim, h_dim, z_dim, channels=1, keep_prob = 0.8):
        super(VAEGAN, self).__init__(x_dim, h_dim, z_dim, channels, keep_prob)
        self.disc = Discriminator(x_dim, h_dim, z_dim, channels, keep_prob)
    
    def forward(self, x):
        x = x.to(self.device)
        mu_z, logvar_z = self.encoder(x)
        z = self.reparam(mu_z, logvar_z)
        x_recon = self.decoder(z)
        
        p1, p1_inter = self.disc(x)
        p2, p2_inter = self.disc(x_recon)
        prior_sample = torch.FloatTensor(np.random.normal(size=(x.shape[0], self.z_dim))).to(self.device) #expensive... :(
        x_sample = self.decoder(prior_sample) 
        p3, _ = self.disc(x_sample)

        #x_dec is the reconstructed output dec(enc(x))
        #mu_z and logvar_z are the output of the enc
        #p1,p2,p3 are the probability of a fake given by disc for x, dec(enc(x)) and dec(N(z|0,I)) respectively
        #p1_inter and p2_inter are intermediate output of disc for x and dec(enc(x)) respectively. 
        return x_recon, (mu_z, logvar_z), (p1, p2, p3), (p1_inter, p2_inter)
    

        
