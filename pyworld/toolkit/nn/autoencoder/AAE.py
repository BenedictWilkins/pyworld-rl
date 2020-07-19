#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:57:20 2019

@author: ben
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import AE
from ...tools import torchutils as tu
from ..inverse import inverse

def prior_gaussian(size, device='cpu'):
    return torch.as_tensor(np.random.randn(*size), device=device, dtype=torch.float)

class AAE(AE.AE):
    
    def __init__(self, encoder, decoder, discriminator, prior_latent=prior_gaussian):
        super(AAE, self).__init__(encoder, decoder)
        self.discriminator = discriminator
        self.prior_latent = prior_latent

    def forward(self, x):
        z = self.encode(x)
        x_ = self.decode(z)
        p_real = self.discriminator(self.prior_latent(z.shape, self.device))
        p_fake = self.discriminator(z)
        return x_, p_real, p_fake
    
    def reconstruct(self, x):
        return self.decode(self.encode(x))
    
def default2D(input_shape, latent_dim, share_weights=True):
    assert len(input_shape) == 2
    s1 = tu.conv_output_shape(input_shape, kernel_size=4, stride=2)
    s2 = tu.conv_output_shape(s1, kernel_size=4, stride=1)
    s3 = tu.conv_output_shape(s2, kernel_size=4, stride=1)
    
    layers = [nn.Conv2d(1, 64, kernel_size=4, stride=2),
              nn.Conv2d(64, 32, kernel_size=4, stride=1),
              nn.Conv2d(32, 16, kernel_size=4, stride=1),
              nn.Linear(s3[0] * s3[1] * 16, latent_dim)]

    inverse_layers = inverse(*layers, share_weights=share_weights)

    encoder = AE.Encoder(layers[0], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64),
                         layers[1], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(32),
                         layers[2], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(16),
                         lambda x: x.view(x.shape[0], -1),
                         layers[3],  #nn.BatchNorm1d(latent_dim), # nn.Tanh(),
                         latent_dim=latent_dim)
    
    decoder = AE.Decoder(inverse_layers[0], nn.LeakyReLU(0.2, inplace=True), lambda x : x.view(x.shape[0], 16, *s3),
                         inverse_layers[1], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(32),
                         inverse_layers[2], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64),
                         inverse_layers[3], #nn.Sigmoid(),  
                         latent_dim=latent_dim)
     
    discriminator = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True),
                                  #nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(512, 1)) #logits
    
    return encoder, decoder, discriminator

def reparam(args):
    mean, logvar = args
    return torch.FloatTensor(mean.size()).normal_().to(mean.device) * torch.exp(logvar / 2.) + mean

def default2D_VAAE(input_shape, latent_dim, share_weights=True):
    assert len(input_shape) == 2
    s1 = tu.conv_output_shape(input_shape, kernel_size=4, stride=2)
    s2 = tu.conv_output_shape(s1, kernel_size=4, stride=1)
    s3 = tu.conv_output_shape(s2, kernel_size=4, stride=1)
    
    layers = [nn.Conv2d(1, 64, kernel_size=4, stride=2),
              nn.Conv2d(64, 32, kernel_size=4, stride=1),
              nn.Conv2d(32, 16, kernel_size=4, stride=1)]
    
    l_mu = nn.Linear(s3[0] * s3[1] * 16, latent_dim)
    l_logvar = nn.Linear(s3[0] * s3[1] * 16, latent_dim)
    l2 = nn.Linear(latent_dim, s3[0] * s3[1] * 16)
    
    inverse_layers = AE.construct_inverse(*layers, share_weights=share_weights)
    print("encoder")
    encoder = AE.Encoder(layers[0], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64),
                         layers[1], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(32),
                         layers[2], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(16),
                         lambda x: x.view(x.shape[0], -1),
                         (l_mu, l_logvar), reparam,
                         latent_dim=latent_dim)
    print("decoder")
    decoder = AE.Decoder(l2, nn.LeakyReLU(0.2, inplace=True), lambda x : x.view(x.shape[0], 16, *s3),
                         inverse_layers[0], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(32),
                         inverse_layers[1], nn.LeakyReLU(0.2, inplace=True), nn.BatchNorm2d(64),
                         inverse_layers[2], #nn.Sigmoid(),  
                         latent_dim=latent_dim)
    print("disc")
    discriminator = nn.Sequential(nn.Linear(latent_dim, 512), nn.LeakyReLU(0.2, inplace=True),
                                  #nn.Linear(512, 256), nn.LeakyReLU(0.2, inplace=True),
                                  nn.Linear(512, 1)) #logits
    
    return encoder, decoder, discriminator
    



if __name__ == "__main__":
    
    device = tu.device()
    encoder, decoder, discrim = default2D_VAAE((28,28), 2)
    encoder.to(device)
    t = torch.as_tensor(np.random.randn(1,1,28,28), dtype=torch.float).to(device)
    print(encoder(t))
    
        
        