#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 12:25:12 2019

@author: ben
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...tools import torchutils as tu
from ..inverse import inverse

class AE(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
  
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

#=============== TODO just use CNet as default? ===================

def default_conv2d(input_shape):    
    s1 = tu.conv_output_shape(input_shape, 16, kernel_size=4, stride=2)
    s2 = tu.conv_output_shape(s1, 32, kernel_size=4, stride=1)
    s3 = tu.conv_output_shape(s2, 64, kernel_size=4, stride=1)
    
    layers = [nn.Conv2d(input_shape[0], 16, kernel_size=4, stride=2),
              nn.Conv2d(16, 32, kernel_size=4, stride=1),
              nn.Conv2d(32, 64, kernel_size=4, stride=1)]
    
    return layers, [s1, s2, s3]

def default2D(input_shape, latent_shape, output_activation=nn.Identity(), share_weights=True, **kwargs):
    '''
        Constructs the default (convolutional) encoder and decoder for use in an AE.
        Arguments:
            input_shape (tuple): in CHW format
            latent_shape (int): latent space dimension
            share_weights (bool): whether the encoder and decoder share weights
        Returns:
            encoder, decoder
    '''
    input_shape = tu.as_shape(input_shape)
    latent_shape = tu.as_shape(latent_shape)

    layers, shapes = default_conv2d(input_shape)
    layers.append(nn.Linear(np.prod(shapes[-1]), latent_shape[0]))
    inverse_layers = inverse(*layers, share_weights=share_weights)
    
    class Encoder(nn.Module):
        
        def __init__(self):
            super(Encoder, self).__init__()
            self.conv1 = layers[0]
            self.conv2 = layers[1]
            self.conv3 = layers[2]
            self.linear1 = layers[3]
   
        def forward(self, x):
            x_ = F.leaky_relu(self.conv1(x))
            x_ = F.leaky_relu(self.conv2(x_))
            x_ = F.leaky_relu(self.conv3(x_)).view(x.shape[0], -1)
            x_ = self.linear1(x_)
            return x_
        
    class Decoder(nn.Module):
        
        def __init__(self):
            super(Decoder, self).__init__()
            self.linear1 = inverse_layers[0]
            self.conv1 = inverse_layers[1]
            self.conv2 = inverse_layers[2]
            self.conv3 = inverse_layers[3]
            self.output_activation = output_activation

        def forward(self, x):
            x_ = F.leaky_relu(self.linear1(x)).view(x.shape[0], *shapes[-1])
            x_ = F.leaky_relu(self.conv1(x_))
            x_ = F.leaky_relu(self.conv2(x_))
            x_ = self.output_activation(self.conv3(x_))
            return x_
        
    return Encoder(), Decoder()

    
'''
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyworld.toolkit.tools.datautils as du
import pyworld.algorithms.PCA as PCA #TODO UPDATE location (in pyworld.algorithms)

def label_colours(labels, alpha=0.8):
    colours = cm.rainbow(np.linspace(0,1,len(labels)))
    colours[:,3] *= alpha
    result = {}
    for i in range(colours.shape[0]):
        result[labels[i]] = colours[i]
    return result
    
    
def plot_latent2D(ae, x, y=None, fun=None, marker=".", fig=None, clf=True, alpha=0.8, 
                  title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, pause=0.001):
    
    
    if fig is None:
        fig = plt.figure()
    if clf:
        fig.clf()
        
        
    if fun is None:
        fun = ae.encode
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    
    ae.eval()
    labels = np.unique(y)
    colours = label_colours(labels, alpha)
    #print(colours)
    x = torch.FloatTensor(x)
    use_y = y is not None
    if y is None:
        y = np.ones(x.shape[0])
    line_handles = []
    label_handles = []

    z = ae.apply(x, lambda x : tu.to_numpy(fun(x)), (x.shape[0], ae.latent_dim), batch_size=64)
    if z.shape[1] > 2:
        z = PCA.PCA(z, k=2) #reduce dimensions again...!
        
    for label in np.unique(y):
        z_ = z[(y==label).squeeze()]
        line = plt.scatter(z_[:,0], z_[:,1], color=colours[label],  edgecolors='none', label=label, marker=marker)
        line_handles.append(line)
        label_handles.append(label)
        
    plt.legend(line_handles, label_handles, loc="upper right")
    plt.suptitle(title)
    plt.xlabel(xlabel)
    if use_y:
        plt.ylabel(ylabel)
    plt.draw()
    plt.pause(pause)
    return fig


    

class _LatentSeq(nn.Module):
    
    class _ParallelLayer:
        
        def __init__(self, *layers):
            self.layers = layers
            
        def __call__(self, *args):
            return tuple([layer(*args) for layer in self.layers])
    
    def __init__(self, *transforms, latent_dim):
        super(_LatentSeq, self).__init__()
        self.latent_dim = latent_dim
        self._device = 'cpu'
        self.layers = []
        
        for i, l in enumerate(transforms):
            #print(l)
            if isinstance(l, nn.Module):
                self.add_module(type(_LatentSeq).__name__ + str(i), l)
                self.layers.append(l)
            elif isinstance(l, tuple):
                for j, ll in enumerate(l):
                    if isinstance(ll, nn.Module):
                        self.add_module(type(_LatentSeq).__name__ + str(i) + "-" + str(j), ll)
                self.layers.append(_LatentSeq._ParallelLayer(*[g for g in l]))
            else:
                self.layers.append(l)
    
    def device(self, x):
        return x.to(self.device)
    
    def to(self, device):
        self._device = device
        super(_LatentSeq, self).to(device)
    
    def forward(self, x):
        #print(self.layers)
        for a in self.layers:
            x = a(x)
        return x
        
class Decoder(_LatentSeq):
    
    def __init__(self, *layers, latent_dim):
        super(Decoder, self).__init__(*layers, latent_dim=latent_dim)
    
class Encoder(_LatentSeq):
    
    def __init__(self, *layers, latent_dim):
        super(Encoder, self).__init__(*layers, latent_dim=latent_dim)
'''

