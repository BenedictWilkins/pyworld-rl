#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 18:48:45 2019

@author: ben
"""

from . import AE
import numpy as np

class DAE(AE.AE):
    
    def __init__(self, encoder, decoder, noise, noise_p):
        super(DAE, self).__init__(encoder, decoder)
        self.noise = noise
        self.noise_p = noise_p
        
    def forward(self, x):
        x = self.noise(x, self.noise_p)
        return super(DAE, self).forward(x)
    
    def info(self, x):
        return {'noise':self.noise.__name__, 'noise_p':self.noise_p}
    
def noise_gaussian(x, p = 0.3):
    return x + np.random.randn(*x.shape)
    
def noise_pepper(x, p = 0.3):
    return x * (np.random.uniform(size=x.shape) > p)

def noise_saltpepper(x, p = 0.3):
    i = np.random.uniform(size=x.shape) < p
    x[i] = (np.random.uniform(size=np.sum(i)) > 0.5)
    return x


