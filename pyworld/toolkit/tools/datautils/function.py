#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:43:18 2020

@author: ben
"""

import numpy as np

def convolve1D(signal, kernel):
    return np.convolve(signal, kernel)[int(kernel.shape[0]/2):-int(kernel.shape[0]/2)]

def convolve1D_gauss(signal, sigma = 1., kernel_size=None):
    if kernel_size is None:
        kernel_size = 5
        
    x = np.linspace(-kernel_size/2, kernel_size/2, num = kernel_size)
    kernel = np.exp(-np.power(x, 2.) / (2 * np.power(sigma, 2.)))
    #kernel = kernel / np.sum(kernel)
    
    return np.convolve(signal, kernel)[int(kernel.shape[0]/2):-int(kernel.shape[0]/2)] 



if __name__ == "__main__":
    
    signal = np.random.choice([1,0], size = 10, p=[0.2,1-0.2])
    print(signal)
    smooth = convolve1D_gauss(signal)
    print(smooth)
    print(signal.shape)
    print(smooth.shape)
    