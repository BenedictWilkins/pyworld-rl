#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:19:12 2019

@author: ben
"""
import numpy as np

from . import optimise
from . import rl

__all__ = ('optimise', 'rl')

def PCA(x, k=2):
    x = x - np.mean(x, 0)
    U,S,V = np.linalg.svd(x.T)
    return np.matmul(x, U[:,:k])