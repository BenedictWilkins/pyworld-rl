#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:33:22 2019

author: Benedict Wilkins
"""
import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple


import pyworld.toolkit.tools.datautils as du
from .Optimise import Optimiser


class ContrastiveOptimiser(Optimiser):
    
    def __init__(self, model, margin = 0.2, k = 16, lr = 0.0005):
        super(ContrastiveOptimiser, self).__init__(model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = du.CMA('loss')
        self.margin = margin
        self.k = int(k)
        
    def step(self, x, y):
        self.optim.zero_grad()
        loss = self.loss(x, y)
        self.cma.push(loss.item())
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def loss(self, x, y, **kwargs):
        # D = || x1 - x2 || 
        # (1-y)/2 D^2 + y/2 (max(0, margin - D))^2
        # where (x1, x2) are pairs, y = 0 if the pair is from the same class, y = 1 otherwise 
        raise NotImplementedError("TODO")
        
class PairContrastiveOptimiser(Optimiser):
    
    def __init__(self, model, margin = 0.2, k = 16, lr = 0.0005):
        super(ContrastiveOptimiser, self).__init__(model)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = du.CMA('loss')
        self.margin = margin
        self.k = int(k)
        
    def step(self, x1, x2):
        '''
            Step with pairs of input. ``(x1_i, x2_i)`` are a pair - share the same label
            (x1_i, x2_{j\neq i}) are considered to have different labels (i.e. are not a pair).
        '''
        return super(PairContrastiveOptimiser, self).step(x1, x2)
    
    def loss(self, x1, x2, **kwargs):
        pass
    
    
    
        
        
    