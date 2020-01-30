#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:22:46 2019

@author: Benedict Wilkins
"""

try:
    import torch
    import numpy as np
except:
    pass

from pyworld.toolkit.tools.datautils.accumulate import EMA, CMA

class Optimiser:
    
    def __init__(self, model):
        self.model = model

    def step(self, *args):
        pass

    def __call__(self, *args, **kwargs):
        pass

class TorchOptimiser(Optimiser):

    def __init__(self, model, base_optimiser=None):
        super(TorchOptimiser, self).__init__(model)
        if base_optimiser is None:
            self.base_optimiser = torch.optim.Adam(self.model.parameters(), lr=0.0005)
        else:
            self.base_optimiser = base_optimiser
    
    def __call__(self, *args, **kwargs):
        self.base_optimiser.zero_grad()
        loss = self.step(*args, **kwargs)
        loss.backward()
        self.base_optimiser.step()

    def __str__(self):
        return 'model:' + type(self.model).__name__  + '\nloss:' + self._loss.__name__ + '\noptimiser:' + str(self.optim)

    def __repr__(self):
        return str(self)
        
