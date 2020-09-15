#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-07-2020 17:01:37
"""

__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


import torch
import torch.nn.functional as F
import numpy as np

from collections import defaultdict
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

        self.record = defaultdict(list) #used to record interesting values for plotting.
    
    def __call__(self, *args, **kwargs):
        self.base_optimiser.zero_grad()
        loss = self.step(*args, **kwargs)
        loss.backward()
        self.base_optimiser.step()
        return loss.item()

    def __str__(self):
        return 'model:' + type(self.model).__name__  + '\noptimiser:' + str(self.base_optimiser)

    def __repr__(self):
        return str(self)

class BCEOptimiser(TorchOptimiser):

    def __init__(self, model, logits=True, base_optimiser=None):
        super(BCEOptimiser, self).__init__(model, base_optimiser=base_optimiser)
        self.__loss_fun = (F.binary_cross_entropy, F.binary_cross_entropy_with_logits)[int(logits)]
        self.logits = logits

    def step(self, x, y):
        z = self.model(x)
        return self.__loss_fun(z, y)

class MSEOptimiser(TorchOptimiser):

    def __init__(self, model, base_optimiser=None):
        super(MSEOptimiser, self).__init__(model, base_optimiser=base_optimiser)

    def step(self, x, y):
        if isinstance(x, tuple):
            return F.mse_loss(self.model(*x), y)
        else:
            return F.mse_loss(self.model(x), y)