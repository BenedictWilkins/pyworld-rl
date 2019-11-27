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
from pyworld.algorithms.optimise import Optimiser



class ContrastiveOptimiser(Optimiser):
    
    def __init__(self, model, margin = 0.2, k = 16, lr = 0.0005):
        super(ContrastiveOptimiser, self).__init__(model)
        self.mode
        
    