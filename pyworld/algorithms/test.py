#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:48:37 2019

@author: ben
"""
import torch

from torch.distributions import Categorical

logits = torch.FloatTensor([0.1, 0.2, 0.6])

d = Categorical(logits)
print(d.probs)

