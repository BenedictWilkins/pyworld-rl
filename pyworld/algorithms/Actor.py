#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 14:16:13 2019

author: Benedict Wilkins
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

raise NotImplementedError() #... TODO

class Actor:
    
    def __init__(self, model, logits=False):
        self.model = model
        self.distribution = None
        if logits:
            self.__distribution = distribution_logits
        else:
            self.__distribution = self.distribution_probs
        self.distribution = None
        self.probs = None
        self.logits = None
                
    def __distribution_logits(self, logits):
        return Categorical(logits = logits)
    
    def __distribution_probs(self, probs):
        return Categorical(probs = probs)
    
    def __call__(self, state):
        self.distribution = self.__distribution(self.model(state))    
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
    def log_probs(self, state):
        
        
    def policy(self, state):
        