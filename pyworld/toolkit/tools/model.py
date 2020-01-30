#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:41:46 2019

@author: ben
"""
from abc import ABC, abstractmethod

class Model(ABC):
    
    def __init__(self):
        pass

class LearningModel(Model):
    
    def __init__(self, model, loss, optim):
        super(LearningModel, self).__init__()
        self.model = model
        self.loss = loss
        self.optim = optim
    
    @abstractmethod
    def step(*args):
        pass