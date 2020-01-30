#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:55:33 2019

@author: ben
"""

from tensorboardX import SummaryWriter
from abc import ABC, abstractmethod

class Variable:

    def __init__(self, value=None):
        self._trackers = []
        self._value = value
    
    def value(self, value=None):
        if value is not None:
            self._value = value
            for tracker in self._trackers:
                tracker.update(self)
        return self.value
    
    def __str__(self):
        return self._value.__str__()
    
    def __repr__(self):
        return self._value.__repr__()
    
class Tracker(ABC):
    
    def __init__(self, value, label, update):
        value._trackers.append(self)
        self.label = label
        self.step = 0
        self._update = update
        if value._value is not None:  
            self.update(value)
        
    def update(self, arg):
        self._update(arg, self.label, self.step)
        self.step += 1

if __name__ == "__main__":
    
    v = Variable(0)
    t = Tracker(v, 'value', lambda x, label, step: print(step, label, x))
    
    v.value(1)