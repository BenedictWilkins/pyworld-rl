#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A set of decorates that can be applied to the OpenAI gym enviroment.step method 
for custom environments that will perform some transformation on state, action or reward.

Created on 2020-03-13 12:06:47

author: Benedict Wilkins
"""
from functools import wraps

from ..visutils import transform as T

def HWC(func):
    '''
        A state transformation that converts an state (image) from CHW format to HWC format.
    '''
    @wraps(func)
    def decorator(self, *args, **kwargs):
        state, reward, *info = func(self, *args, **kwargs)
        assert T.isCHW(state)
        state = T.HWC(state)
        return state, reward, *info

    return decorator

def CHW(func):
    '''
        A state transformation that converts an state (image) from HWC format to CHW format.
    '''
    @wraps(func)
    def decorator(self, *args, **kwargs):
        state, reward, *info = func(self, *args, **kwargs)
        assert T.isHWC(state)
        state = T.CHW(state)
        return state, reward, *info

    return decorator

def Float(func):
    
    @wraps(func)
    def decorator(self, *args, **kwargs):
        state, reward, *info = func(self, *args, **kwargs)
        state = T.to_float(state)
        return state, reward, *info

    return decorator

def Integer(func):
    
    @wraps(func)
    def decorator(self, *args, **kwargs):
        state, reward, *info = func(self, *args, **kwargs)
        state = T.to_integer(state)
        return state, reward, *info

    return decorator

