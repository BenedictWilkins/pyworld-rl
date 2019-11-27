#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:11:46 2019

author: Benedict Wilkins
"""
import gym
import numpy as np
import cv2
from collections import namedtuple
#from enum import Enum

def assert_box(space):
    assert isinstance(space, gym.spaces.Box)
    
def assert_interval(space, high=255, low=0):
    u = np.unique(space.high.ravel())
    assert len(u) == 1
    assert u[0] == high
    u = np.unique(space.low.ravel())
    assert len(u) == 1
    assert u[0] == low
    
def assert_unique(space):
    high = np.unique(space.high.ravel())
    assert len(high) == 1
    low = np.unique(space.low.ravel())
    assert len(low) == 1
    return low[0], high[0] 
    
class __OM_Gray:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(*env.observation_space.shape[:2], 1), dtype=np.float32)
    
    def __call__(self, state, *args):
        state = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float) / 255.
        return state.reshape(*state.shape, 1)
    
class __OM_Interval:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)
    
    def __call__(self, state, *args):
        state.astype(np.float) / 255.0
        
class __OM_Crop:
    
    def __init__(self, env, shape):
        assert_box(env.observation_space)
        self.observation_space = env.observation_space #gym.spaces.Box(low=env.observation_space.low, high=env.observation_space.high, shape=shape, dtype=np.float32)
    
    def __call__(self, state, shape):
        raise NotImplementedError()
    
class __OM_Resize:
    
    def __init__(self, env, shape):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)
    
    def __call__(self, state):
        return cv2.resize(state, self.observation_space.shape, interpolation=cv2.INTER_AREA)


class __OM_Binary:
    def __init__(self, env, threshold):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        
        self.threshold = threshold * (high - low)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=env.observation_space.shape, dtype=np.float32)
    
    def __call__(self, state):
        return np.where(state > self.threshold, 1., 0.)
        
class __OM_Default:
    
    def __init__(self, env):
        assert_box(env.observation_space)
        assert_interval(env.observation_space)
        self.observation_space = gym.spaces.Box(low=0., high=1.0, shape=(84, 84, 1), dtype=np.float32)
        
    def __call__(self, state):
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float32) #gray
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA) #resize
        img = img[18:102, :] / 255.0 #crop
        return img.reshape((*img.shape, 1))  

class __OM_CHW:
    
    '''
        Transforms states from HWC to CHW format suitable for pytorch image processing. 
    '''
    
    def __init__(self, env):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        #assumes HWC format
        channels = env.observation_space.shape[2]
        height = env.observation_space.shape[0]
        width = env.observation_space.shape[1]
        
        assert channels == 1 or channels == 3 or channels == 4
        self.observation_space = gym.spaces.Box(low = low, high = high, shape=(channels, height, width), dtype = env.observation_space.dtype)
        
    def __call__(self, state):
        return state.transpose((2,0,1))
        #return np.swapaxes(state, 0, 2)

class __OM_HWC:
    
    '''
        Transforms states from CHW to HWC format suitable for opencv (or PIL) processing.
    '''
    
    def __init__(self, env):
        assert_box(env.observation_space)
        low, high = assert_unique(env.observation_space)
        #assumes CHW format
        channels = env.observation_space.shape[0]
        height = env.observation_space.shape[1]
        width = env.observation_space.shape[2]
        
        assert channels == 1 or channels == 3 or channels == 4
        self.observation_space = gym.spaces.Box(low = low, high = high, shape=(height, width, channels), dtype = env.observation_space.dtype)
        
    def __call__(self, state):
        return state.transpose((1,2,0))
'''
class observation_mode(Enum):
    gray = __OM_Gray
    interval = __OM_Interval
    crop = __OM_Crop
    resize = __OM_Resize
    chw = __OM_CHW
    default = __OM_Default
'''

mode = namedtuple('observation_transform', 'gray interval crop resize binary chw default')(
                                    __OM_Gray, __OM_Interval, __OM_Crop, __OM_Resize, __OM_Binary, __OM_CHW, __OM_Default)
