#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:47:39 2019

author: Benedict Wilkins
"""
from collections import namedtuple
'''
s = namedtuple('s', ['state'])
s.__new__.__defaults__ = (None,)
r = namedtuple('r', ['reward'])
r.__new__.__defaults__ = (None,)
sa = namedtuple('sa', ['state', 'action'])
sa.__new__.__defaults__ = (None,None)
ss = namedtuple('ss', ['state', 'nstate'])
ss.__new__.__defaults__ = (None,None)
sr = namedtuple('sr', ['state', 'reward'])
sr.__new__.__defaults__ = (None,None)
sar = namedtuple('sar', ['state', 'action', 'reward'])
sar.__new__.__defaults__ = (None,None,None)
ars = namedtuple('ars', ['action', 'reward', 'nstate'])
ars.__new__.__defaults__ = (None,None,None)
sas = namedtuple('sas', ['state', 'action', 'nstate'])
sas.__new__.__defaults__ = (None,None,None)
sars = namedtuple('sars', ['state', 'action', 'reward', 'nstate'])
sars.__new__.__defaults__ = (None,None,None,None)
'''

import numpy as np

class observation:
    
    def __init__(self, *data):
        self.__data = data
    
    def __getitem__(self, index):
        return self.__data[index]
    
    def __iter__(self):
        return self.__data.__iter__()

    def __str__(self):
        return "observation-{0}".format(self.__class__.__name__)

    def __repr__(self):
        return str(self)

class s(observation):
    
    def __init__(self, state=None, **kwargs):
        super(s, self).__init__(state)
    
    @property
    def state(self):
        return self[0]
        
class r(observation):
    
    def __init__(self, reward=None, **kwargs):
        super(r, self).__init__(reward)

    @property
    def reward(self):
        return self[0]

class sa(observation):
    
    def __init__(self, state=None, action=None, **kwargs):
        super(sa, self).__init__(state, action)

    @property
    def state(self):
        return self[0]

    @property
    def action(self):
        return self[1]

class ss(observation):
    
    def __init__(self, state=None, nstate=None, **kwargs):
        super(ss, self).__init__(state, nstate)

    @property
    def state(self):
        return self[0] 

    @property
    def nstate(self):
        return self[1]


class sr(observation):
    
    def __init__(self, state=None, reward=None, **kwargs):
        super(sr, self).__init__(state, reward)

    @property
    def state(self):
        return self[0] 

    @property
    def reward(self):
        return self[1]

        
class sar(observation):
    
    def __init__(self, state=None, action=None, reward=None, **kwargs):
        super(sar, self).__init__(state, action, reward)

    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def reward(self):
        return self[2] 

class ars(observation):
    
    def __init__(self, action=None, reward=None, nstate=None, **kwargs):
        super(ars, self).__init__(action, reward, nstate)

    @property
    def action(self):
        return self[0]

    @property
    def reward(self):
        return self[1] 

    @property
    def nstate(self):
        return self[2] 

class sas(observation):
        
    def __init__(self, state=None, action=None, nstate=None, **kwargs):
        super(sas, self).__init__(state, action, nstate)
    
    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def nstate(self):
        return self[2] 
        
class sars(observation):
        
    def __init__(self, state=None, action=None, reward=None, nstate=None, **kwargs):
        super(sars, self).__init__(state, action, reward, nstate)

    @property
    def state(self):
        return self[0] 

    @property
    def action(self):
        return self[1]

    @property
    def reward(self):
        return self[2]

    @property
    def nstate(self):
        return self[3] 

def pack(observations):
    """ 
        Packs a list of observations
    """
    return tuple([np.array(d) for d in [i for i in zip(*observations)]])
