#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:47:39 2019

author: Benedict Wilkins
"""
from collections import namedtuple

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
class observation:
    
    def __init__(self, *data):
        self.__data = data
    
    def __getitem__(self, index):
        return self.__data[index]
    
    def __iter__(self):
        return self.__data.__iter__()
 
class s(observation):
    
    def __init__(self, *data):
        super(s, self).__init__(*data)
        self.state = data[0]
        
class r(observation):
    
    def __init__(self, *data):
        super(r, self).__init__(*data)
        self.reward = data[0]
        
class sa(observation):
    
    def __init__(self, *data):
        super(sa, self).__init__(*data)
        self.state = data[0]
        self.action = data[1]

class ss(observation):
    
    def __init__(self, *data):
        super(ss, self).__init__(*data)
        self.state = data[0]
        self.next_state = data[1]

class sr(observation):
    
    def __init__(self, *data):
        super(sr, self).__init__(*data)
        self.state = data[0]
        self.reward = data[1]
        
class sar(observation):
    
    def __init__(self, *data):
        super(sar, self).__init__(*data)
        self.state = data[0]
        self.action = data[1]
        self.reward = data[2]
        
class ars(observation):
    
    def __init__(self, *data):
        super(ars, self).__init__(*data)
        self.action = data[0]
        self.reward = data[1]
        self.next_state = data[2]
    
class sas(observation):
        
    def __init__(self, *data):
        super(sas, self).__init__(*data)
        self.state = data[0]
        self.action = data[1]
        self.next_state = data[2]
        
class sars(observation):
        
    def __init__(self, *data):
        super(sas, self).__init__(*data)
        self.state = data[0]
        self.action = data[1]
        self.reward = data[2]
        self.next_state = data[3]
    
'''