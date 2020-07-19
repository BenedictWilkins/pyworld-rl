#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:17 2019

author: Benedict Wilkins
"""
import numpy as np
import copy
import itertools
from functools import wraps

from . import wrappers
from . import mode as m


from . import policy as P
from ..visutils import transform as T

def _d_CHW(func):
    @wraps(func)
    def decorator(self, action=None):
        r = func(self, action)
        return (T.CHW(r[0]), *r[1:])
    return decorator

def _d_HWC(func):
    @wraps(func)
    def decorator(self, action=None):
        r = func(self, action)
        return (T.HWC(r[0]), *r[1:])
    return decorator

def _d_Float(func):
    @wraps(func)
    def decorator(self, action=None):
        r = func(self, action)
        return (T.to_float(r[0]), *r[1:])
    return decorator

def _d_Integer(func):
    @wraps(func)
    def decorator(self, action=None):
        r = func(self, action)
        return (T.to_integer(r[0]), *r[1:])
    return decorator

def _d_OneHot(func):
    @wraps(func)
    def decorator(self, action=None):
        state, action, reward, done, *info = func(self, action)
        _action = np.zeros(self.env.action_space.n, dtype=np.float32)
        _action[action] = 1.
        return (state, _action, reward, done, *info)
    return decorator

class StepMeta(type):

    def __new__(mcls, name, bases, local):
        return super(StepMeta, mcls).__new__(mcls, name, bases, local)

    @property
    def CHW(self):
        return type(self.__name__ + ".CHW", (self,), {'step':_d_CHW(self.step), 'reset':_d_CHW(self.reset)})

    @property
    def HWC(self):
        return type(self.__name__ + ".HWC", (self,), {'step':_d_HWC(self.step), 'reset':_d_HWC(self.reset)})

    @property
    def Float(self):
        return type(self.__name__ + ".Float", (self,), {'step':_d_Float(self.step), 'reset':_d_Float(self.reset)})

    @property
    def Integer(self):
        return type(self.__name__ + ".Integer", (self,), {'step':_d_Integer(self.step), 'reset':_d_Integer(self.reset)})

    @property
    def OneHot(self):
        return type(self.__name__ + ".Integer", (self,), {'step':_d_OneHot(self.step)})

class Step(metaclass=StepMeta):

    def __init__(self, env):
        self.env = env
    
    def step(self, action):
        state, reward, done, *info = self.env.step(action)
        return (state, action, reward, done, *info)

    def reset(self, *_):
        return (self.env.reset(),)

class GymIteratorMeta(type):

    '''
        This is rather nebulous, but essentially it adds all of the properties of the 
        metaclass StepMeta to GymIterator. This is to allow a chain of calls that will modify 
        the particular iterator and apply a transform to any generated states, actions, or rewards.

        Example:
            iterator = GymIterator(env)
            for obs in iterator.CHW.Float:
                pass #states in obs will be in CHW float32 [0-1] format
    '''

    def __new__(mcls, name, bases, local):
        step_properties = {k:v for k,v in StepMeta.__dict__.items() if isinstance(v, property)}
        #print(step_properties)

        step_properties = {k:property(lambda self, p=v: GymIteratorMeta.set_step_transform(self, p)) for k,v in step_properties.items()}
        
        #print(step_properties)
        local.update(step_properties)
        #print(local)
        return super(GymIteratorMeta, mcls).__new__(mcls, name, bases, local)
    
    def set_step_transform(self, step): #setter for an instance of GymIterator
        self._iterator = None
        new_iterator = copy.deepcopy(self)
        new_iterator._step_transform = step.fget(new_iterator._step_transform)
        return new_iterator

class GymIterator(metaclass=GymIteratorMeta):

    def __init__(self, env, policy=None, mode=m.s):
        self._step_transform = Step

        assert mode in iterators
        self._mode = mode
        
        #if onehot:
        #    env = wrappers.OnehotUnwrapper(env)
            
        self._env = env

        if policy is None:
            policy = P.uniform(env.action_space)

        self._policy = policy
        
        self._iterator_type = iterators[self._mode]
        self._iterator = None

    def __iter__(self):
        self._env.reset()
        self._iterator = self._iterator_type(self._env, self._policy, self._step_transform)
        return self._iterator


def dataset(env, policy=None, mode=m.s, size=10000):
    pass #TODO

def episode(env, policy=None, mode=m.s, max_length=10000):
    '''
        Creates an episode from the given environment and policy.
        Arguments:
            env: to play
            policy: to select actions from - a function with signature: action = policy(state)
            mode: one of `gyutils.mode`, default to state
            max_length: number of environment steps before the episode is cut short.
    '''
    iterator = GymIterator(env, policy, mode)
    iterator = itertools.islice(iterator, 0, max_length)
    return m.pack(iterator)
  
def episodes(env, policy, mode=m.s, max_length=10000, n=10):
    '''
        TODO

        Example:
            for episode in episodes(env, policy):
                # do something with the episode
    '''
    iterator = GymIterator(env, policy, mode)
    for i in range(n):
        _iterator = itertools.islice(iterator, 0, max_length)
        yield m.pack(_iterator)

# ============================ ITERATORS ============================ #
# Each iterator corresponds to a mode in gymutils.mode, and is used to gather states, 
# actions, and/or rewards.
# =================================================================== #

def s_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    yield m.s(state)
    done = False
    while not done:
        action = policy(state)
        state, _, _, done, _ = step.step(action)
        yield m.s(state)
        
def r_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        state, _, reward, done, _ = step.step(action)
        yield m.r(reward)
        
def sa_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, _, done, _ = step.step(action)
        yield m.sa(state, action)
        state = nstate
    #print(state, policy(state))
    yield m.sa(state, action)
        
def sr_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, reward, done, _ = step.step(action)
        yield m.sr(state, reward)
        state = nstate
    #yield m.sr(state, 0.), True #?? maybe..
        
def ss_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, _, done, _ = step.step(action)
        yield m.ss(state, nstate)
        state = nstate

def sar_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, reward, done, _ = step.step(action)
        yield m.sar(state, action, reward)
        state = nstate
    
def ars_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    yield m.ars(None, None, state)
    done = False
    while not done:
        action = policy(state)
        state, action, reward, done, _ = step.step(action)
        yield m.ars(action, reward, state)

def sas_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, _, done, _ = step.step(action)
        yield m.sas(state, action, nstate)
        state = nstate

def sars_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, reward, done, _ = step.step(action)
        yield m.sars(state, action, reward, nstate)
        state = nstate

iterators = {m.s:s_iterator, m.r:r_iterator, m.sa:sa_iterator, m.ss:ss_iterator, m.sr:sr_iterator, 
             m.sar:sar_iterator, m.ars:ars_iterator, m.sas:sas_iterator, 
             m.sars:sars_iterator}
    

