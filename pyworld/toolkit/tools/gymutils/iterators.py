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
from .policy import uniform_random_policy

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
        return state, action, reward, done, *info

    def reset(self, *_):
        return (self.env.reset(),)

# ============== ITERATORS ============== #

def s_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    yield m.s(state), False
    done = False
    while not done:
        action = policy(state)
        state, _, _, done, _ = step.step(action)
        yield m.s(state), done
        
def r_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        state, _, reward, done, _ = step.step(action)
        yield m.r(reward), done
        
def sa_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, _, done, _ = step.step(action)
        yield m.sa(state, action), False
        state = nstate
        print(done)
    #print(state, policy(state))
    print(True)
    yield m.sa(state, action), True
        
def sr_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, reward, done, _ = step.step(action)
        yield m.sr(state, reward), done
        state = nstate
    #yield m.sr(state, 0.), True #?? maybe..
        
def ss_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, _, done, _ = step.step(action)
        yield m.ss(state, nstate), done
        state = nstate

def sar_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, reward, done, _ = step.step(action)
        yield m.sar(state, action, reward), done
        state = nstate
    
def ars_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    yield m.ars(None, None, state), False ##??
    done = False
    while not done:
        action = policy(state)
        state, action, reward, done, _ = step.step(action)
        yield m.ars(action, reward, state), done

def sas_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, _, done, _ = step.step(action)
        yield m.sas(state, action, nstate), done
        state = nstate

def sars_iterator(env, policy, step=Step):
    step = step(env)
    state, = step.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, action, reward, done, _ = step.step(action)
        yield m.sars(state, action, reward, nstate), done
        state = nstate

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

    def __init__(self, env, policy=None, mode=m.s, onehot=False, episodic=True):
        self.episodic = episodic
        self._done = False
        self._step_transform = Step
        self._mode = mode
        
        if onehot:
            env = wrappers.OnehotUnwrapper(env)
        self._env = env

        if policy is None:
            policy = uniform_random_policy(env.action_space, onehot=onehot)
        self._policy = policy
        
        self._iterator_type = iterators[self._mode]
        self._iterator = None

    def reset(self):
        self._done = False
    
    def __getitem__(self, _slice):
        return itertools.islice(self, _slice.start, _slice.stop, _slice.step)
    
    def __next__(self):
        if self.episodic:
            #print(self.episodic)
            result, _ = next(self._iterator)
            return result
        else:
            try:
                result, done = next(self._iterator) #TODO remove done..
                return result
            except StopIteration:
                self.reset()
                result, done = next(self._iterator)
                return result
        
    def __iter__(self):
        self._iterator = self._iterator_type(self._env, self._policy, self._step_transform)
        return self
    
iterators = {m.s:s_iterator, m.r:r_iterator, m.sa:sa_iterator, m.ss:ss_iterator, m.sr:sr_iterator, 
             m.sar:sar_iterator, m.ars:ars_iterator, m.sas:sas_iterator, 
             m.sars:sars_iterator}
    

