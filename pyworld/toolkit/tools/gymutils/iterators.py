#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:17 2019

author: Benedict Wilkins
"""

import itertools

from . import wrappers
from . import mode as m
from .policy import uniform_random_policy

def s_iterator(env, policy):
        state = env.reset()
        yield m.s(state), False
        done = False
        while not done:
            action = policy(state)
            state, _, done, _ = env.step(action)
            yield m.s(state), done
        
def r_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield m.r(reward), done
        
def sa_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield m.sa(state, action), done
        state = nstate
        
def sr_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield m.sr(state, reward), done
        state = nstate
        
        
def ss_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield m.ss(state, nstate), done
        state = nstate

def sar_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield m.sar(state, action, reward), done
        state = nstate

def ars_iterator(env, policy):
    state = env.reset()
    yield m.ars(None, None, state), False ##??
    done = False
    while not done:
        action = policy(state)
        state, reward, done, _ = env.step(action)
        yield m.ars(action, reward, state), done

def sas_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, _, done, _ = env.step(action)
        yield m.sas(state, action, nstate), done
        state = nstate

def sars_iterator(env, policy):
    state = env.reset()
    done = False
    while not done:
        action = policy(state)
        nstate, reward, done, _ = env.step(action)
        yield m.sars(state, action, reward, nstate), done
        state = nstate
    
class GymIterator:

    def __init__(self, env, policy=None, mode=m.s, onehot=False, episodic=True):
        self.episodic = episodic
        self.__done = False
        
        if onehot:
            env = wrappers.OnehotUnwrapper(env)
        self.__env = env

        if policy is None:
            policy = uniform_random_policy(env.action_space, onehot=onehot)
        self.__policy = policy
        
        self.__iterator_type = iterators[mode]
        self.__iterator = self.__iterator_type(self.__env, self.__policy)
     
    def reset(self):
        self.__done = False
        self.__iterator = self.__iterator_type(self.__env, self.__policy)
        
    def __getitem__(self, _slice):
        return itertools.islice(self, _slice.start, _slice.stop, _slice.step)
    
    '''
    def __next__(self):
        if self.__done:
            self.reset()
            if self.episodic:
                raise StopIteration
        result, self.__done = next(self.__iterator)
        return result
    '''
    
    def __next__(self):
        if self.episodic:
            #print(self.episodic)
            result, _ = next(self.__iterator)
            return result
        else:
            try:
                result, done = next(self.__iterator)
                return result
            except StopIteration:
                self.reset()
                result, done = next(self.__iterator)
                return result
        
    def __iter__(self):
        return self
    
iterators = {m.s:s_iterator, m.r:r_iterator, m.sa:sa_iterator, m.ss:ss_iterator, m.sr:sr_iterator, 
             m.sar:sar_iterator, m.ars:ars_iterator, m.sas:sas_iterator, 
             m.sars:sars_iterator}
    