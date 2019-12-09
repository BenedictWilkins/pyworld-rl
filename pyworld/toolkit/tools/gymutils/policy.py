#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:25 2019

author: Benedict Wilkins
"""
import numpy as np

class value:
    
    class boltzmann:
        
        def __init__(self, t=1.):
            self.t = t
        
        def __call__(self, v):
            e = np.exp(v / self.t) 
            return e / e.sum()
        
    class weighted:
        
        def __call__(self, v):
            return v / v.sum()

def onehot(action, n):
    action_onehot = np.zeros(n)
    action_onehot[action] = 1
    return action_onehot

def onehot_policy(policy, n):
    return lambda s: onehot(policy(s), n)

def uniform_random_policy(action_space, onehot=False):
    policy = lambda _: action_space.sample()
    if onehot:
        policy = onehot_policy(policy, action_space.n)
    return policy

def e_greedy_policy(action_space, critic, epsilon=0.01, onehot=False): 
    def __policy(state):
        if np.random.uniform() > epsilon:
            return action_space.sample()
        else:
            return np.argmax(critic(state))
    
    policy = __policy 
    
    if onehot:
        policy = onehot_policy(policy, action_space.n)
    return policy

def probabilistic_policy(action_space, actor, onehot=False):
    actions = np.arange(action_space.n)
    def __policy(s):
        return np.random.choice(actions, p = actor(s))
    policy = __policy
    if onehot:
        policy = onehot_policy(policy, action_space.n)
    return policy

if __name__ == "__main__":
    from gym.spaces.discrete import Discrete
    action_space = Discrete(3)
    
    
    policy = e_greedy_policy(action_space)
    
    
    
    
    
    