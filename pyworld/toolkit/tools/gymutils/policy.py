#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:25 2019

author: Benedict Wilkins
"""
import numpy as np

class value:
    
    def boltzmann(t=1.):
        def __boltzmann(v):
            e = np.exp(v) * t
            return e / e.sum()
        return __boltzmann

    def weighted():
        return lambda v: v / v.sum()

def onehot(action, n):
    action_onehot = np.zeros(n)
    action_onehot[action] = 1
    return action_onehot

def onehot_policy(policy, n):
    return lambda s: onehot(policy(s), n)

def uniform_random_policy(env, onehot=False):
    policy = lambda _: env.action_space.sample()
    if onehot:
        policy = onehot_policy(policy, env.action_space.n)
    return policy

def value_based_policy(env, value_fun, value_p=value.weighted, onehot=False):
    actions = np.arange(env.action_space.n)
    def __policy(s):
        vs = np.array([value_fun(s,a) for a in actions])
        return np.random.choice(actions, p=value_p(vs))
    policy = __policy
    if onehot:
        policy = onehot_policy(policy, env.action_space.n)
    return policy