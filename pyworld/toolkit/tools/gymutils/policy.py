#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:25 2019

author: Benedict Wilkins
"""
from abc import ABC, abstractmethod
import numpy as np
import gym

from . import spaces

try:
    import torch
except:
    pass # torch is not installed... ehhh. make this streamlined


class P:
    
    class boltzmann:
        
        def __init__(self, t=1.):
            self.t = t
        
        def __call__(self, v):
            e = np.exp(v / self.t) 
            return e / e.sum()
        
    class weighted:
        
        def __call__(self, v):
            return v / v.sum()

class Policy(ABC):

    def __init__(self, action_space):
        self.action_space = action_space
        def _sample(*args, **kwargs):
            raise NotImplementedError("\"sample\" attribute must be set for abstract class Policy")
        self.sample = _sample
    
    def __call__(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

class DiscretePolicy(Policy):

    def __init__(self, action_space, dtype=np.int64):
        if isinstance(action_space, int):
            action_space = gym.spaces.Discrete(action_space)
        action_space.dtype = dtype
        super(DiscretePolicy, self).__init__(action_space)

class ContinuousPolicy(Policy):

    def __init__(self, action_space, dtype=np.float32):
        action_space.dtype = dtype
        super(ContinuousPolicy, self).__init__(action_space)

class NeuralPolicy(DiscretePolicy):

    def __init__(self, action_space, nn, p=lambda x: x, dtype=np.int64):
        super(NeuralPolicy, self).__init__(action_Space, dtype=dtype)
        self.nn = nn
        self.p = p
    
    def sample(self, *args, **kwargs):
        v = self.nn(*args, **kwargs)
        p = self.p(v)
        i = torch.multinomial(p, 1).squeeze()

        raise NotImplementedError("TODO")

def onehot(policy, dtype=np.float32):
    assert np.issubdtype(policy.action_space.dtype, np.integer)
    assert isinstance(policy.action_space, gym.spaces.Discrete)

    sample =  policy.sample
    policy.action_space = spaces.OneHot(policy.action_space.n, dtype)

    def oh(x):
        r = np.zeros(policy.action_space.shape[0], dtype=policy.action_space.dtype)
        r[sample(x)] = 1
        return r
    policy.sample = oh
    
    return policy



    
def uniform(action_space, dtype=np.int64):
    """ Uniform random policy that selects an action uniformly from the given (discrete) action space.

    Args:
        action_space (gym.spaces.Discrete, int): action space
        dtype (type, optional): dtype of a sampled action. Defaults to np.int64.

    Returns:
        DiscretePolicy: the policy
    """
    policy = DiscretePolicy(action_space, dtype=dtype)
    action_space = policy.action_space
    policy.sample = lambda *args, **kwargs: action_space.dtype(np.random.randint(0, action_space.n))
    return policy

def random(action_space, p=None, dtype=np.int64): #TODO assume discrete action_space?
    """ Random policy that selects an action from the given (discrete) action space according to the given probabilities p.

    Args:
        action_space (gym.spaces.Discrete, int): action space
        p (sequence, optional): action probabilities associated with each action. Defaults to uniform probability.
        dtype (type, optional): dtype of a sampled action. Defaults to np.int64.

    Returns:
        DiscretePolicy: the policy
    """
    if p is None:
        return uniform(action_space, dtype=dtype)
    
    policy = DiscretePolicy(action_space, dtype=dtype)
    assert len(p) == policy.action_space.n
    sample_space = np.arange(0, policy.action_space.n)
    policy.sample = lambda *args, **kwargs: action_space.dtype(np.random.choice(sample_space, p=p))
    return policy
    

# TODO update others to follow DiscretePolicy!

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
    
    
    
    
    
    