#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:02 2019

author: Benedict Wilkins
"""
import gym
import atari_py as ap

import numpy as np

from .. import datautils as du

from . import iterators
from . import policy
from . import wrappers
from . import transform
from . import mode


__all__ = ('iterators', 'policy', 'wrappers', 'transform', 'mode')

PYWORLD_ENVIRONMENTS = ['ObjectMover-v0', 'ObjectMover-v1', 'CoinCollector-NoJump-v0', 'CoinCollector-Easy-v0',
                        'CoinCollector-NoSpeed-v0', 'CoinCollector-Hard-v0']

no_transform  = ['ObjectMover-v0', 'ObjectMover-v1', 'CoinCollector-NoJump-v0', 'CoinCollector-Easy-v0',
                 'CoinCollector-NoSpeed-v0', 'CoinCollector-Hard-v0']

def name(env):
    return env.unwrapped.spec.id

def make(name = 'Pong-v0', binary=None, stack=None):
    '''
        Creates pre-wrapped environments from gym. The state space is reduce to (H,W,C) format - (84,84,1). The action space is unchanged.
        Arguments:
            name: of the envionment to make
            binary: a value [0,1] as the threshold for binarising the state space, or None if the binary transform is not required.
            stack: stacks N > 1 previous frames included as part of the state, new observation shape is [N * C, H, W], or None is frame stacking is not required.
        Returns:
            a gym environment
    '''
    env = gym.make(name) #'PongNoFrameskip-v4')

    if name in no_transform:
        return env
    
    env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.default)
    
    if binary is not None:
        env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.binary, threshold=binary)
        
    
    env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.chw)
    
        
    if stack is not None and stack > 1:
        env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.stack, stack=stack)
    
    return env

def atari0():
    '''
        Getter for a list of all the -v0 atari games provided in openAI gym
        
        Returns:
            The list of -v0 atari games
    '''
    import atari_py as ap
    games = []
    for g in ap.list_games():
        gg = g.split("_")
        gg = [g.capitalize() for g in gg]
        games.append(''.join(gg) + "-v0")
    return games

def video(env, policy=None):
    for state in iterators.GymIterator(env, policy, mode=mode.s):
        yield state.state

def dataset(env, policy=None, mode=mode.s, size=1000, onehot=False, progress=0):
    '''
        Constructs a ordered dataset given the environment, policy and iteration mode. Based on ``datautils.dataset``.
        
        Arguments:
            **env** (gym.env): the environment to collect from, should follow openai-gym conventions.
            
            **policy** (callable): a policy that maps states to actions i.e. a = p(s).
            
            **mode** (iterator.mode): iterator mode (see ``gymutils.iterators``). defaults to state mode (s).
            
            **size** (int): the size of the dataset.
            
            **onehot** (bool): if the (discrete) actions should be encoded as onehot vectors.
            
            **progress** (int): whether to print progress of constructions (every ``progress`` iterations), the default value ``0`` will not print progress information.
                
        Returns:
            the constructed dataset as a tuple of numpy arrays that follow the specified ``mode`` format.
    '''
    assert mode in iterators.iterators
    iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = False) 
    return mode(*du.dataset(iterator, size, progress=progress))

def datasets(env, policy=None, mode=mode.s, size=1000, onehot=False, epochs=1):
    assert mode in iterators.iterators
    iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = False)
    template = None 
    for e in range(epochs):
        template = mode(*du.dataset(iterator, size, template = template))
        
        yield template
  
def episode(env, policy=None, mode=mode.s, onehot=False, max_episode_size=10000):
    '''
        Creates an episode from the given environment and policy.
        Arguments:
            env: to play
            policy: to select actions from - a function with signature: action = policy(state)
            mode: one of `gyutils.mode`, default to state
            onehot: if actions should be given as onehot vectors
            max_episode_size: number of environment steps before the episode is cut short.
        Returns:

    '''

    env.reset()
    assert mode in iterators.iterators
    iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = True)
    iterator = iterators.itertools.islice(iterator, 0, max_episode_size)
    
    return mode(*du.pack(iterator))


def episodes(env, policy=None, mode=mode.s, onehot=False, epochs=1, max_episode_size=10000):
    env.reset()
    assert mode in iterators.iterators
    for i in range(epochs):
        env.reset() #maybe...
        iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = True)
        #iterator = iterators.itertools.islice(iterator, 0, max_episode_size)
        
        yield mode(*du.pack(iterator))

def returns(rewards, gamma=0.99):
    returns = np.zeros_like(rewards) #np.empty_like(rewards)
    returns[-1] = rewards[-1]
    for i in range(2, rewards.shape[0]+1):
        returns[-i] = rewards[-i] + gamma * returns[-i+1]
    return returns

    
    
    
    










