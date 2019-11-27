#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:02 2019

author: Benedict Wilkins
"""
import gym

from .. import datautils as du

from . import iterators
from . import policy
from . import wrappers
from . import transformation
from . import mode

__all__ = ('iterators', 'policy', 'wrappers', 'transformation', 'mode')

def env(name = 'Pong-v0', binary=None):
    '''
        Creates pre-wrapped environments from gym. The state space is reduce to (H,W,C) format - (84,84,1). The action space is unchanged.
        Arguments:
            name: of the envionment to make
            binary: a value [0,1] as the threshold for binarising the state space, or None if the binary transformation is not required.
        Returns:
            a gym environment
    '''
    env = gym.make(name) #'PongNoFrameskip-v4')
    env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.default)
    if binary is not None:
        env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.binary, binary)
    env = wrappers.ObservationWrapper(env, wrappers.ObservationWrapper.mode.chw)
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
  
def episode(env, policy=None, mode=mode.s, onehot=False):
    assert mode in iterators.iterators
    iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = True)
    return mode(*du.pack(iterator))


def episodes(env, policy=None, mode=mode.s, onehot=False, epochs=1):
    assert mode in iterators.iterators
    for i in range(epochs):
         iterator = iterators.GymIterator(env, policy, mode, onehot=onehot, episodic = True)
         yield mode(*du.pack(iterator))

