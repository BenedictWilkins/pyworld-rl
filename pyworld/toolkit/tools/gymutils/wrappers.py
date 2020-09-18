#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:55 2019

author: Benedict Wilkins
"""
import gym
import numpy as np
import copy

from collections import deque

from . import transform

from ..fileutils import save as fu_save
from ..visutils import transform as T

import skimage

class EpisodeRecordWrapper(gym.Wrapper):

    def __init__(self, env, path, compress=True):
        super(EpisodeRecordWrapper, self).__init__(env)
        self.states = []
        self.actions = []
        self.rewards = []

        self.state_t = None
        self.reward_t = None

        self.path = path

        self.already_done = False

    def step(self, action_t):
        assert not self.already_done #dont save multiple times just because someone isnt calling reset!
         
        state, reward, done, info = self.env.step(action_t)
        state = np.copy(state)


        self.states.append(self.state_t)
        self.actions.append(action_t)
        self.rewards.append(self.reward_t)

        self.state_t = state
        self.reward_t = reward

        if done:
            self.states.append(self.state_t)
            self.actions.append(np.nan)
            self.rewards.append(self.reward_t)
            
            print("SAVING: {0:<5} frames (states, actions, rewards)".format(len(self.states)))
            fu_save(self.path, {"state":self.states,"action":self.actions,"reward":self.rewards}, overwrite=False, force=True)

            self.states.clear()
            self.actions.clear()
            self.rewards.clear()
            self.already_done = True
        
        return state, reward, done, info

    def reset(self, **kwargs):
        self.state_t = self.env.reset(**kwargs)
        self.reward_t = 0.
        self.already_done = False
        return self.state_t

class ResetEnvWrapper(gym.Wrapper): #TODO refactor
    
    def __init__(self, env_name, env_snapshot):
        super(ResetEnvWrapper, self).__init__(gym.make(env_name))
        self.snapshot = env_snapshot
        
    def step(self, action):
        return self.env.step(action)
        
    def reset(self, **kwargs):
        self.env.reset(**kwargs) #must be done for some reason
        self.env.unwrapped.restore_full_state(self.snapshot)
        return self.env.unwrapped._get_obs()
    
class OnehotUnwrapper(gym.Wrapper):
    
    def __init__(self, env):
        super(OnehotUnwrapper, self).__init__(env)
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(np.where(action==1)[0][0])
    

# TODO deprecate this....
class ObservationWrapper(gym.ObservationWrapper):
    
    mode = transform.mode

    def __init__(self, env, mode, **modeargs):
        super(ObservationWrapper, self).__init__(env)
        self.mode = mode(env, **modeargs)
        self.observation_space = self.mode.observation_space
        
    def observation(self, obs):
        return self.mode(obs)
    
class RewardWrapper(gym.Wrapper):
    
    def __init__(self, env, reward_fun):
        super(RewardWrapper, self).__init__(env)
        self.reward_fun = reward_fun
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward_fun(observation, action, reward), done, info
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class Float(gym.Wrapper):

    def __init__(self, env):
        super(Float, self).__init__(env)
        self.observation_space = gym.spaces.Box(np.float32(0), np.float32(1), shape=env.observation_space.shape, dtype=np.float32)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.astype(np.float32) / 255., *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.astype(np.float32) / 255.

class Integer(gym.Wrapper):

    def __init__(self, env):
        super(Integer, self).__init__(env)
        self.observation_space = gym.spaces.Box(np.uint8(0), np.uint8(255), shape=env.observation_space.shape, dtype=np.uint8)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return ((observation * 255).astype(np.uint8), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return (observation * 255).astype(np.uint8)

class CHW(gym.Wrapper):

    def __init__(self, env):
        super(CHW, self).__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        h,w,c = self.observation_space.shape
        assert c == 1 or c == 3 or c == 4 # invalid channels
        self.observation_space.shape = (c,h,w)
        self.observation_space.low = self.observation_space.low.reshape(self.observation_space.shape)
        self.observation_space.high = self.observation_space.high.reshape(self.observation_space.shape)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.transpose((2,0,1)), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.transpose((2,0,1))
        
class HWC(gym.Wrapper):

    def __init__(self, env):
        super(HWC, self).__init__(env)
        self.observation_space = copy.deepcopy(env.observation_space)
        c,h,w = self.observation_space.shape
        assert c == 1 or c == 3 or c == 4 # invalid channels
        self.observation_space.shape = (h,w,c)
        self.observation_space.low = self.observation_space.low.reshape(self.observation_space.shape)
        self.observation_space.high = self.observation_space.high.reshape(self.observation_space.shape)

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        return (observation.transpose((1,2,0)), *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        return observation.transpose((1,2,0))


class Crop(gym.Wrapper):
    pass        

class Atari(gym.Wrapper):

    def __init__(self, env):
        super(Atari, self).__init__(env)
        h,w,c = self.observation_space.shape
        assert c == 3 # invalid channels
        h -= (h//2)
        w -= (w//2)
        self.observation_space = gym.spaces.Box(np.float32(0), np.float32(1), shape=(1,h,w), dtype=np.float32)
       

    def fast_transform(self, observation):
        observation = observation.transpose((2,0,1)) #to CHW
        observation = observation[:,::2,::2] #fast downsample
        #components=(0.299, 0.587, 0.114) #to grayscale
        #observation = (observation[0, ...] * components[0] + 
        #               observation[1, ...] * components[1] + 
        #               observation[2, ...] * components[2])[np.newaxis, ...]
        observation = observation.astype(np.float32) / 255. #to float
        return observation

    def step(self, action, *args, **kwargs):
        observation, *rest = self.env.step(action)
        observation = self.fast_transform(observation)
        return (observation, *rest)

    def reset(self, *args, **kwargs):
        observation = self.env.reset()
        observation = self.fast_transform(observation)
        return observation


class Stack(gym.Wrapper):

    def __init__(self, env, n=3):
        super(Stack, self).__init__(env)
        shape = list(self.observation_space.shape)
        channel = np.isin(shape, [1,3,4])
        print(channel, np.sum(channel))
        if np.sum(channel) != 1:
            raise ValueError("Invalid channels in observation space: {0}".format(self.observation_space))
        self.__channel = np.argwhere(channel).item()
        shape[self.__channel] = n * shape[self.__channel]
        self.__buffer = deque(maxlen=n)
        self.observation_space = gym.spaces.Box(self.observation_space.low.flat[0], self.observation_space.high.flat[0], shape=shape, dtype=self.observation_space.dtype)

    def step(self, *args, **kwargs):
        state, *rest = super(Stack, self).step(*args, **kwargs)
        self.__buffer.append(state)
        return (np.concatenate(self.__buffer, axis=self.__channel), *rest)

    def reset(self, *args, **kwargs):
        state = super(Stack, self).reset(*args, **kwargs)
        for i in range(self.__buffer.maxlen):
            self.__buffer.append(state)
        return np.concatenate(self.__buffer, axis=self.__channel)

class ResetSkip(gym.Wrapper):

    def __init__(self, env, n=1):
        super(ResetSkip, self).__init__(env)
        self.n = n
    
    def step(self, *args, **kwargs):
        return super(ResetSkip, self).step(*args,**kwargs)
    
    def reset(self, *args, **kwargs):
        super(ResetSkip, self).reset(*args, **kwargs)
        for i in range(self.n):
            state, *rest = self.step(0)
        return state

import inspect
import sys
import re


__all__ = dict(inspect.getmembers(sys.modules[__name__], inspect.isclass))


def wrap(env, *wrappers):
    def _wrap(env, wrapper, **kwargs):
        if isinstance(wrapper, str):
            env = __all__[wrapper](env, **kwargs)
        elif isinstance(wrapper, gym.Wrapper):
            env = wrapper(env, **kwargs)
        else:
            raise ValueError("Invaliad wrapper: {0}".format(wrapper))
        return env

    for wrapper in wrappers:
        if isinstance(wrapper, (tuple, list)):
            env = _wrap(env, wrapper[0], **wrapper[1])
        else:
            env = _wrap(env, wrapper)
        
    return env


        
        