#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:55 2019

author: Benedict Wilkins
"""
import gym
import numpy as np

from . import transformation

class ResetEnvWrapper(gym.Wrapper):
    
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
    
class ObservationWrapper(gym.ObservationWrapper):
    
    mode = transformation.mode

    def __init__(self, env, mode, **modeargs):
        super(ObservationWrapper, self).__init__(env)
        self.mode = mode(env, **modeargs)
        self.observation_space = self.mode.observation_space
        
    def observation(self, obs):
        return self.mode(obs)