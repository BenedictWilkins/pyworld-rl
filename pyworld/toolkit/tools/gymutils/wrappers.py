#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:04:55 2019

author: Benedict Wilkins
"""
import gym
import numpy as np

from . import transformation

from ..fileutils import save as fu_save

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
    
class ObservationWrapper(gym.ObservationWrapper):
    
    mode = transformation.mode

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
        
        
        
        
        
        
        