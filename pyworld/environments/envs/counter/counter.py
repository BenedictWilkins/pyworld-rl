#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:24:29 2019

@author: ben
"""
import gym

class Counter(gym.Env):
    
    def __init__(self):
        self.i = 0
        self.actions = [-1, 1]
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=0, high=10e99, shape=(1,), dtype=int)
        
    def step(self, action):
        self.i += 1 #self.actions[action]
        return np.array([self.i]), 0, self.i > 5, None
        
    def reset(self):
        self.i = 0
        return self.i
        
    def render(self):
        print("[{0}]".format(self.i))
        
        
        
if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.toolkit.tools.datautils as du
    import numpy as np
    
    env = Counter()
    policy = gu.policy.uniform_random_policy(env.action_space)
    dataset = gu.datasets(env, policy, size=10, mode=gu.mode.ss, epochs=3)
    
    for state, next_state in dataset:
        print(np.concatenate((state, next_state), 1))    