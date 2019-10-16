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
        return self.i, 0, False, None
        
    def reset(self):
        self.i = 0
        return self.i
        
    def render(self):
        print("[{0}]".format(self.i))
        
        
        
if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.toolkit.tools.datautils as du
    env = Counter()
    policy = gu.uniform_random_policy(env)
    data_generator = gu.dynamic_dataset(env, policy, size=10, chunk=5, mode=gu.s)

    for i, batch in du.batch_iterator(*next(data_generator), circular=True):
        print(i, batch)
        next(data_generator)
        if i > 100:
            break
        
    
    