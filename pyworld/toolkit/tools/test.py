#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:32:54 2019

@author: ben
"""

if __name__ == "__main__":
    import pyworld.environments.objectmover as om
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.gymutils as gu
    import gym
    
    env = gym.make('Pong-v0')
    env = gu.ObservationWrapper(env, gu.observation_mode.default)
    #env = ObservationWrapper(env, observation_mode.chw)
    print(env.observation_space)
    policy = gu.uniform_random_policy(env)
    
    #for d in gu.dataset(env, policy, size=10):
    #    vu.play(d[0])
    #    print("next")
        
    import numpy as np
    
    l22 = lambda x, y: np.dot((x-y), (x-y))

    d = l22(np.arange(1,4), np.arange(0,3))
    print(d)
