#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:38:16 2019

@author: ben
"""

import pyworld.toolkit.tools.visutils as vu
import numpy as np

def test_video():
    import gymutils as gu
    import gym
    
    env = gym.make('SpaceInvaders-v0')
    policy = gu.uniform_random_policy(env)
    
    def video(env, policy):
        for s in gu.s_iterator(env, policy):
            yield s.state
            
    vu.play(video(env, policy))
    
def test_plot2D():
    vu.plot2D(lambda x: x, np.random.uniform(size=(1000,2)), np.random.randint(0,10,size=(1000,)))

test_plot2D()