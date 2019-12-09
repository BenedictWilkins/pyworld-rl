#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:05:53 2019

@author: ben
"""
from gym.envs.registration import register

'''
import gym
env_names = ['ObjectMover-v0', 'ObjectMover-v1']    
for env_name in env_names:
    if env_name in gym.envs.registry.env_specs:
         del gym.envs.registry.env_specs[env_name]
'''


register(id='ObjectMover-v0', 
                 entry_point='pyworld.environments.envs.objectmover:default')

register(id='ObjectMover-v1',
             entry_point='pyworld.environments.envs.objectmover:a', kwargs = {'shape':(1,64,64)})

'''
from gym import envs
print(envs.registry.all())

gym.make('ObjectMover-v0')
'''