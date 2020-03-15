#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 12:05:53 2019

@author: ben
"""
from gym.envs.registration import register as gym_register
from gym.envs.registration import registry as gym_registry
import copy

env_names = ['ObjectMover-v0', 'ObjectMover-v1', 
             'CoinCollector-NoJump-v0', 'CoinCollector-Easy-v0', 
             'CoinCollector-NoSpeed-v0', 'CoinCollector-Hard-v0']

def environments():
    return copy.copy(env_names)

def unregister():
   
   for env_name in env_names:
       if env_name in gym_registry.env_specs:
            del gym_registry.env_specs[env_name]

def register(id, entry_point, **kwargs):
    if not id in gym_registry.env_specs:
        try:
            gym_register(id=id, entry_point=entry_point, **kwargs)
        except Exception as e:
            print("warning: failed to register environment\n" + str(e))
            
register(id='ObjectMover-v0', 
    entry_point='pyworld.environments.envs.objectmover:default')

register(id='ObjectMover-v1',
    entry_point='pyworld.environments.envs.objectmover:a', kwargs = {'shape':(1,64,64)})

register(id='CoinCollector-Easy-v0', 
    entry_point='pyworld.environments.envs.pygame.CoinCollector:CoinCollector', kwargs = {'jump':False, 'speed':False})

register(id='CoinCollector-NoJump-v0', 
    entry_point='pyworld.environments.envs.pygame.CoinCollector:CoinCollector', kwargs = {'jump':False, 'speed':True})

register(id='CoinCollector-NoSpeed-v0', 
    entry_point='pyworld.environments.envs.pygame.CoinCollector:CoinCollector', kwargs = {'jump':True, 'speed':False})

register(id='CoinCollector-Hard-v0', 
    entry_point='pyworld.environments.envs.pygame.CoinCollector:CoinCollector', kwargs = {'jump':True, 'speed':True})




'''
from gym import envs
print(envs.registry.all())

gym.make('ObjectMover-v0')
'''