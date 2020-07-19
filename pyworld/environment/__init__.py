#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 03-09-2019 12:05:53 

    Collection of environments defined in pyworld with the OpenAI Gym API.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from gym.envs.registration import register as gym_register
from gym.envs.registration import registry as gym_registry
import copy

envs = {}

def environments():
    return copy.copy(envs.keys())

def unregister():
   for env in envs:
       if env in gym_registry.env_specs:
            del gym_registry.env_specs[env]

def reregister():
    for env in envs:
        del gym_registry.env_specs[env]
        gym_register(id=env, entry_point=envs[env][0], **envs[env][1])

def register(id, entry_point, **kwargs):
    if not id in gym_registry.env_specs:
        try:
            gym_register(id=id, entry_point=entry_point, **kwargs)
        except Exception as e:
            print("warning: failed to register environment\n" + str(e))
    envs[id] = (entry_point, kwargs)



register(id="Circular-v0", entry_point='pyworld.environment.circular:Circular')

register(id="Paddles-v0", entry_point='pyworld.environment.paddles:Paddles')
register(id="Paddles-v1", entry_point='pyworld.environment.paddles:PaddlesShared')
register(id="Paddles-v2", entry_point='pyworld.environment.paddles:PaddlesAlternative')

register(id="Paddles-noop-v0", entry_point='pyworld.environment.paddles:PaddlesNoop')

register(id='ObjectMover-v0', 
    entry_point='pyworld.environment.objectmover:default')

register(id='ObjectMover-v1',
    entry_point='pyworld.environment.objectmover:a', kwargs = {'shape':(1,64,64)})

register(id="ObjectMover-v2", entry_point='pyworld.environment.objectmover:stochastic1')

register(id='CoinCollector-Easy-v0', 
    entry_point='pyworld.environment.pygame.CoinCollector:CoinCollector', kwargs = {'jump':False, 'speed':False})

register(id='CoinCollector-NoJump-v0', 
    entry_point='pyworld.environment.pygame.CoinCollector:CoinCollector', kwargs = {'jump':False, 'speed':True})

register(id='CoinCollector-NoSpeed-v0', 
    entry_point='pyworld.environment.pygame.CoinCollector:CoinCollector', kwargs = {'jump':True, 'speed':False})

register(id='CoinCollector-Hard-v0', 
    entry_point='pyworld.environments.pygame.CoinCollector:CoinCollector', kwargs = {'jump':True, 'speed':True})




'''
from gym import envs
print(envs.registry.all())

gym.make('ObjectMover-v0')
'''