#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:39:58 2019

author: Benedict Wilkins
"""

from . import Environment

import numpy as np

class CoinCollector(Environment.PyGameEnvironment):
   
   physics = { 'up':           lambda env: env.set_position(env.position + env.speed * np.array([0,-1])),
               'up_right' :    lambda env: env.set_position(env.position + env.speed * np.array([1,-1])),
               'right':        lambda env: env.set_position(env.position + env.speed * np.array([1,0])),
               'down_right':   lambda env: env.set_position(env.position + env.speed * np.array([1,1])),
               'down':         lambda env: env.set_position(env.position + env.speed * np.array([0,1])),
               'down_left':    lambda env: env.set_position(env.position + env.speed * np.array([-1,1])),
               'left':         lambda env: env.set_position(env.position + env.speed * np.array([-1,0])),
               'up_left':      lambda env: env.set_position(env.position + env.speed * np.array([-1,-1])),
               'jump':         lambda env: env.set_position(np.array(env.observation_space.shape[:2])/2.), 
               'slow':         lambda env: env.set_speed(max(env.speed - 1, 1)),
               'fast':         lambda env: env.set_speed(min(env.speed + 1, 10))}
              
   
   actions = ['up', 'up_right', 'right', 'down_right', 'down', 'down_left', 'left', 'up_left', 'jump', 'slow', 'fast']
   actions_simple = actions[:-3]
   actions_no_speed = actions[:-2]
   actions_no_jump = actions[:-3] + actions[-2:]

   def __init__(self, size=128, background_colour=(0,0,0), speed=True, jump=True):
      actions = CoinCollector.actions
      if not speed and not jump:
         actions = CoinCollector.actions_no_speed_or_jump
      elif not speed:
         actions = CoinCollector.actions_no_speed
      elif not jump:
         actions = CoinCollector.actions_no_jump
      
      super(CoinCollector, self).__init__(actions, display_size=(size, size), background_colour=background_colour)
      self.speed = 1.
      self.position = np.array(self.observation_space.shape[:2]) / 2.
      
   def set_position(self, new_position):
      self.position = new_position
   
   def set_speed(self, new_speed):
      self.speed = new_speed
   
   def step(self, action):
      CoinCollector.physics[self.actions[action]](self) #physics will update stuff
      
      #update graphics
      self.display.fill(self.background_colour)
      self.fill_circle(self.position, 8, (255,255,255))
      #print(CoinCollector.actions[action], self.position, self.speed)
      image = self.get_image()
      #print(image.shape)
      done = np.any(self.position < 0.) or np.any(self.position > self.observation_space.shape[1]) 
      return image, 0, done, None
   
   def reset(self):
       self.speed = 1.
       self.position = np.array(self.observation_space.shape[:2]) / 2.
       self.display.fill(self.background_colour)
       self.fill_circle(self.position, 8, (255,255,255))

       return self.get_image()


if __name__ == "__main__":
   
   import pyworld.toolkit.tools.visutils as vu
   import pyworld.toolkit.tools.gymutils as gu
   
   speed = True
   jump = False
   env = CoinCollector(speed=speed, jump=jump)
   
   policy = gu.policy.uniform_random_policy(env.action_space)
   
   state = env.reset()
   
   for i in range(100):
      state, reward, done, _ = env.step(policy(state))
      #vu.show(state)
   
   #vu.close()
   
   