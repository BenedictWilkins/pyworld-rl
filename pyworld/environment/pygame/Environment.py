#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:06:13 2019

author: Benedict Wilkins
"""
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame
import pygame.gfxdraw

# set SDL to use the dummy NULL video driver, 
#   so it doesn't need a windowing system.
OLD_VDRIVER = os.environ.get("SDL_VIDEODRIVER", None)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import numpy as np

from functools import wraps

def update_decorator(func):
   
    @wraps(func)
    def decorator(self, *args, **kwargs):
      ret = func(self, *args, **kwargs)
     # pygame.display.update()
      return ret
            
    return decorator


class StepMeta(type):
   
    def __new__(mcls, name, bases, local):
      try:
          local['step'] = update_decorator(local['step'])
      except:
          raise AttributeError('Attribute \'step\' must be defined by class:' + mcls)
      try:
         local['reset'] = update_decorator(local['reset']) 
      except:
          raise AttributeError('Attribute \'reset\' must be defined by class:' + mcls)
   
      return super(StepMeta, mcls).__new__(mcls, name, bases, local)

class GameEnvironment(gym.Env, metaclass = StepMeta):

    def __init__(self, actions, display_size=(128,128), background_colour=(255,255,255)):
        super(GameEnvironment, self).__init__()
        self.display = pygame.display.set_mode(display_size) 
        
        self.background_colour = background_colour

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((display_size[0], display_size[1], 3)), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(actions))
        self.actions = actions
    
    def get_action_meanings(self):
        return self.actions
     
    def get_image_raw(self):
        '''
            Get the current state as an image in WHC 0-255 format (default 3 channels).
        '''
        pygame.display.update()
        return pygame.surfarray.array3d(self.display)

    def get_image(self):
        '''
            Get the current state as an image.
        '''
        pygame.display.update()
        return np.copy(pygame.surfarray.array3d(self.display))
   
    def fill_circle(self, position, radius, colour):
        pygame.gfxdraw.filled_circle(self.display, int(position[0]), int(position[1]), int(radius), colour)
        pygame.gfxdraw.aacircle(self.display, int(position[0]), int(position[1]), int(radius), colour)
      
    def draw_circle(self, position, radius, colour):
         pygame.gfxdraw.aacircle(self.display, int(position[0]), int(position[1]), int(radius), colour)
   
    def step(self, action, **kwargs):
        pass

    def reset(self, **kwargs):
        pass
   
    def render(self, **kwargs):
        os.environ["SDL_VIDEODRIVER"] = OLD_VDRIVER

class TestGame(GameEnvironment):
   
   def __init__(self, actions, **kwargs):
      super(TestGame, self).__init__(actions, **kwargs)
      
   def step(self, action):
      image = self.get_image()
      return image, 0.0, False, None
      
   def reset(self):
      return self.get_image()


if __name__ == "__main__":
   env = TestGame(['attack'])
   env.reset()
   #env.uses_metaclass()
   env.display.fill(env.background_colour)
   print(env.step(None))
   
   print(env.reset())
   env.step(None)
      
      


