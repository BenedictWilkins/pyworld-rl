#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:06:13 2019

author: Benedict Wilkins
"""

# set SDL to use the dummy NULL video driver, so it doesn't need a windowing system.
import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = 'dummy'
# NOTE: for some reason this doesnt work in jupyter, it has to be place at the top of a notebook to activate headless mode

#os.environ["SDL_VIDEO_X11_WMCLASS"] = '' #ipykernel_launcher.py
#os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"



import pygame
import pygame.gfxdraw

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

class PyGameEnvironment(gym.Env, metaclass = StepMeta):

    def __init__(self, actions, display_size=(128,128), background_colour=(255,255,255)):
        super(PyGameEnvironment, self).__init__()
        pygame.init() #??
        self.display = pygame.display.set_mode(display_size) 
        #pygame.display.init()


        self.background_colour = background_colour

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((display_size[1], display_size[0], 3)), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(len(actions))
        self.actions = actions
    
    @property
    def action_meanings(self):
        return self.actions
     
    def get_image_raw(self):
        '''
            Get the current state as an image in HWC 0-255 format (default 3 channels).
        '''
        pygame.display.update()
        return pygame.surfarray.array3d(self.display).swapaxes(0,1) #WHC to HWC format

    def get_image(self):
        '''
            Get the current state as an image.
        '''
        pygame.display.update()
        return np.copy(pygame.surfarray.array3d(self.display).swapaxes(0,1)) #TODO is copy needed?
   
    def clear(self, colour=(0,0,0)):
        self.display.fill(colour)

    def fill_circle(self, position, radius, colour=(255,255,255)):
        pygame.gfxdraw.filled_circle(self.display, int(position[0]), int(position[1]), int(radius), colour)
        pygame.gfxdraw.aacircle(self.display, int(position[0]), int(position[1]), int(radius), colour)
      
    def draw_circle(self, position, radius, colour=(255,255,255)):
         pygame.gfxdraw.aacircle(self.display, int(position[0]), int(position[1]), int(radius), colour)
   
    def draw_rect(self, position, size, colour=(255,255,255)):
        pygame.gfxdraw.rectangle(self.display, (*position, *size), colour)

    def fill_rect(self, position, size, colour=(255,255,255)):
        pygame.gfxdraw.box(self.display, (*position, *size), colour)

    def step(self, action, **kwargs):
        pass

    def reset(self, **kwargs):
        pass
   
    def render(self, **kwargs):
        os.environ["SDL_VIDEODRIVER"] = OLD_VDRIVER

    def quit(self):
        pygame.quit()

class TestGame(PyGameEnvironment):
   
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

   import time
   time.sleep(10)
      
      


