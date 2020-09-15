#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 12:06:13 2019

author: Benedict Wilkins
"""

import os
os.putenv('SDL_VIDEODRIVER', 'fbcon')
os.environ["SDL_VIDEODRIVER"] = 'dummy'

import numbers

import pygame
import pygame.gfxdraw

import gym
import numpy as np

from functools import wraps

class StepMeta(type): #TODO is this really needed... just use ABC?
   
    def __new__(mcls, name, bases, local):
      try:
          local['step']
      except:
          raise AttributeError('Attribute \'step\' must be defined by class:' + mcls)
      try:
         local['reset']
      except:
          raise AttributeError('Attribute \'reset\' must be defined by class:' + mcls)
   
      return super(StepMeta, mcls).__new__(mcls, name, bases, local)

def format_image(image, fold, fnew):
    old = np.array([fold.index(i) for i in "CHW"])
    new = np.array([fnew.index(i) for i in "CHW"])
    swap = new[old]
    return image.transpose(*swap)

class PyGameEnvironment(gym.Env, metaclass = StepMeta):

    def __init__(self, actions, display_size=(32,32), background_colour=(0,0,0), format="WHC", dtype=np.uint8):
        super(PyGameEnvironment, self).__init__()
        pygame.init() #??

        self.background_colour = background_colour
        self.display = pygame.display.set_mode(display_size) 
        self.action_space = gym.spaces.Discrete(len(actions))
        self.actions = actions

        # initial space: uint8 WHC, will be transformed below
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=((*display_size, 3)), dtype=np.uint8)

        # state format (default WHC uint8)
        if np.issubdtype(dtype, np.floating):
            format_dtype = lambda x: x.astype(dtype) / 255. #creates a copy?
            high = 1.
        elif np.issubdtype(dtype, np.integer):
            format_dtype = lambda x: x
            high = 255
        else:
            raise TypeError("Invalid dtype: {0}".format(dtype))
        
        format_shape = lambda x: format_image(x, "WHC", format)

        self.__format_image = lambda x: format_dtype(format_shape(x))
        
        self.observation_space.high = self.__format_image(self.observation_space.high)
        self.observation_space.low =  self.__format_image(self.observation_space.low)
        self.observation_space.dtype = dtype
        self.observation_space.shape = self.observation_space.low.shape

        self.display_size = display_size # W, H

    @property
    def action_meanings(self):
        return self.actions
     
    def get_image_raw(self):
        '''
            Get the current state as an image in WHC 0-255 format (default 3 channels). Creates a direct copy of the pygame display pixel buffer.
        '''
        pygame.display.update()
        return pygame.surfarray.array3d(self.display)

    def get_image(self):
        '''
            Get the current state as an image in the format specified on creation of the environments. Creates a copy of the pygame display pixel buffer.
        '''
        pygame.display.update()
        buffer = pygame.surfarray.pixels3d(self.display)
        return self.__format_image(buffer)

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
        raise NotImplementedError()

    def reset(self, **kwargs):
        raise NotImplementedError()
   
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
    env = TestGame(['attack'], display_size=(4,5), background_colour=(255,255,255), format="CHW", dtype=np.uint8)
    print(env.observation_space.shape, env.observation_space.dtype)
    
    env.reset()
    #env.uses_metaclass()
    env.display.fill(env.background_colour)
    state, *_ = env.step(None)
    print(state.shape, state.dtype)
    print(state)

    state, *_ = env.step(None)
    #print(state)




        




      


