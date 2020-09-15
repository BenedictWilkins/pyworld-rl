#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 17-06-2020 13:59:41
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import copy
import numpy as np
from ..pygame import PyGameEnvironment 

DEFAULT_SIZE = (64,64)

class Player:

    def __init__(self, center, radius, angle, w, h, speed=1):
        self.angle  = angle
        self.center = center
        self.radius = radius
        self.size = np.array([w,h])
        self.speed = speed

    @property
    def position(self):
        cx, cy = self.center
        r, a = self.radius, self.angle
        return np.array([cx + r * np.cos(a) - self.size[0]/2, cy + r * np.sin(a) - self.size[1]/2])


  
physics = [
    {"left":1, "right":-1},                           # simple
    {"left":1, "right":-1, "noop":0},                 # simple + no-op
]

DEG2RAD = np.pi/180

class Expander(PyGameEnvironment):

    def __init__(self, size=DEFAULT_SIZE, physics=physics[0], dtype=np.float32, format="CHW", speed=2):
        super(Expander, self).__init__(list(physics.keys()), display_size=size, background_colour=(0,0,0), format=format, dtype=dtype)
        self.physics = physics
        radius = min(size) / 3
        self.player1 = Player((size[0]/2, size[1]/2), radius, 0, size[0]/8, size[1]/8, speed=speed)
        self.__initial_state = copy.deepcopy(self.player1)

        self.expand = 0 #current size of the expansion

    """
    def cover(self):
        assert isinstance(self.unwrapped, Circular)
        self.reset()
        self.player1.speed = 1
        cover = [self.step(0)[0] for i in range(360)]
        cover = np.array(cover)
        self.reset()
        return cover, np.zeros(cover.shape[0])
    """

    def step(self, action):
        #update state
        self.player1.angle += DEG2RAD * self.player1.speed * self.physics[self.actions[action]]
        self.expand += 1
        # update graphics
        self.clear(self.background_colour) 
        
        w,h = self.display_size
        self.fill_rect((w/2 - self.expand/2, h/2-self.expand/2), (self.expand,self.expand), colour=(255,0,0))
        self.fill_rect(self.player1.position, self.player1.size, colour=(0,255,0))

        return self.get_image(), 0., self.expand > min(w,h), None

    def reset(self):
        self.player1 = copy.deepcopy(self.__initial_state)
        self.player1.angle = np.random.uniform(0, np.pi*2)

        self.clear(self.background_colour) # clear graphics buffer
        self.fill_rect(self.player1.position, self.player1.size, colour=(0,255,0))
        self.expand = 0

        return self.get_image()

def ExpanderNoop():
    return Expander(physics=physics[1], speed=5)