#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:26:42 2019

@author: ben
"""
import gym

import numpy as np
import cv2
import copy


import pyworld.toolkit.tools.visutils as vu

class Object:
    
    def __init__(self, img, pos=np.array([0.,0.], dtype=np.float32), vel=np.array([0.,0.], dtype=np.float32)):
        assert len(img.shape) == 3
        assert img.shape[0] == 1 or img.shape[0] == 3
        self.img = img
        self.pos = pos
        self.vel = vel

mode_image = 0
mode_position = 1
mode_image_position = 2

class ObjectMover(gym.Env):
    
    '''
        ObjectMover is an environment in which a single object can be moved around in the cardinal directions.
    '''
    def __init__(self, shape, obj, base_vel=1.0, cinvert=False, mode=mode_image, none_action=False):
        super(ObjectMover, self).__init__()
        assert len(shape) == 3
        assert shape[0] == 1 or shape[0] == 3
        
        self.__empty_state = np.ones(shape, dtype=np.float32) - int(cinvert)
        self.__empty_state[:,1:-1,1:-1] = int(cinvert)
        
        self.state = np.copy(self.__empty_state)

        self.obj = obj
        self.__obj_init = copy.deepcopy(obj)
        self.__place()
        
        self.mode = [self.mode_image, self.mode_position, self.mode_image_position][mode]

            
        self.v_map = {0:np.array([0.,-base_vel]),1:np.array([0.,base_vel]),
                      2:np.array([-base_vel,0.]),3:np.array([base_vel,0.]),
                      4:np.array([0.,0.])}
        self.action_labels = {0:'NORTH', 1:'SOUTH', 2:'EAST', 3:'WEST', 4:'NONE'}

        self.action_space = gym.spaces.Discrete(4 + int(none_action))
        self.observation_space = gym.spaces.Box(low=np.float32(0.), high=np.float32(1.), shape=shape, dtype=np.float32)
        
    def get_action_meanings(self):
        return [self.action_labels[i] for i in range(len(self.action_labels))]
    
    def sample_step(self, position, policy):
        self.obj.pos = position
        self.__place()
        
        state = self.mode()
        action = policy(state)
        nstate, action, _, _ = self.step(action)
        return state, 0., nstate

    @staticmethod
    def cover(self, x=100, y=100):
        """ 
            Returns all posssible states of the game.
        """
        y = min(y, self.observation_space.shape[1] + 1)
        x = min(x, self.observation_space.shape[2] + 1)

        y_space = np.linspace(0,self.observation_space.shape[1], num=x)
        x_space = np.linspace(0,self.observation_space.shape[2], num=y)
        x, y = np.meshgrid(x_space, y_space)
        z = np.empty((x.size, *self.observation_space.shape), dtype=self.observation_space.dtype)

        for i, c in enumerate(zip(x.flatten(), y.flatten())):
            self.obj.pos = c
            self.__place()
            z[i] = self.state

        return z, None #TODO action cover?
        
    def step(self, action):
        self.obj.vel = self.v_map[action]
        self.obj.pos += self.obj.vel
        self.__place()
        return self.mode(), 0., self.__done(), None
    
    def __done(self):        
        x = int(self.obj.pos[0])
        y = int(self.obj.pos[1])
        
        minx = 0
        miny = 0
        maxx = self.state.shape[2] - self.obj.img.shape[2]
        maxy = self.state.shape[1] - self.obj.img.shape[1]
    
        
        return x == minx or y == miny or x == maxx or y == maxy
    
    def mode_image(self):
        return self.state
    
    def mode_position(self):
        return self.__real_position(self.obj.pos)
    
    def mode_image_position(self):
        real_position = self.__real_position(self.obj.pos)
        return self.state, real_position
    
    def __place(self):
        self.state = np.copy(self.__empty_state)
        x = int(self.obj.pos[0])
        y = int(self.obj.pos[1])
        x = np.clip(x, 0, self.state.shape[2] - self.obj.img.shape[2])
        y = np.clip(y, 0, self.state.shape[1] - self.obj.img.shape[1])
        #(C,H,W)
        self.state[:, y:y+self.obj.img.shape[1], x:x+self.obj.img.shape[2]] = self.obj.img
    
    def __real_position(self, position):
        return (position + np.array(self.obj.img.shape[1:3])/2.) / np.array(self.state.shape[1:3])
    
    def reset(self):
        self.obj = copy.deepcopy(self.__obj_init)
        self.__place()
        return self.mode()
    
    def render(self):
        cv2.imshow('BlockMove-v0', vu.channels_to_cv(self.state))
        if cv2.waitKey(60) == ord('q'):
            cv2.destroyAllWindows()
            return True
        return False

def a(shape=(1,64,64), mode=mode_image, none_action=False):
    from PIL import Image, ImageDraw, ImageFont
    import os
    #img = PIL.Image.open('imgs/a.png'))
    img = Image.new('RGB', (12,12), color = (0,0,0))
    d = ImageDraw.Draw(img)
    #print(os.path.dirname(__file__))
    font = ImageFont.truetype(os.path.dirname(__file__) + "/ArialCE.ttf", 18)
    d.text((1,-6), "a", font=font, fill=(255,255,255)) #dont ask...
    
    obj_image = vu.CHW(vu.gray(np.array(img))/ 255.)
    
    obj_pos = (np.array(shape) / 2 - np.array(obj_image.shape) / 2)[1:]

    return ObjectMover(shape, Object(obj_image, obj_pos), 2., mode=mode, none_action=none_action)    


def default(mode=mode_image, none_action=False):
    return ObjectMover((1,64,64), Object(np.ones((1,12,12)), np.array([26.,26.])), 2., mode=mode, none_action=none_action)

def stochastic1():
    class vmap:
        def __getitem__(self, k):
            basevel = 2
            r = (np.random.randint(0,2))*2 - 1 #1/-1
            return (np.array([0,r*basevel]), np.array([r*basevel, 0]))[k]
            
    env = default()
    env.action_labels = {0:'NORTH-SOUTH', 1:'EAST-WEST'}
    env.action_space = gym.spaces.Discrete(2)
    env.v_map = vmap()

    return env
        