#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 14:26:11 2019

@author: ben
"""

import gym

import numpy as np
import cv2
import copy

import pyworld.toolkit.tools.visutils as vu

class Object:
    def __init__(self, img, pos=np.array([0.,0.], dtype=np.float32), speed=1.0):
        assert len(img.shape) == 3
        assert img.shape[0] == 1 or img.shape[0] == 3
        self.img = img
        self.pos = pos
        self.speed = speed
        
    def update(self, direction):
        self.pos += (direction * self.speed)


   


    
class CatMouse(gym.Env):
    
    
        
    def __init__(self, cat_img=None, mouse_img=None, cat_speed = 1.0, mouse_speed = 1.0, height = 84, width = 84):
        if cat_img == None:
            cat_img = vu.images.character("c")
        if mouse_img is None:
            mouse_img = vu.images.character("m")
            
        self.__empty_state = np.ones((1, height, width), dtype=np.float32) - 1.
        self.cat = Object(cat_img, self.__random_pos(cat_img, self.__empty_state), cat_speed)
        self.mouse = Object(mouse_img, self.__random_pos(mouse_img, self.__empty_state), mouse_speed)
        
        self.objects = {'cat':self.cat, 'mouse':self.mouse}
        self.__initial_objects  = copy.deepcopy(self.objects)
        
        self.state = np.copy(self.__empty_state)
        self.__update_state()

        self.base_vel = 2.
        
        self.observation_space = gym.spaces.Box(0.,1.,shape=(1, height, width))
        
        '''
        self.action_space = gym.spaces.Dict({'cat':gym.spaces.Discrete(8), 'mouse':gym.spaces.Discrete(8)})
        self.action_labels = ['north', 'north-east', 'east', 'south-east', 'south', 'south-west', 'west', 'north-west']
        self.v_map = [np.array([0.,-1.]), #north
                      np.array([1.,-1.]),  #north-east
                      np.array([1.,0.]),   #east
                      np.array([1.,1.]),   #south-east
                      np.array([0.,1.]),   #south
                      np.array([-1.,1.]),  #south-west
                      np.array([-1.,0.]),  #west
                      np.array([-1.,-1.])] #north-west
        '''
        
        self.action_space = gym.spaces.Box(low=0, high=2*np.pi, shape=(2,))
        
    def __update_state(self):
        self.state = np.copy(self.__empty_state)
        for obj in self.objects.values():
            obj.pos[0] = np.clip(obj.pos[0], 0, self.state.shape[2] - obj.img.shape[2])
            obj.pos[1] = np.clip(obj.pos[1], 0, self.state.shape[1] - obj.img.shape[1])
            x = int(obj.pos[0])
            y = int(obj.pos[1])
            self.state[:, y:y+obj.img.shape[1], x:x+obj.img.shape[2]] = obj.img
    
    def __random_pos(self, img, state):
        return np.array([np.random.randint(low=0, high=state.shape[1]-img.shape[1]),
                  np.random.randint(low=0, high=state.shape[2]-img.shape[2])], dtype=np.float64)
    
    def step(self, action):
        #assert self.action_space.contains(action)
        self.cat.update(np.array([np.cos(action[0]), np.sin(action[0])]))
        self.mouse.update(np.array([np.cos(action[1]), np.sin(action[1])]))
        self.__update_state()
        return self.state, 0, False, None
        
    def reset(self):
        self.cat = copy.deepcopy(self.__initial_objects['cat'])
        self.mouse = copy.deepcopy(self.__initial_objects['mouse'])
        self.objects = {'cat':self.cat, 'mouse':self.mouse}
        self.__update_state()
        return self.state
        
    
def chase_policy(env):
    
    def __chase_policy(state):
        #cheating!
        d = env.cat.pos - env.mouse.pos
        action = np.array([np.arctan2(d[1], d[0]), np.random.uniform(0,2*np.pi)])
        print(action)
        return action
    
    return __chase_policy

if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu
    env = CatMouse()
    #vu.play(gu.video(env, gu.uniform_random_policy(env)))
    vu.play(gu.video(env, chase_policy(env)))
    
    
    
    