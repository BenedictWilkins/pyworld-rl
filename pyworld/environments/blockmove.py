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

class Object:
    
    def __init__(self, img, pos=np.array([0.,0.], dtype=np.float32), vel=np.array([0.,0.], dtype=np.float32)):
        self.img = img
        self.pos = pos
        self.vel = vel

class BlockMove(gym.Env):
    
    def __init__(self, shape, obj, base_vel=1.0, bug_prob=0.):
        super(BlockMove, self).__init__()
        assert len(shape) == 2
        self.state = np.zeros(shape, dtype=np.float32)
        self.obj = obj
        self.obj_init = copy.deepcopy(obj)
        self.__place()
        
        self.bug_prob = bug_prob
        
        self.v_map = {0:np.array([0.,-base_vel]),1:np.array([0.,base_vel]),2:np.array([-base_vel,0.]),3:np.array([base_vel,0.])}
        self.action_labels = {0:'NORTH', 1:'SOUTH', 2:'EAST', 3:'WEST'}

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(*shape, 1), dtype=np.float32)
        
        
    def step(self, action):
        self.obj.vel = self.v_map[action]
        if np.random.uniform() > self.bug_prob:
            self.obj.pos += self.obj.vel
            self.__place()
           
        #else dont update!
        return self.state, 0., False, None
    
    def __place(self):
        self.state = np.zeros(self.state.shape, dtype=np.float32)
        x = int(self.obj.pos[0])
        y = int(self.obj.pos[1])
        x = np.clip(x, 0, self.state.shape[0] - self.obj.img.shape[0])
        y = np.clip(y, 0, self.state.shape[1] - self.obj.img.shape[0])
        self.state[x:x+self.obj.img.shape[0], y:y+self.obj.img.shape[1]] = self.obj.img
    
    def reset(self):
        self.obj = copy.deepcopy(self.obj)
        self.__place()
        return self.state
    
    def render(self):
        cv2.imshow('BlockMove-v0', self.state)
        return cv2.waitKey(60) == ord('q')

def default():
    return BlockMove((64,64), Object(np.ones((12,12)), np.array([26.,26.])), 2.)
        
if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu

    env = BlockMove((64,64), Object(np.ones((12,12)), np.array([26.,26.])), 2.)
    policy = gu.uniform_random_policy(env)

    for s,a in gu.sa_iterator(env, policy):
        env.render()

        