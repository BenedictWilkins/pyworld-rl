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
    def __init__(self, shape, obj, base_vel=1.0, cinvert=False, mode=mode_image):
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

            
        self.v_map = {0:np.array([0.,-base_vel]),1:np.array([0.,base_vel]),2:np.array([-base_vel,0.]),3:np.array([base_vel,0.])}
        self.action_labels = {0:'NORTH', 1:'SOUTH', 2:'EAST', 3:'WEST'}

        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=shape, dtype=np.float32)
        
    def sample_step(self, position, policy):
        self.obj.pos = position
        self.__place()
        
        state = self.mode()
        action = policy(state)
        nstate, action, _, _ = self.step(action)
        return state, None, nstate
        
    def step(self, action):
        self.obj.vel = self.v_map[action]
        self.obj.pos += self.obj.vel
        self.__place()
        return self.mode(), 0, False, None
    
    def __done(self):
        raise NotImplementedError()
    
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
        x = np.clip(x, 0, self.state.shape[2] - self.obj.img.shape[1])
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

def a(environment_shape=(1,64,64), mode=mode_image):
    from PIL import Image, ImageDraw, ImageFont
    #img = PIL.Image.open('imgs/a.png'))
    img = Image.new('RGB', (16, 16), color = (0,0,0))
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype("'/Library/Fonts/Arial.ttf'", 15)
    d.text((0,0), "a", font=font,fill=(255,255,255))
    
    obj_image = np.array(img) / 255.
    cv2.imshow('t', obj_image)
    
    obj_pos = (np.array(environment_shape) / 2 + np.array(obj_image.shape) / 2)[0:2]
    return ObjectMover(environment_shape, Object(obj_image, obj_pos), 4.,mode=mode)    


def default(mode=mode_image):
    return ObjectMover((1,64,64), Object(np.ones((1,12,12)), np.array([26.,26.])), 2., mode=mode)
        
if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu
    #(C,H,W)
    env = default() #a((1,32,64))
    policy = gu.uniform_random_policy(env)

    for s,a in gu.sa_iterator(env, policy):
        print(env.action_labels[a])
        if env.render():
            break

        