#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:54:57 2019

@author: ben
"""
import argparse
import torch
from collections import namedtuple
import gym
import numpy as np
import random
from abc import ABC, abstractmethod
import collections

def torch_cuda():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    print("cuda enabled = ", args.cuda)
    return torch.device("cuda" if args.cuda else "cpu")

    
Time = namedtuple('Time', ['episode', 'step', 'global_step', 'end'])

def isiterable(arg):
    try:
       iter(arg)
    except TypeError:
        return False
    else:
        return True



#TODO make more efficient - dont create a class every time this is called.
def batch(batch_labels):
     t = namedtuple('batch', batch_labels)
     t.__new__.__defaults__ = tuple([[] for _ in range(len(batch_labels))])
     return t
    

         
class Info:
    
    def __init__(self, summary_writer = None, info_interval=1000):
        self.info_interval = info_interval
        self.summary_writer = summary_writer
        self.info_labels = ['avg_reward/' + str(self.info_interval)]
        self.info = {k:0. for k in self.info_labels}
        self.print_info = []
        self.print_info.extend(self.info_labels) #things to print
       
    def __call__(self, obs):
        (state, action, reward, nstate, time) = obs
        #update tensorboard
        if time.global_step % self.info_interval == 0:
            self.update_summary(time.global_step)
            self.print_info(time)
        
    def update_summary(self, global_step):
        for k,v in self.info.items():
            self.summary_writer.add_scalar(k, v, global_step) #TODO deal with non scalars
                
    def print_info(self, time):
        print('INFO %d:' %(time.episode))
        for k in self.print_info:
            print('  %s:%s' %(k,self.info[k]))      

class Tracker:
    
    def __init__(self):
        self.value = None
    
    def update(self, obs):
        pass
    
    def get(self):
        pass
    
    def done(self):
        pass
    
class RewardTracker :
    
    def __init__(self, interval=10):
        self.episode_rewards = collections.deque(maxlen=interval)
        self.current_episode_reward = 0
        self.interval = 10
    
    def update(self, obs):
        _, _, reward, _, time = obs
        self.current_episode_reward += reward
                         
    def get(self):
        self.value = sum(self.episode_rewards) / len(self.episode_rewards)
        
    def done(self):
         self.episode_rewards.append(self.current_episode_reward)
         self.current_episode_reward = 0 
            
class UnrollSensor1:
    
    def __init__(self, callback, gamma=0.99, steps=2):
        self.callback = callback
        self.gamma =  gamma
        self.steps = steps
        self.total_reward = 0

    def __call__(self, obs):
        pstate, action, reward, state, time = obs
        if time.step % self.steps == 1:
            self.state_i = pstate
            self.action_i = action
            self.total_reward = 0

        self.total_reward *= self.gamma
        self.total_reward += reward
        if time.done or time.step % self.steps == 0:
            #print("callback: ", time)
            self.callback((self.state_i, self.action_i, self.total_reward, state, time))
            self.state_i = pstate
            self.action_i = action
            self.total_reward = 0
            
class UnrollSensor:
    
    def __init__(self, callback, gamma=0.99, steps=3):
        self.callback = callback
        self.gamma =  gamma
        self.steps = steps
        self.total_reward = 0
        self.unroll_states = [None] * steps
        self.unroll_actions = [None] * steps
        self.unroll_times = [None] * steps
        self.unroll_rewards = np.zeros(steps)


    def __call__(self, obs):
        pstate, action, reward, state, time = obs
        
        i = time.step % self.steps
        
        r_state = self.unroll_states[i]
        r_action = self.unroll_actions[i]
        r_time = self.unroll_times[i]
        r_reward = self.unroll_rewards[i]

        if r_state is not None:
            #print('i:', i, r_reward)
            self.callback((r_state, r_action, r_reward, pstate, r_time))
       
        #provide all of the unrolled rewards capped at the end time to the agent and reset
        if time.end:
           # print(self.unroll_rewards)
            for j in range(i+1, i + self.steps):
                k = j % self.steps
                if self.unroll_states[k] is not None:
                   # print(k, self.unroll_rewards[k])
                    self.callback((self.unroll_states[k], self.unroll_actions[k], self.unroll_rewards[k], pstate, self.unroll_times[k]))
            self.unroll_states = [None] * self.steps
            self.unroll_actions = [None] * self.steps
            self.unroll_times = [None] * self.steps
            self.unroll_rewards = np.zeros(self.steps)
        
        self.unroll_states[i] = pstate
        self.unroll_actions[i] = action
        self.unroll_times[i] = time
        self.unroll_rewards[i] = 0
        
        #compute the unrolled discounted reward
        gg = 1
        for j in range(self.steps):
            self.unroll_rewards[(i + self.steps - j) % self.steps] += (gg * reward)
            gg *= self.gamma
        
        
            
            
        




    
    
   
'''   
ag = RandomAgent()
sim = GymSimulator('CartPole-v0', ag)

for t in sim:
    print(t)
'''
    


    