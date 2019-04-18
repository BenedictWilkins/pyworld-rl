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

def torch_cuda():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    print("cuda enabled = ", args.cuda)
    return torch.device("cuda" if args.cuda else "cpu")

    
Time = namedtuple('Time', ['episode', 'step', 'global_step', 'end'])



def is_iterable(arg):
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
    
    def __init__(self, summary_writer = None, info_interval=10):
        self.info_interval = info_interval
        self.summary_writer = summary_writer
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.info_labels = ['avg_reward/' + str(self.info_interval)]
        self.info = {k:0. for k in self.info_labels}
       
    def __call__(self, agent, obs):
        (state, action, reward, nstate, time) = obs
        #update tensorboard
        if agent.update_summary and self.summary_writer:
            self.update_summary(agent, time.global_step)

        self.current_episode_reward += reward
        if time.done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0 
            if time.episode % self.info_interval == 0:
                rs = self.episode_rewards[-self.info_interval:]
                avg_rwds = sum(rs) / len(rs) 
                self.info = {self.info_labels[0] : avg_rwds}
                self.info.update(agent.info)
                self.print_info(time, self.info)
        
    def update_summary(self, agent, global_step):
        for k,v in agent.summary_info.items():
            self.summary_writer.add_scalar(k, v, global_step) #TODO deal with non scalars
        agent.update_summary = False
                
    def print_info(self, time, info):
        print('INFO %d:' %(time.episode))
        for k,v in info.items():
            self.summary_writer.add_scalar(k, v, time.global_step) #TODO deal with non scalars
            print('  %s:%s' %(k,v))            
            
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
    


    