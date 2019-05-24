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
import time as datetime
import copy

import os as os


def save_net(net, path):
    pathn, filename = os.path.split(path)
    if not os.path.exists(pathn):
        os.makedirs(pathn)
    if len(filename.split('.')) == 1:
        filename += '.pt'
    torch.save(net.state_dict(), path)

def torch_cuda():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    args = parser.parse_args()
    print("cuda enabled = ", args.cuda)
    return torch.device("cuda" if args.cuda else "cpu")
    
Time = namedtuple('Time', ['episode', 'step', 'global_step', 'end'])

def hook(*hooks):
    return {k:None for k in hooks}

def info(lcls, hooks=None):
    for k,v in hooks.items():
        hooks[k] = lcls[k]

def isiterable(arg):
    try:
       iter(arg)
    except TypeError:
        return False
    else:
        return True

def device():  
    if(torch.cuda.is_available()): 
        return'cuda'
    else:
        return 'cpu'
    
GRADIENT_LABELS = ['grad/grad_max', 'grad/grad_l2']
QVAL_LABELS = ['qval']

#general debugging
def gradient_info(net, info):
        #gradient information
        grad_max = 0.
        grad_means = 0.
        grad_count = 0
        for p in net.parameters():
            grad_max = max(grad_max, p.grad.abs().max().item())
            grad_means += (p.grad ** 2).mean().sqrt().item()
            grad_count += 1
        
        info.info_trackers['grad/grad_max'].step(grad_max)
        info.info_trackers['grad/grad_l2'].step(grad_means / grad_count)
        
def qval_info(states, net, info):
    qvals = net(states).detach().cpu().numpy()
    info.info_trackers['qval'].step(np.mean(qvals.max(1)))


#TODO make more efficient - dont create a class every time this is called.
def batch(batch_labels):
     t = namedtuple('batch', batch_labels)
     t.__new__.__defaults__ = tuple([[] for _ in range(len(batch_labels))])
     return t
    

class Tracker:
    
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    @abstractmethod
    def step(self, obs):
        pass
    
    @abstractmethod
    def get(self):
        pass        
    
    def episode(self, _):
        pass #TODO remove in favour of step at interval / episode
    
    def summarise(self, summary_writer, label, step):
        summary_writer.add_scalar(label, self.get(), step)

class EpisodeAverageTracker(Tracker):
    
    def __init__(self, size, enabled=False):
        super(EpisodeAverageTracker, self).__init__(enabled)
        self.values = collections.deque(maxlen=size)
        self.current = 0
        self.step = self.__step_first
    
    def __step_first(self, value):
        self.current += value
        self.step = self.__step_rest
        self.enabled = True
        
    def __step_rest(self, value):
        self.current += value
   
    def episode(self, _):
        self.values.append(self.current)
        self.current = 0
    
    def get(self):
        return sum(self.values) / len(self.values)

class StepTracker(Tracker):
    
    def __init__(self, enabled=False):
        super(StepTracker, self).__init__(enabled)
        self.value = None
        self.step = self.__step_first
     
    def __step_first(self, value):
        self.value = value
        self.step = self.__step_rest
        self.enabled = True
        
    def __step_rest(self, value):
        self.value = value
        
    def episode(self, _):
        pass
    
    def get(self):
        return self.value
    
class StepAverageTracker(Tracker):
    
    def __init__(self, size, enabled=False):
        super(StepAverageTracker, self).__init__(enabled)
        self.values = collections.deque(maxlen=size)
        self.step = self.__step_first
        
    def episode(self, _):
        pass #TODO remove 
    
    def __step_first(self, value):
        self.values.append(value)
        self.step = self.__step_rest
        self.enabled = True
        
    def __step_rest(self, value):
        self.values.append(value)
    
    def get(self):
        return sum(self.values) / len(self.values)
    
    
class FrameTracker(Tracker):
    
    def __init__(self, enabled=True):
        super(FrameTracker, self).__init__(enabled)
        self.real_time = datetime.time()
        self.fps = 0.
        
    def step(self, _):
        pass
    
    def episode(self, time):
        n_real_time = datetime.time()
        self.fps = time.step / ((n_real_time - self.real_time)) #time is given in seconds?????
        self.real_time = n_real_time
        
    def get(self):
        return self.fps
        
class Info:
    
    def __init__(self, summary_writer = None):
        self.summary_writer = summary_writer
        self.info_trackers = {}
        self.print_trackers = []
        
    def add_tracker(self, label, tracker, pprint=False, enabled=True):
        self.info_trackers[label] = tracker
        if pprint:
            self.print_trackers.append(label)
     
    def episode(self, time):
        for v in self.info_trackers.values():
            v.episode(time)
        
    def print_info(self, time):
        print('INFO %s:' %(time))
        for k in self.print_trackers:
            print('  %s:%s' %(k,self.info_trackers[k].get()))  
            
    def summarise(self, time, trackers=None):
        if not trackers:
            for k,v in self.info_trackers.items():
                v.summarise(self.summary_writer, k, time.global_step)
        else:
            for k in trackers:
                if self.info_trackers[k].enabled:
                    self.info_trackers[k].summarise(self.summary_writer, k, time.global_step)
                
def batch_to_numpy(batch, types, copy=False):
    return [np.array(batch[i], copy=copy, dtype=types[i]) for i in range(len(batch))]

def batch_to_tensor(batch, types, device='cpu'):
        return [types[i](batch[i]).to(device) for i in range(len(batch))]
        '''    
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
        
ag = RandomAgent()
sim = GymSimulator('CartPole-v0', ag)

for t in sim:
    print(t)
'''
    


    