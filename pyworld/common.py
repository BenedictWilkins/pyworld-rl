#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 11:54:57 2019

@author: ben
"""
import argparse
import torch
from collections import namedtuple
import numpy as np

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
    


    