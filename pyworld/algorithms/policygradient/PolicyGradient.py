#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 16:18:51 2019

@author: ben
"""

from ... import agent as pwag
from ... import model as pwmo
from ...toolkit import diagnostics as diag
from ...toolkit.tools import datautils

import torch 
import torch.nn as nn
import torch.nn.functional as nnf
import numpy as np


class PGAgent(pwag.Agent):
   
    BATCH_LABELS = ['action', 'reward', 'state']
    
    def __init__(self, model, sensor, actuator, gamma=0.99):
        super(PGAgent, self).__init__()
        self.model = model
        self.sensor = sensor
        self.sensor.register(self)
        self.actuator = actuator
        
        self.obs_state = None
        #episodic unroll for reinforce algorithm, batch_size = 1 episode
        self.experience = pwag.UnrollExperience(batch_update, batch_labels = PGAgent.BATCH_LABELS, gamma=gamma, batch_size=0)
        
        #info
        self.episode_reward = diag.Variable()
        
    def sense(self, obs):
        
        self.obs_state = obs.state
        self.experience.append(obs)
        if self.experience.full():
            batch = self.experience.popall()
            self.model.step(batch)
            
        #info
        datautils
    
    def reset(self, obs):
        self.obs_state = obs.state
        
    def attempt(self):
        p_actions = self.model(self.obs_state)
        #print(p_actions)
        self.actuator(p_actions)
        
def batch_update(batch, obs):
    batch.action.append(obs.action)
    batch.reward.append(obs.reward)
    batch.state.append(obs.state)
    

class PolicyGradientNetwork(nn.Module):
    
    def __init__(self, input_size, actions_size):
        super(PolicyGradientNetwork, self).__init__()
        
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, actions_size))
        
    def forward(self, x):
        return self.net(x)
            
class Reinforce(pwmo.NNModel):
    
    def __init__(self, net, *params):
        super(Reinforce, self).__init__(net, *params)
        # trackable values
        self.loss = diag.Variable(None)
        self.q_mean = diag.Variable(None)
        self.q_std = diag.Variable(None)
        
    def __call__(self, state):
        return self.net(torch.FloatTensor(state).to(self.device)).detach().numpy()
        
    def step(self, batch):
        '''
            Performs one gradient step using the current episode data.
        '''
        self.optim.zero_grad()
        states_v = torch.FloatTensor(batch.state).to(self.device)
        actions_v = torch.LongTensor(batch.action).to(self.device)
        #q-values should have already been computed
        qs_v = torch.FloatTensor(batch.reward).to(self.device)

        logits_v = self.net(states_v)
        log_prob_v = nnf.log_softmax(logits_v, dim=1)
        log_prob_actions_v = qs_v * log_prob_v[range(states_v.shape[0]), actions_v]
        loss =  - log_prob_actions_v.mean()
        
        loss.backward()
 
        self.optim.step()
        
        #track values
        self.loss.value(loss.item())

        self.q_mean.value(qs_v.mean())
        self.q_std.value(qs_v.std())
        
        return loss.item()

        
        
  