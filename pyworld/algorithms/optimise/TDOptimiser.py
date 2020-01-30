#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 12:53:45 2019

author: Benedict Wilkins
"""

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

from pyworld.toolkit.tools import gymutils as gu
from pyworld.toolkit.tools import datautils as du
from pyworld.toolkit.tools import torchutils as tu   

if __name__ != "__main__":
    from .Optimise import Optimiser
else:
    from pyworld.algorithms.optimise.Optimise import Optimiser
    
class TDO(Optimiser):
    
    def __init__(self, critic, logits=True, gamma=0.99, lr=0.002):
        super(TDO, self).__init__(critic)
        
        
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma

        self.cma = du.CMA('loss')
        
        if logits: 
            self.__categorical = lambda p: Categorical(logits = p)
        else:
            self.__categorical = lambda p: Categorical(probs = p)
        
    def step(self, episode):
        states, actions, rewards = episode
        rewards = rewards.astype(np.float32)
        
        total_reward = rewards.sum()
    
        returns = gu.returns(rewards)
        #states = gu.transformation.stack(states, frames=3, step=1)
        
        states, actions, returns = du.shuffle(states, actions, returns) #shuffle the data... sigh...
        
        states = torch.as_tensor(states, device=self.model.device).detach()  #torch.FloatTensor(states).detach()
        actions = torch.as_tensor(actions, device=self.model.device).detach() #torch.FloatTensor(actions).detach()
        returns = torch.as_tensor(returns, device=self.model.device).detach() #torch.FloatTensor(returns).detach() 
        
        #returns = (returns - returns.mean()) / (returns.std() + 1e-5) #is this needed?

        #optimise the value function for a bit... (so that it can catch up with the policy current)
 
        for i in range(40):
            values = self.model(states).squeeze()
            value_loss = 0.5 * F.mse_loss(values, returns).sum()
            self.optimiser.zero_grad()
            value_loss.backward()
            self.optimiser.step()
                
            #for b_states, b_returns in du.batch_iterator(states, returns, count=False, shuffle=False, batch_size=64):                
            # take gradient step
            #todo critic loss...  


        self.cma.push(track_values)
        
        
    def action_logprobs(self, actor, states, actions):
        dist = self.__categorical(actor(states))
        return dist.log_prob(actions)
        
    def evaluate(self, states, actions):
        '''
            Compute log action probablities, value predictions and action entropy
        '''
        dist = self.__categorical(self.actor(states))
    
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        
        state_value = self.critic(states)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def policy(self, env):
        action_probs = lambda state: tu.to_numpy(self.__categorical(self.actor(torch.FloatTensor(state[np.newaxis]))).probs.squeeze())
        return gu.policy.probabilistic_policy(env, action_probs)
