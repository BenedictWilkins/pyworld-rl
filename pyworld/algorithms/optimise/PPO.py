#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 14:42:16 2019

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
    
class PPOModel(nn.Module):
        
    def __init__(self, actor, critic):
        assert actor.device == critic.device
        super(PPOModel, self).__init__()
        self.actor = actor
        self.critic = critic
        self.device = actor.device #critic.device
        super(PPOModel, self).add_module('actor', self.actor)
        super(PPOModel, self).add_module('critic', self.critic)
        
    def forward(self):
        raise NotImplementedError
        
class PPO(Optimiser):
    
    def __init__(self, actor, critic, logits=True, gamma=0.99, lr=0.002, eps_clip=0.2):
        super(PPO, self).__init__(PPOModel(actor, critic))
      
        self.actor = actor
        self.critic = critic
        
        self.old_actor = copy.deepcopy(actor)
        
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.cma = du.CMA('loss', 'policy_loss', 'value_loss', 'entropy_loss', 'reward')
        
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
        
        track_values = np.zeros((5,))
        track_values[-1] = total_reward
        
        #optimise the value function for a bit... (so that it can catch up with the policy current)
        '''
        for i in range(40):
            values = self.critic(states).squeeze()
            value_loss = 0.5 * F.mse_loss(values, returns).sum()
            self.optimiser.zero_grad()
            value_loss.backward()
            self.optimiser.step()
                
            #for b_states, b_returns in du.batch_iterator(states, returns, count=False, shuffle=False, batch_size=64):                
            # take gradient step
            #todo critic loss...  
        '''       
        
        for i in range(4): #hmmm?
            for b_states, b_actions, b_returns in du.batch_iterator(states, actions, returns, count=False, shuffle=False, batch_size=64):
                
                b_old_logprobs = self.action_logprobs(self.old_actor, b_states, b_actions)
                
                b_logprobs, b_values, b_action_entropy = self.evaluate(b_states, b_actions)
            
                # Finding the ratio (pi_theta / pi_theta__old):
                b_ratios = torch.exp(b_logprobs - b_old_logprobs.detach())
                    
                # Finding Surrogate Loss:
                b_advantages = b_returns - b_values.detach()
                surr1 = b_ratios * b_advantages
                surr2 = torch.clamp(b_ratios, 1-self.eps_clip, 1+self.eps_clip) * b_advantages
                policy_loss = - torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(b_values, b_returns).mean()
                entropy_loss = - 0.01 * b_action_entropy.mean()
                loss = policy_loss + value_loss + entropy_loss

                # take gradient step
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                track_values[:4] += np.array([loss.item(), policy_loss.item(), value_loss.item(), entropy_loss.item()])
            
        self.cma.push(track_values)
        
        self.old_actor.load_state_dict(self.actor.state_dict())
        
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

    def policy(self, action_space):
        actor = lambda state: tu.to_numpy(self.__categorical(self.actor(torch.as_tensor(state[np.newaxis], device=self.model.device).detach())).probs.squeeze())
        return gu.policy.probabilistic_policy(action_space, actor)

        
if __name__ == "__main__":
    from pyworld.toolkit.nn.CNet import CNet2
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.torchutils as tu
    from pyworld.toolkit.tools.wbutils import WB
    
    config = {'eps_clip':0.2,
              'batch_size':64,
              'gamma':0.99,
              'learning_rate':0.0005,
              'env':'Pong-v0',
              'env_binary':0.45,
              'env_stack':3,
              'no_entropy':False,
              'mse_reduction':'sum'}

    device = tu.device()
    
    env = gu.env(config['env'], binary=config['env_binary'], stack=config['env_stack'])
    input_shape = env.observation_space.shape
    action_shape = env.action_space.n
    
    critic = CNet2(input_shape, 1).to(device)
    actor = CNet2(input_shape, action_shape).to(device)
    
    ppo = PPO(actor, critic, lr=config['learning_rate'], eps_clip=config['eps_clip'], gamma=config['gamma'])
    
    policy = ppo.policy(env)
    
    wb = WB('ppo', ppo.model, config=config)
    
    
    with wb:
        #t = time.time()
        for i, episode in enumerate(gu.episodes(env, policy, mode=gu.mode.sar, epochs=10000)):
            #ot = t
            #t = time.time()
            #data_time = episode[0].shape[0] / (t - ot) 

            ppo.step(episode)
            #train_time = episode[0].shape[0] / (time.time() - t)
            
            
            print(ppo.cma.labelled())
            wb(**ppo.cma.recent())#, 
              # **{"time/data_time_bps":data_time,
              #"time/train_time_bps":train_time})

    
            if not i % 50:
                wb.save()
                ppo.cma.reset()
                vu.play(episode.state, wait=40)
                
            #t = time.time()

        
        
       
        
        
        
    
    
    
