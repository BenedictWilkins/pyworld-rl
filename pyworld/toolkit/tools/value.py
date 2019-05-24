#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 14:56:45 2019

@author: ben
"""

import torch
import torch.nn as nn
import numpy as np
import model

class ValueEstimator(model.LearningModel):
    
    def __init__(self, model, loss= torch.nn.MSELoss(), optim=None):
        if optim is None:
            optim = torch.optim.Adam(model.parameters(), lr=0.001)
        super(ValueEstimator, self).__init__(model, loss, optim)
    
    def step(self, batch):
        self.optim.zero_grad()
        states, discounted_rewards = batch
        vals_v = self.model(states)
        
        loss_v = self.loss(vals_v, discounted_rewards)
        loss_v.backward()
        
        self.optim.step()
        return loss_v.item()
    
    def __call__(self, obs):
        return self.model(obs)
    
class ValueConvNet1(nn.Module):
    
    def __init__(self, input_shape):
        super(ValueConvNet1, self).__init__()
        self.pipe = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU())
        conv_out_size = self._get_conv_out(input_shape) #lazy...
        self.fv = nn.Sequential(
            nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, 1))

    def _get_conv_out(self, shape):
        o = self.pipe(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.pipe(x).view(x.size()[0], -1)
        return self.fv(conv_out).squeeze()


   
if __name__ == "__main__":
    import gym
    import cv2
    import datautils
    import gymutils
    import torch
    
    def transform(state):
        # if state.size == 210 * 160 * 3:
        #   img = np.reshape(state, [state.shape[1], state.shape[2], state.shape[0]]).astype(np.float)
        # elif state.size == 250 * 160 * 3:
        #   img = np.reshape(state, [250, 160, 3]).astype(np.float)
        # else:
        #      assert False, "Unknown resolution."
        img = (state[:, :, 0] * 0.299 + state[:, :, 1] * 0.587 + state[:, :, 2] * 0.114).astype(np.float)
        img = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        return np.reshape(img[18:102, :], [1, 84, 84]).astype(np.float) / 255.0
    
    def transform_iterator(env, policy):
        for ob in gymutils.sr_iterator(env, policy):
            yield transform(ob[0]), ob[1]
    
    env_name = "SpaceInvaders-v0"
    env = gym.make(env_name)
    policy = gymutils.uniform_random_policy(env)
    EPISODES = 100
    BATCH_SIZE = 16
    
    
    ve = ValueEstimator(ValueConvNet1((1,84,84)))
    i = 0
    for episode in gymutils.sdr_episode_iterator(env, policy, iterator=transform_iterator):
        #shuffle episode and sample batches
        losses = []
        for batch in datautils.batch_iterator(*episode, batch_size=BATCH_SIZE, shuffle=True):
            batch = datautils.batch_to_tensor(batch, [torch.FloatTensor, torch.FloatTensor])
            losses.append(ve.step(batch))
        print("episode:", i, "loss:",np.mean(losses))
        losses.clear()
        i+=1
        if i > EPISODES:
            break
    
    
    
    
    
    
    
    