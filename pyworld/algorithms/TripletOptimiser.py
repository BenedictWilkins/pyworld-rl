#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:26:27 2019

@author: ben
"""

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple


import pyworld.toolkit.tools.datautils as du
from pyworld.algorithms.optimise import Optimiser
        
class TripletOptimiser(Optimiser):
         
    MODE = namedtuple('mode', 'all top')(0,1)
    
    def __init__(self, model, margin = 0.2, norm=None, mode = MODE.all, k = 64, lr=0.0005):
        super(TripletOptimiser, self).__init__(model)
        self.__losses = [self.__loss_all, self.__loss_top]
        self.mode = mode
        if self.mode == 1:
            self.k = k
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = du.CMA('loss')
        self.margin = margin

        if norm is None:
            self.norm = lambda x, y: (x-y).pow(2)
        else:
            self.norm = norm
    
    def step(self, x, y):
        self.optim.zero_grad()
        loss = self.__losses[self.mode](x, y.squeeze())
        self.cma.push(loss.item())
        loss.backward()
        self.optim.step()
        return loss.item()

    def __loss_top(self, x, y):
        #selects the top k distances only
        raise NotImplementedError()
        
    def __loss_all(self, x, y):
        x_ = self.model(x)
        d = self.distance_matrix(x_)
        unique = np.unique(y)
        loss = torch.FloatTensor(np.array([0.])).to(self.model.device)
        for u in unique:
            pi = np.nonzero(y == u)[0]
            ni = np.nonzero(y != u)[0]
            xp = d[pi][:,pi]
            xn = d[pi][:,ni]
            #3D tensor, (a - p) - (a - n) 
            #indexed as xf[:,i,:] for the ith anchor
            xf = xp.unsqueeze(2) - xn
            xf = F.relu(xf + self.margin) #triplet loss
            # xf.sum(0).sum(1) loss for a given anchor i
            loss += xf.sum()
        return loss
            
    def distance_matrix(self, x):
        # TODO speed up...
        n_dif = x.unsqueeze(1) - x.unsqueeze(0)
        return torch.sum(n_dif * n_dif, -1)
    
    def topk(self, x, k, large=False):
        indx = torch.topk(x.view(-1), k, largest=large)[1]
        return indx / x.shape[1], indx % x.shape[1]


if __name__ == "__main__":
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.torchutils as tu
    from pyworld.algorithms.CNet import CNet2
  
    device = tu.device()
    latent_dim = 2
    epochs = 10
    x,y,_,_ = du.mnist()

    x = torch.FloatTensor(x)

    model = CNet2(x.shape[1:], latent_dim).to(device)
    fig = vu.plot2D(tu.tonumpy(model), x[:10000], y[:10000])
    
    tro = TripletOptimiser(model)
    for e in range(epochs):
        for batch_x, batch_y, i in du.batch_iterator(x, y, shuffle=True):
            tro.step(batch_x, batch_y)
            if not i % 10:
                print(tro.cma())
                tro.cma.reset()
                fig = vu.plot2D(tu.tonumpy(model), x[:10000], y[:10000], fig = fig, xlim=(-1.5,1.5), ylim=(-1.5,1.5))
    

    
    
    