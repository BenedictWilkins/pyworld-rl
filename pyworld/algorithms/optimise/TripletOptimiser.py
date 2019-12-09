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
from .Optimise import Optimiser

mode = namedtuple('mode', 'all top top_n, top_p')(0,1,2,3)  #enum?

class TripletOptimiser(Optimiser):
         
    def __init__(self, model, margin = 0.2, mode = mode.all, k = 16, lr=0.0005):
        super(TripletOptimiser, self).__init__(model)
        self.mode = mode
        self.__top = [(False, False), (True, True), (True, False), (False, True)] #topk_n, topk_p
        self.k = int(k) #should be related to the batch size or number of p/n examples expected
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = du.CMA('loss')
        self.margin = margin
    
    def step(self, x, y):
        self.optim.zero_grad()
        loss = self.__loss(x, y.squeeze(), *self.__top[self.mode])
        self.cma.push(loss.item())
        loss.backward()
        self.optim.step()
        return loss.item()
        
    def __loss(self, x, y, topk_n = False, topk_p = False):
        x_ = self.model(x)
        #d = self.distance_matrix(x_)
        unique = np.unique(y)
        loss = torch.FloatTensor(np.array([0.])).to(self.model.device)

        for u in unique:
            pi = np.nonzero(y == u)[0]
            ni = np.nonzero(y != u)[0]
            
            #xp_t = d[pi][:,pi]
            #xn_t = d[pi][:,ni]
            #slightly more efficient below
            xp_ = x_[pi]
            xn_ = x_[ni]
            xp = self.distance_matrix(xp_, xp_)
            xn = self.distance_matrix(xp_, xn_)

            if topk_p:
                xp = self.topk2(xp, self.k, large=True)
            if topk_n:
                xn = self.topk2(xn, self.k, large=False)
                
            #3D tensor, (a - p) - (a - n) 
            
            xf = xp.unsqueeze(2) - xn
            xf = F.relu(xf + self.margin) #triplet loss
            loss += xf.sum()

        return loss
        
    def distance_matrix(self, x1, x2=None):
        # TODO speed up...
        if x2 is None:
            x2 = x1
        n_dif = x1.unsqueeze(1) - x2.unsqueeze(0)
        return torch.sum(n_dif * n_dif, -1)
    
    ''' #speed up??!
    def dmatrix(x1,x2=None):
        if x2 is None:
            x2 = x1
        dists = -2 * np.dot(x1, x2.T) + np.sum(x1**2, axis=1) + np.sum(x2**2, axis=1)[:, np.newaxis]
        return dists
    '''
    
    def topk(self, x, k, large=False):
        # if we want the top k in the whole matrix, this makes later computations a bit tricky...
        # use topk2
        indx = torch.topk(x.view(-1), k, largest=large)[1]
        return indx / x.shape[1], indx % x.shape[1]

    def topk2(self, x, k, large=False):
        if k >= x.shape[1]:
            return x
        else:
            return torch.topk(x, k, dim=1, largest=large)[0]

class PairTripletOptimiser(TripletOptimiser):
    
    def __init__(self, model, margin = 0.2, mode = mode.all, k = 16, lr=0.0005):
        super(PairTripletOptimiser, self).__init__(model, margin, mode, k, lr)
        
    def step(self, x1, x2):
        '''
            Step with pairs of input. ``(x1_i, x2_i)`` are a pair - share the same label
            (x1_i, x2_{j\neq i}) are considered to have different labels (i.e. are not a pair).
        '''
        self.optim.zero_grad()
        loss = self.__loss(x1, x2, *self._TripletOptimiser__top[self.mode])
        self.cma.push(loss.item())
        loss.backward()
        self.optim.step()
        return loss.item()
    
    def __loss(self, x1, x2, topk_n = False, topk_p = False):
        
        x1_ = self.model(x1)
        x2_ = self.model(x2)
        
        d = self.distance_matrix(x1_, x2_)
        xp = torch.diag(d).unsqueeze(1)
        xn = d # careful with the diagonal!

        if topk_n and self.k < xn.shape[0]:
            xn[range(d.shape[0]), range(d.shape[1])] = float('inf') #hopefully this doesnt mess up autograd.....
            xn = self.topk2(d, self.k, large=False) #select the k best negative values for each anchor
            xf = xp.unsqueeze(2) - xn #should only consist of only ||A-P|| - ||A-N||
        else:
            xf = xp.unsqueeze(2) - xn
            xf[:,range(d.shape[0]), range(d.shape[1])] = 0. #remove all ||A-P|| - ||A-P||
            
        xf = F.relu(xf + self.margin) 
        '''
        print(xn.shape)
        print(xp.shape)
        print(xn)
        print(xp)
        print(xf)
        print(xf.sum())
        '''
        return xf.sum()
    


if __name__ == "__main__":
    
    class StubModel(torch.nn.Module):
        
        def __init__(self):
            super(StubModel, self).__init__()
            self.device = 'cpu'
            self.l1 = torch.nn.Linear(1,1)
        
        def forward(self, x):
            return x
        
    
    model = StubModel()
    x1 = torch.FloatTensor([1,3,5]).unsqueeze(1)
    x2 = torch.FloatTensor(np.arange(3)).unsqueeze(1)
    print(x1)
    print(x2)
    tro = PairTripletOptimiser(model, k=1, mode=mode.top)
    tro.step(x1, x2)
    
    

    
    
    
