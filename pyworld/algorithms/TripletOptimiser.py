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

mode = namedtuple('mode', 'all top top_n, top_p')(0,1,2,3)  
 
class TripletOptimiser(Optimiser):
         
    def __init__(self, model, margin = 0.2, norm=None, mode = mode.all, k = 16, lr=0.0005):
        super(TripletOptimiser, self).__init__(model)
        self.mode = mode
        self.__top = [(False, False), (True, True), (True, False), (False, True)]
        self.k = k #should be related to the batch size or number of p/n examples expected
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = du.CMA('loss')
        self.margin = margin

        if norm is None:
            self.norm = lambda x, y: (x-y).pow(2)
        else:
            self.norm = norm
    
    def step(self, x, y):
        self.optim.zero_grad()
        loss = self.__loss(x, y.squeeze(), *self.__top[self.mode])
        self.cma.push(loss.item())
        loss.backward()
        self.optim.step()
        return loss.item()
        

    def __loss(self, x, y, topk_n = False, topk_p = False):
        x_ = self.model(x)
        d = self.distance_matrix(x_)
        unique = np.unique(y)
        loss = torch.FloatTensor(np.array([0.])).to(self.model.device)
        #print("y", y)
        for u in unique:
            pi = np.nonzero(y == u)[0]
            ni = np.nonzero(y != u)[0]
            
            xp = d[pi][:,pi]
            xn = d[pi][:,ni]
            
            if topk_p:
                xp = self.topk2(xp, self.k, large=True)
            if topk_n:
                #print(xn)
                xn = self.topk2(xn, self.k, large=False)
                #print(xn)
                
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
        # if we want the top k in the whole matrix, this makes later computations a bit tricky...
        # use topk2
        indx = torch.topk(x.view(-1), k, largest=large)[1]
        return indx / x.shape[1], indx % x.shape[1]

    def topk2(self, x, k, large=False):
        if k >= x.shape[1]:
            return x
        else:
            return torch.topk(x, k, dim=1, largest=large)[0]



if __name__ == "__main__":
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.torchutils as tu
    from pyworld.algorithms.CNet import CNet2
  
    savepath = 'test/imgs/mnist'
    video =  True
    np.random.seed(0)
    
    device = tu.device()
    latent_dim = 2
    epochs = 1
    batch_size = 64
    x,y,_,_ = du.mnist()
    k = 12

    x = torch.FloatTensor(x)

    model = CNet2(x.shape[1:], latent_dim).to(device)
    fig = vu.plot2D(tu.tonumpy(model), x[:10000], y[:10000], pause=1.0,  xlim=(-2,2), ylim=(-2,2), draw=not video)
    if video:
        video = [vu.figtoimage(fig)]

    tro = TripletOptimiser(model, k = k, mode=mode.top_n)
    iterator = du.repeat(du.batch_iterator, epochs, x, y, batch_size = batch_size, shuffle = True)
    iterator = du.exit_on(iterator, vu.matlplot_isclosed)
    
    #for e, i, batch_x, batch_y in du.epoch_iterator(x, y, batch_size = batch_size, shuffle = True, epochs = epochs, exit_on = vu.matlplot_isclosed):
    for e, i, batch_x, batch_y in iterator:  
        tro.step(batch_x, batch_y)
        if not (i / batch_size) % 10:
            #fig = vu.plot2D(tu.tonumpy(model), x[:10000], y[:10000], fig = fig, xlim=(-2,2), ylim=(-2,2), draw=not video)
            print(int(i / batch_size), tro.cma())
            tro.cma.reset()
            
        if video:
            fig = vu.plot2D(tu.tonumpy(model), x[:10000], y[:10000], fig = fig, xlim=(-2,2), ylim=(-2,2), draw=not video)
            video.append(vu.figtoimage(fig))

    if video:
        vu.savevideo(video, savepath)

    
    
            

    
    
    
