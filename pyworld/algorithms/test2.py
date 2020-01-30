#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:29:22 2019

@author: ben
"""


import numpy as np
import torch
size = 6
np.random.seed(1)
x = torch.FloatTensor(np.random.randint(0, 10, size = (size,2)))
y = np.array([0,1,0,1,1,1]) #np.random.randint(0,3, size=size)



def distance_matrix(x):
    # TODO speed up...
    n_dif = x.unsqueeze(1) - x.unsqueeze(0)
    return torch.sum(n_dif * n_dif, -1)

def topk(x, k, large=False):
    if k >= np.prod(x.shape):
        indx = np.indices(x.shape)
        return indx
        #return indx[0].flatten(), indx[1].flatten()
    print("topk", torch.topk(x.view(-1), k, largest=large)[1])
    indx = torch.topk(x.view(-1), k, largest=large)[1]
    return indx / x.shape[1], indx % x.shape[1]

def topk2(x, k, large=False):
    return torch.topk(x, k, dim=1, largest=large)[0]


def latex_matrix(x, dtype=float):
    print(x)
    s = " "
    for i in range(x.shape[0]):
        for j in range(x.shape[1] - 1):
            s += str(dtype(x[i, j].item())) + " & "
        s += str(dtype(x[i,x.shape[1]-1].item())) + " \\\\ \n "
    return s
    


k = 2
d = distance_matrix(x)
unique = np.unique(y)
print(d)
for u in unique:
    pi = np.nonzero(y == u)[0]
    ni = np.nonzero(y != u)[0]
    
    xp = d[pi][:,pi]
    xn = d[pi][:,ni]
    
    xf = xp.unsqueeze(2) - xn
    
    print(xp)
    print(xn)
    #print(xf)
    xn = topk2(xn, k)
    print(xn)

    #print(xn)
    xf = xp.unsqueeze(2) - xn
    
    print(xf)
    break
    
    
