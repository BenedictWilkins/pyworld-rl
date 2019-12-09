#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:29:42 2019

author: Benedict Wilkins
"""
import numpy as np
import torch

import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu

fun = lambda x: x + 1

data1 = np.array([[0],[1],[2],[3],[4]])
data2 =  np.array([[0],[1],[2],[3],[4]])

print(du.collect(fun, data1, data2, batch_size=2))

data1 = torch.FloatTensor(data1)
data2 = torch.FloatTensor(data2)

print(tu.collect(fun, data1, data2, batch_size=2))
