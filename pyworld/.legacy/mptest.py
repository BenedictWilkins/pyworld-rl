#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:30:22 2019

@author: ben
"""

import torch.multiprocessing as mp #used for multi-environment parallelism
import random

class T:
    
    def __init__(self):
        self.i = 0.
        
    def inc(self, x):
        print(x)
        self.i += x
        
PROCS = 2
t = T()
v = mp.Value('t', t)

def change():
    v.inc(random.random())
    


processes = []
for i in range(PROCS):
    p = mp.Process(target=change)
    p.start()
    processes.append(p)
for p in processes:
    p.join()
    