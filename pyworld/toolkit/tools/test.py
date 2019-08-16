#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:32:54 2019

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.normal(size=1000)
x2 = np.random.normal(size=1000)

y = np.outer(x1, x2)
print(y)
y = y.reshape(x1.shape[0] * x2.shape[0])

plt.hist(y, normed=True, bins=50) #hmm.. wishart?