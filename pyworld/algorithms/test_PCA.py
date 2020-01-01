#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 15:36:03 2019

@author: ben
"""
from pyworld.algorithms import PCA

import numpy as np

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()

x = iris.data
y = iris.target
x_ = PCA(x)

for i, target_name in enumerate(iris.target_names):
    plt.scatter(x_[y == i, 0], x_[y == i, 1], label=target_name)

plt.legend()
plt.title('PCA of IRIS dataset')
plt.show()