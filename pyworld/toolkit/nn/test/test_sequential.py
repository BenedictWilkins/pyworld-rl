#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 12:31:54

    Unit tests for Sequential.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import unittest

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pyworld.toolkit.nn.Sequential as Sequential


class TestSequential(unittest.TestCase):

    def test_init(self):
        net = Sequential.Sequential(l1=nn.Linear(10,10), l2=nn.Linear(10,10))
        self.assertIn('l1', dir(net))
        self.assertIn('l2', dir(net))

    def test_inverse(self):
        net = Sequential.Sequential(l1 = nn.Linear(10,20), l2=nn.Linear(20,2))
        inet = net.inverse()
        d = torch.from_numpy(np.random.uniform(0,1,size=(1,10)).astype(np.float32))

        o = net(d)
        self.assertEqual(tuple(o.shape), (1,2))
        i = inet(o)
        self.assertEqual(tuple(i.shape), (1,10))

if __name__ == "__main__":
    unittest.main()




