#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:56:21 2019

@author: ben
"""
import os
import pickle

def name_file(file):
    i = 0
    o_file = file
    while os.path.isfile(file):
        i += 1
        file = o_file + "-" + str(i)
    return file

def load(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data