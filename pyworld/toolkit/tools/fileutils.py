#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:56:21 2019

@author: ben
"""
import os
import pickle

def file(file):
    o_file, ext = split_extension(file)
    i = 0
    while os.path.isfile(file):
        i += 1
        file = '.'.join([o_file + "(" + str(i) + ")", ext])
    return file

def load(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def split_extension(file):
    fsplit = file.split('.')
    ext = ".".join(fsplit[1:])
    return fsplit[0], ext
    
def has_extension(file):
    return len(file.split('.')) > 1