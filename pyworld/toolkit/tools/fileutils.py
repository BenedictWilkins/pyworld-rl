#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 16:56:21 2019

@author: ben
"""
import os
import pickle
import datetime
import json

import numpy as np

def __load_mpz(path, max_size=100000):
    if os.path.isfile(path):
        data = np.load(path)
        for a in data:
            print(data[a])
            
        yield {k:v for k,v in np.load(path)}
    elif os.path.isdir(path): 
        fs = files(path)
        for f in fs:

            yield {k:v for k,v in np.load(f)}
    
def __save_mpz(path, data, z=False):
    if not z:
        np.savez(path, data)
    else:
        np.savez_compressed(path, data)

def __load_pickle(file):
    with open(file, 'rb') as fp:
        data = pickle.load(fp)
    return data

def __save_pickle(file, data):
    with open(file, 'wb') as fp:
        pickle.dump(data, fp)


__load = {'.npz':__load_mpz, '.pickle':__load_pickle, '.pkl':__load_pickle, '.p':__load_pickle}
__save = {'.npz':__save_mpz, '.pickle':__save_pickle, '.pkl':__save_pickle, '.p':__save_pickle}

def load(path, **kwargs):
    if has_extension(path):
       _, ext = os.path.splitext(path)
       return __load[ext](path, **kwargs)

def save(path, data, **kwargs):
   if has_extension(path):
       _, ext = os.path.splitext(path)
       __save[ext](path, data, **kwargs)
    

def file(file, force=False):
    path, _ = os.path.split(file)
    if not os.path.exists(path):
        os.makedirs(path)
    
    o_file, ext = os.path.splitext(file)
    i = 0
    while os.path.isfile(file):
        i += 1
        file = '.'.join([o_file + "(" + str(i) + ")", ext])
    return file

def files(path):
    assert os.path.isdir(path)
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def dirs(path):
    assert os.path.isdir(path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def file_datetime():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def has_extension(file):
    return len(file.split('.')) > 1

def save_json(obj, f):
    f = file(f)
    with open(f, 'w') as fp:
        json.dump(obj, fp)

def load_json(f):
    with open(f, 'r') as fp:
        data = json.load(fp)
    return data


if __name__ == "__main__":
    save('./test/test.npz', (np.array([10,10,10]), np.array([1,1,1,1])))
    print(load('./test/test.npz'))
    for data in load('./test/test.npz'):

        print(data)