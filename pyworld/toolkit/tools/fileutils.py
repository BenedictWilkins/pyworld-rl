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

try:
    import h5py #for storing large numeric arrays
except:
    pass

try:
   import torch #for torch model save/load
except:
   pass

try:
    import cv2 #for image save/load
except:
    pass

try:
    import moviepy.editor as mpy
except:
    pass

try:
    import yaml
except:
    pass

def __save_yaml(path, data):
    with open(path) as file:
        yaml.dump(data, file)

def __load_yaml(path):
    with open(path) as file:
        data = yaml.full_load(file)
    return data

def __save_json(f, obj):
    with open(f, 'w') as fp:
        json.dump(obj, fp)

def __load_json(f):
    with open(f, 'r') as fp:
        data = json.load(fp)
    return data

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

def __save_torch(file, model):
    torch.save(model.state_dict(), file)
   
def __load_torch(file, model=None):
    assert model is not None #must provide a template model when loading a pytorch model
    model.load_state_dict(torch.load(file))
   
def __save_image(file, image):
    cv2.imwrite(file, image)

def __load_image(file):
    return cv2.imread(file)

def __save_gif(file, video,fps=24, duration=None):
    if isinstance(video, list):
        sequence = video
    else:
        sequence = [x for x in video] #sigh...
    if duration is not None:
        durations = [duration/len(sequence) for _ in range(len(sequence))]
        clip = mpy.ImageSequenceClip(sequence, durations = durations)
    else:
        clip = mpy.ImageSequenceClip(sequence, fps=fps)

    clip.write_gif(file, fps=fps, program='ffmpeg')

def __load_gif(file):
    raise NotImplementedError("TODO!")

def __save_hdf5(file, data, chunk=None, groups=[], attrs={}, compress=True):
    print("SAVE: ", file)
    if chunk is not None:
        raise NotImplementedError("chunking is not implemented yet!")
    if len(attrs) > 0:
        raise NotImplementedError("attrs not implemented yet!")
    
    f = h5py.File(file, "w")

    compression = [None, 'gzip'][int(compress)]

    for group in groups:
        group = f.create_group(group)

    if isinstance(data, np.ndarray) or isinstance(data, list):
       data  = {"dataset":data}

    for k,d in data.items():
        dataset = f.create_dataset(str(k), data = d, compression=compression)

def __load_hdf5(file):
    return h5py.File(file, 'r')

__load = {'.yaml':__load_yaml, '.json':__load_json, '.npz':__load_mpz, '.pickle':__load_pickle, '.pkl':__load_pickle, '.p':__load_pickle, '.pt':__load_torch,
          '.png':__load_image, '.jpg':__load_image, '.gif':__load_gif, '.hd5f':__load_hdf5}
__save = {'.yaml':__save_yaml, '.json':__save_json, '.npz':__save_mpz, '.pickle':__save_pickle, '.pkl':__save_pickle, '.p':__save_pickle, '.pt':__save_torch,
          '.png':__save_image, '.jpg':__save_image, '.gif':__save_gif, '.hd5f':__save_hdf5}

def load(path, **kwargs):
    path = expand_user(path)

    if has_extension(path):
       _, ext = os.path.splitext(path)
       return __load[ext](path, **kwargs)

def save(path, data, force=True, overwrite=False, **kwargs):
    path = file(path, force=force, overwrite=overwrite)

    if has_extension(path):
       _, ext = os.path.splitext(path)
       return __save[ext](path, data, **kwargs)
    
def expand_user(path):
    if path.startswith("~"):
        return os.path.expanduser(path)
    return path

def file(file, force=True, overwrite=False):
    file = expand_user(file)

    if force:
        path, _ = os.path.split(file)
        if not os.path.exists(path):
            os.makedirs(path)

    if not overwrite:
        o_file, ext = os.path.splitext(file)
        i = 0
        while os.path.isfile(file):
            i += 1
            file = o_file + "(" + str(i) + ")" + ext

    return file

def next(file, force=True):
    o_file, ext = os.path.splitext(file)

    i = 0
    while os.path.exists(file):
        i += 1
        file = o_file + "(" + str(i) + ")" + ext
        
    if force:
        os.makedirs(file)

    return file

def files(path, full=False):
    path = expand_user(path)
    assert os.path.isdir(path)
    if not full:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [path + f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def dirs(path):
    assert os.path.isdir(path)
    return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

def file_datetime():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def has_extension(file):
    return len(file.split('.')) > 1



if __name__ == "__main__":
    save('./test/test.npz', (np.array([10,10,10]), np.array([1,1,1,1])))
    print(load('./test/test.npz'))
    for data in load('./test/test.npz'):

        print(data)