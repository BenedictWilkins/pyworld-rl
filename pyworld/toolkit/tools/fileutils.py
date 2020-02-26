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
import re
import numpy as np
import matplotlib.pyplot as plt


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




def __matplot_figure_to_image(fig):
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return np.flip(buf.reshape((h, w, 3)), 2) #bgr format for opencv!

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
    #for some reason returning the model here can cause issues (it is not loaded properly at the return...?)
   
def __save_image(file, image):
    if isinstance(image, plt.Figure):
        image = __matplot_figure_to_image(image)
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

def __save_txt(file, string):
    f = open(file, 'a')
    f.write(string)
    return f
    
def __load_txt(file):
    with open(file, 'r') as f:
        return f.readlines() 

def __save_mp4(file, video, fps=24, format='bgr'):
    #TODO ensure NHWC format...
    if format.lower() == 'rgb': #opencv works with bgr format, reverse the 
        video = video[:,:,:,::-1]
    
    if issubclass(video.dtype.type, np.integer):
        if video.dtype.type != np.uint8:
            video = video.astype(np.uint8)
    elif issubclass(video.dtype.type, np.floating):
        print("VIDEO WRITE WARNING: the supplied video dtype is {0} assuming an interval 0-1 for video encoding".format(str(video.dtype)))
        video = (video * 255.).astype(np.uint8)
    
    #video must be CV format (NHWC)
    colour = len(video.shape) == 4 and video.shape[3] == 3

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(file, fourcc, fps, (video.shape[2],video.shape[1]), colour)
    for frame in video:
        writer.write(frame)
    
def __load_mp4(file):
    raise NotImplementedError("TODO!")

def __save_avi(file):
    raise NotImplementedError("TODO!")

def __load_avi(file, as_numpy=True): #will be read as NHWC int format
    vidcap = cv2.VideoCapture(file)
    success, image = vidcap.read()
    images = []
    while success:
        images.append(image)
        success, image = vidcap.read()
    return np.array(images)

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

    for k, d in data.items():
        dataset = f.create_dataset(str(k), data = d, compression=compression)

def __load_hdf5(file):
    return h5py.File(file, 'r')

__load = {'.txt':__load_txt, '.yaml':__load_yaml, '.json':__load_json, '.npz':__load_mpz, '.pickle':__load_pickle, '.pkl':__load_pickle, '.p':__load_pickle, '.pt':__load_torch,
          '.png':__load_image, '.jpg':__load_image, '.gif':__load_gif, '.hdf5':__load_hdf5, '.mp4':__load_mp4, '.avi':__load_avi}
__save = {'.txt':__save_txt, '.yaml':__save_yaml, '.json':__save_json, '.npz':__save_mpz, '.pickle':__save_pickle, '.pkl':__save_pickle, '.p':__save_pickle, '.pt':__save_torch,
          '.png':__save_image, '.jpg':__save_image, '.gif':__save_gif, '.hdf5':__save_hdf5, '.mp4':__save_mp4, '.avi':__save_avi}

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




#TODO rename
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
    if not os.path.isdir(path):
        raise ValueError("path {0} does not exist".format(path))

    if not full:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def sort_files(files):
    '''
        Sorts file by their tag (X): 
        foo.txt
        foo(1).txt
        ...
        foo(N).txt
    '''
    def number(file):
        f = os.path.splitext(os.path.basename(file))[0]
        x = re.findall("\(([0-9]+)\)", f)
        if len(x) == 1:
            return int(x[0])
        elif len(x) == 0:
            return 0
        else:
            raise ValueError("Invalid file name for sort: {0}".format(f))
    return sorted(files, key = lambda x: number(x))


def dirs(path, full=False):
    if not os.path.isdir(path):
        raise ValueError("path {0} does not exist".format(path))
    if not full:
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def file_datetime():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def has_extension(file):
    return len(file.split('.')) > 1

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def files_with_extention(path, extension, full=False):
    return [file for file in files(path, full=full) if file.endswith(extension)]

if __name__ == "__main__":
    save('~/Documents/test.mp4', np.random.uniform(size=(100,200,100, 3)))