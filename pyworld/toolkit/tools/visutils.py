#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:41:32 2019

@author: ben
"""
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from . import fileutils as fu
from . import datautils as du

def save(image, path, extension = ".png"):
    image = du.normalise(image)
    if not fu.has_extension(path):
        path = path + extension
    path = fu.file(path)
    print(path)
    plt.imsave(path, colour(image))

def resize_all(images, shape):
    result = np.empty((images.shape[0], *shape, images.shape[3]))
    for i in range(len(images)):
        result[i] = cv2.resize(images[i], shape)[:,:,np.newaxis]
    return result

def crop(image, shape):
    cv2.crop

def resize(image, shape, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(image, (shape[1], shape[0]), interpolation=interpolation)

def scale(image, x, y):
    return cv2.resize(image, None, fx=x, fy=y)

def gray(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def colour(image):
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

def channels_to_torch(data):
    if len(data.shape) == 4:
        return data.reshape((data.shape[0], data.shape[3], data.shape[1], data.shape[2]))
    elif len(data.shape) == 3:
        return data.reshape((data.shape[2], data.shape[0], data.shape[1]))

def channels_to_cv(data):
    if len(data.shape) == 4:
        return data.reshape((data.shape[0], data.shape[2], data.shape[3], data.shape[1]))
    elif len(data.shape) == 3:
        return data.reshape((data.shape[1], data.shape[2], data.shape[0]))
    
def shape_to_torch(shape):
    if len(shape) == 4:
        return (shape[0], shape[3], shape[1], shape[2])
    elif len(shape) == 3:
        return (shape[2], shape[0], shape[1])

def shape_to_cv(shape):
    if len(shape) == 4:
        return (shape[0], shape[2], shape[3], shape[1])
    elif len(shape) == 3:
        return (shape[1], shape[2], shape[0])

'''
def gallery(array, ncols=3):
    nindex, height, width = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols))
    return result
'''
def label_colours(labels, alpha=0.8):
    colours = cm.rainbow(np.linspace(0,1,len(labels)))
    colours[:,3] *= alpha
    result = {}
    for i in range(colours.shape[0]):
        result[labels[i]] = colours[i]
    return result
    

def plot2D(model, x, y=None, fig=None, clf=True, marker=".", alpha=0.8, 
                  title=None, xlabel=None, ylabel=None, xlim=None, ylim=None, pause=0.001):
    if fig is None:
        fig = plt.figure()
    if clf:
        fig.clf()

    
    z = du.collect(model, x)
    assert z.shape[1] == 2
    line_handles = []
    label_handles = []
    if y is None:
        y = np.ones(x.shape[0])

    data = du.splitbylabel(z, y)
    colours = label_colours(list(data.keys()), alpha)
    
    for label,d in data.items():
        line = plt.scatter(d[:,0], d[:,1], color=colours[label],  edgecolors='none', label=label, marker=marker)
        line_handles.append(line)
        label_handles.append(label)
       
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
        
    plt.legend(line_handles, label_handles, loc="upper right")
    plt.suptitle(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.draw()
    plt.pause(pause)
    return fig

#123, 252, 3
def show_attention(values, queries, weights, embed_size=None, display_shape=(240,480),
                   wait=240, line_colour=(0.48235, 0.98823, 0.01176), line_tickness=2):
    if embed_size is None:
        embed_size = np.max([*values.shape, *queries.shape])
    embed = np.ones((2 + embed_size, embed_size * 2 + 3, values.shape[2]))
    embed[1:queries.shape[0]+1,1:queries.shape[1]+1] = queries
    embed[1:values.shape[0]+1,2+embed_size:2+embed_size+values.shape[1]] = values
    
    values_size = np.prod(values.shape[:-1])
    queries_size = np.prod(queries.shape[:-1])
    
    lines1, lines2 = [], []
    incx = int(display_shape[1] / embed.shape[1])
    incy = int(display_shape[0] / embed.shape[0])
    incx2 = int(incx / 2)
    incy2 = int(incy / 2)
    for i in range(queries_size):
        for j in range(values_size):
            lines1.append((incx * ((i % queries.shape[0]) + 1) + incx2, incx * ((j % values.shape[0]) + 2 + embed_size) + incx2))
            lines2.append((incy * ((i // queries.shape[1]) + 1)+ incy2, incy * (( j // values.shape[1]) + 1) + + incy2))

    
    embed = resize(embed.astype(np.float32), display_shape)
    embed = colour(embed)
    
    #construct video

    ii = 0
    frames = []  
    plt.ioff()
    for i in range(values_size, queries_size * values_size + 1, values_size):
        plt.clf()
        plt.imshow(embed)
        for p1, p2 in zip(lines1[ii:i], lines2[ii:i]):
            print(p1, p2)
            plt.plot(p1, p2, linewidth=line_tickness, alpha=0.5, color=line_colour)
            #cv2.line(change, p1, p2, line_colour, line_tickness, lineType=cv2.LINE_AA)
        frames.append(figure_to_numpy(plt.gcf()))
        ii = i

    play(frames, 'attention', wait=wait, repeat=True)

def covariance(images, title=None, ):
    print(images.shape)
    flattened = images.reshape(-1, np.prod(images[0].shape)).T
    covar = np.cov(flattened)
    print(covar.shape)
    plt.figure()
    plt.title(title)
    #plt.axis('off')
    plt.imshow(covar, cmap='viridis')
    
    
def figure_to_numpy(fig):
    # draw the renderer
    fig.canvas.draw() 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height() 
    return np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h,w,3)  
    
def gallery(array, ncols=3):
    nindex, height, width, intensity = array.shape
    nrows = nindex//ncols
    assert nindex == nrows*ncols
    # want result.shape = (height*nrows, width*ncols, intensity)
    result = (array.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return result

def show(array, name='image', wait=60):
    cv2.imshow(name, array)
    if wait >= 0:
        return cv2.waitKey(wait)
    return
    
def play(video, name='image', wait=60, repeat=False):
    while True: 
        for f in video:
            cv2.imshow(name, f)
            if cv2.waitKey(wait) == ord('q'):
                close(name)
                return
        if not repeat:
            close(name)
            return
    

def close(name=None):
    if name is None: 
        cv2.destroyAllWindows()
    else:
        cv2.destroyWindow(name)
        
def waitclose(wait=60):
    if cv2.waitKey(wait) == ord('q'):
        close()
        return True
    return False