#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:53:40 2019

@author: Benedict Wilkins
"""
import numpy as np
import itertools

from inspect import signature

from ..debugutils import assertion

from . import accumulate
from . import function
from . import random
from . import timeseries
from . import stat

DATASET_REPOSITORY = "/home/ben/Documents/repos/datasets/" #what ever you want...

__all__ = ('accumulate', 'function', 'random', 'timeseries', 'stat')

''' #meh remove them...
def arg(args, name, default):
    if name in args:
        return args[name]
    else:
        args[name] = default
        return default

def exit_on(iterator, on):
    for x in iterator:
        yield x
        if on():
            return
'''

from .batch import batch_iterator

def window1d(x, size, step=1):
    """ Compute a sliding window over the given 1D array. If the size/step are not compatible with tje size of, trailing elements of x will be trimmed.
        WARNING: uses np.lib.stride_tricks, this function should only be used for preprocessing, modifying the resulting array could have undesirable effects.

    Args:
        x (np.ndarray): array to slide a window over
        size (int): window size
        step (int, optional): step. Defaults to 1.

    Returns:
        numpy.ndarray: 2D windowed array
    """
    return np.lib.stride_tricks.as_strided(x, shape=(x.shape[0]//step - int(np.ceil(size/step)) + 1, size), strides=(x.strides[0]*step,x.strides[0]))



def correlation(x, y):
    """ Compute correlation of x and y. Assumes [n,...] for x and y where n is the number of samples.
    
    Args:
        x (numpy.ndarray): Collection of samples
        y (numpy.ndarray): Collection of samples

    Returns:
        numpy.ndarray :
    """
    if len(x.shape) == 1:
        x = x[...,np.newaxis]
    if len(y.shape) == 1:
        y = y[...,np.newaxis]
    
    xx, yy = x - x.mean(0), y - y.mean(0)
    xx, yy = xx.reshape(xx.shape[0],-1), yy.reshape(yy.shape[0],-1)
    xy = xx.T[...,np.newaxis] * yy[np.newaxis,...]
    xys = xy.sum(1)
    xys = xys.reshape(*x.shape[1:], *y.shape[1:])
    xs,ys = (xx*xx).sum(0), (yy*yy).sum(0)
    xxyys = np.sqrt(xs.T[...,np.newaxis] * ys[np.newaxis,...])
    xxyys = xxyys.reshape(*x.shape[1:], *y.shape[1:])
    return np.nan_to_num(xys / xxyys)



def window(x, shape):
    '''
        Slides a window (2D) of the given shape across the given array.
        Arguments:
            x: array
            shape: of window 
        Returns:
            3D array containing each possible window of the given shape (stride = 1), top-left to bottom-right.
    '''
    assert len(x.shape) == 2 #TODO colour... (HWC) currently only 2D arrays are supported :(
    assert len(shape) == 2

    s = (x.shape[0] - shape[0] + 1,) + (x.shape[1] - shape[1] + 1,) + shape
    strides = x.strides + x.strides
    r = np.lib.stride_tricks.as_strided(x, shape=s, strides=strides)
    return r.reshape(r.shape[0] * r.shape[1], *r.shape[2:])
    #return r


def window_b(x, shape):
    '''
        Batch version of window - applies sliding window over each 2d array in a batch.
        TODO
    '''
    assert len(x.shape) == 3 #use the non-batch version?
    assert len(shape) == 2
    
    shape = tuple(np.subtract(x.shape[1:], shape) + 1) + (x.shape[0],) + shape
    strides = (x.strides * 2)[1:]
    M = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
    return M.reshape(np.prod(M.shape[0:2]), *M.shape[2:]).swapaxes(0,1)


def invert(index, shape):
    '''
        Get inverted index e.g. [1,3,4,5], 10 -> [2,6,7,8,9]
        Args:
            index: to invert
            shape: of the full index array
        Return:
            inverted index
    '''
    t = np.ones(shape, dtype=np.uint8)
    t[index] = 0
    return np.where(t)[0]
     
def repeat(iterator, n, *args, **kwargs):
    '''
        An iterator that repeats another iterator n times
        Args:
            iterator: iterator definiton
            n: number of times to repeat
            *args: arguments for the iterator
            **kwargs: arguments for the iterator
    '''
    for i in range(n):
        for x in iterator(*args, **kwargs):
            yield (i, *x)
            
def limit(iterable, limit):
    '''
        Limits the iterations of an iterable to limit
        Args:
            iterator: an iterable
            limit: number of iterations before stopping
    '''
    return itertools.islice(iterable, limit)

#??? no idea what this was for..
def display_increments2(total):
    j = 0
    i = 0
    while j < total:
        j = i**2
        yield j
        i += 1        

def onehot(x, size, dtype=np.float32):
    x = x.astype(np.int64)
    r = np.zeros((x.shape[0], size), dtype=dtype)
    r[np.arange(x.shape[0]), x] = 1
    return r

def splitbylabel(x, y):
    result = {}
    for label in np.unique(y):
        result[label] = x[(y==label).squeeze()]
    return result


        
''' DATASET '''

def __shape__(i):
    try:
        return i.shape
    except:
        pass
    try:
        return (len(i),)
    except:
        pass
    return (1,)

def __dtype__(i):
    try:
        return i.dtype
    except:
        pass
    try:
        return type(i[0])
    except:
        pass
    return type(i)


def __non_singular(iterator): #TODO refactor - use zip (see batch)
    def non_singular_iterator(iterator):
        for x in iterator:
            yield (x,)
            
    x = next(iterator)
    
    singular = not isinstance(x, tuple)
    if singular:
        iterator = non_singular_iterator(iterator)
        x = (x,)
    return iterator, x    

def __init_dataset(iterator, size, template=None): #TODO refactor
    iterator, x = __non_singular(iterator)
    
    if template is None:
        result = tuple([np.empty((size, *__shape__(i)), dtype=__dtype__(i)) for i in x])
    else:
        assertion(not isinstance(template, tuple), 'template must be a tuple (may be singleton)')
        assertion(not len(template) == len(x), 'invalid template size')
        for i,t in enumerate(template):
            #print(tuple(t.shape[1:]), __shape__(x[i]))
            assertion(not tuple(t.shape[1:]) == __shape__(x[i]), 
                      'template shape {0} and example shape {1} do not match at position {2}'.format(t.shape[1:], __shape__(x[i]), i))
        result = template
        
    for j in range(len(x)):
        result[j][0] = x[j]
    
    return iterator, result
    
def dataset(iterator, size=1000, template=None, progress=0): #TODO refactor
    iterator, result = __init_dataset(iterator, size, template)
    iterator = itertools.islice(iterator, 0, size-1)
    if not progress:
        for i, x in enumerate(iterator, 1):
            for j in range(len(x)):
                result[j][i] = x[j]
        return result
    else:
        print("Constructing dataset of size: %d" % size)
        for i, x in enumerate(iterator, 1):
            if not i % progress:
                print("progress: %d/%d" % (i, size))
            for j in range(len(x)):
                result[j][i] = x[j]
        print("progress: %d/%d, done" % (size, size))
        return result

#refactor at some point...    
def no_count(batch_iterator):
    for b in batch_iterator:
        yield b[1:]

def __shuffle__(*data):
    return shuffle(*data)

def shuffle(*data): #sad that this cant be done in place...
    m = max(len(d) for d in data)
    indx = np.arange(m)
    np.random.shuffle(indx)
    data = [d[indx] for d in data]
    return data

def apply(iterable, fun, unpack=True): ##????
    if unpack:
        for i in iterable:
            yield fun(*i)
    else:
        for i in iterable:
            yield fun(i)
            
def pack_apply(iterable, fun, unpack=True): #TODO ????? 
    iterable = apply(iterable, fun, unpack)
    return pack(iterable)

def pack(iterator): #????
    '''
        Packs the content of an iterator into numpy arrays
        Args:
            iterator: with which to iterate over and collect values
    '''
    iterator, x = __non_singular(iterator)

    result = tuple([[z] for z in x])
        
    for x in iterator:
        for j in range(len(x)):
            result[j].append(x[j])  
            
    return tuple([np.array(z) for z in result])

def normalise(data, axis=None):
    maxd = np.max(data, axis=axis)
    mind = np.min(data, axis=axis)
    
    if axis is not None:
        broadcast_shape = list(mind.shape)
        broadcast_shape.insert(axis,1)
        maxd = maxd.reshape(broadcast_shape)
        mind = mind.reshape(broadcast_shape)

    return (data - mind) / (maxd - mind)

def group_avg(x,y):
    unique = np.unique(x, axis=0)

    avg = np.empty(unique.shape[0])
    for i, u in enumerate(unique):
        avg[i] = y[(x==u).all(1)].mean()
    return unique, avg

import threading

def quit_on_key(key='q'):
    
    class QuitThread(threading.Thread):
    
        def __init__(self, key, name='quit-thread'):
            self.key = key
            super(QuitThread, self).__init__(name=name)
            self.setDaemon(True)
            self.start()
            self.__quit = False
    
        def run(self):
            self.__quit = input() == self.key
            while not self.__quit:
                self.__quit = input() == self.key
                
        def quit(self):
            return self.__quit
    
    return QuitThread(key).quit
        
   


    

    
    

    
    
    