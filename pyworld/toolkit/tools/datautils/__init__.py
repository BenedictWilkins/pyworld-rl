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

__all__ = ('accumulate', 'function')

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
    r = np.zeros((x.shape[0], size), dtype=dtype)
    r[:, x.squeeze()] = 1
    return r

'''
def onehot(y, size=None):
    if size is None:
        size = len(np.unique(y))
    y = y.reshape(y.shape[0])
    #print(leny, y)
    r = np.zeros((y.shape[0], size))
    r[:, y] = 1.
    return r
'''


def onehot_int(y, size): #deprecated... use onehot
    r = np.zeros((size))
    r[y] = 1
    return r

def mnist(normalise=True):
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, 1).astype(np.float32)
    y_train = np.expand_dims(y_train, 1) 
    x_test = np.expand_dims(x_test, 1).astype(np.float32)
    y_test = np.expand_dims(y_test, 1)
    if normalise:
        return x_train / 255.0, y_train, x_test / 255.0, y_test
    else:
        return x_train, y_train, x_test, y_test

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


def __non_singular(iterator):
    def non_singular_iterator(iterator):
        for x in iterator:
            yield (x,)
            
    x = next(iterator)
    
    singular = not isinstance(x, tuple)
    if singular:
        iterator = non_singular_iterator(iterator)
        x = (x,)
    return iterator, x    

def __init_dataset(iterator, size, template=None):
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
    
def dataset(iterator, size=1000, template=None, progress=0):
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

def __count_wrapper__(iterator, singular=True):
    i = 0 
    if singular:
        for d in iterator:
            i += d.shape[0]
            yield i, d
    else:
        for d in iterator:
            i += d[0].shape[0]
            yield (i, *d)

def batch_iterator(*data, batch_size=64, shuffle=False, count=False, force_nonsingular=False): #, circular=False):
    #refactor this... probably count should be a wrapper
    if shuffle:
        data = __shuffle__(*data)
    singular = (len(data) == 1 and not force_nonsingular)
    iterator = None

    if singular:
        iterator = __batch_iterator_singular__(*data, batch_size=batch_size)
    else:
        iterator = __batch_iterator__(*data, batch_size=batch_size)

    #iterator = __batch_iterator_circular__(*data, batch_size=batch_size)
    
    if count:
        iterator = __count_wrapper__(iterator, singular=singular)
    
    return iterator
    
def __batch_iterator_singular__(data, batch_size):
    m = data.shape[0]
    j = 0
    for i in range(batch_size, m, batch_size):
        yield data[j:i]
        j = i
    yield data[j:]
   
def __batch_iterator__(*data, batch_size):
    m = max(len(d) for d in data)
    j = 0
    for i in range(batch_size, m, batch_size):
        yield tuple([d[j:i] for d in data])
        j = i
    yield tuple([d[j:] for d in data])
    
'''
def __batch_iterator_circular__(*data, batch_size):
    i = 0
    m = max(len(d) for d in data)
    while True:
        indx = np.arange(i,i + batch_size) % m
        i += batch_size
        yield (*[d[indx] for d in data])
'''
        
def shuffle(*data): #sad that this can be done in place...
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


def normalise(data):
    maxd = np.max(data)
    mind = np.min(data)
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
        
   


    

    
    

    
    
    