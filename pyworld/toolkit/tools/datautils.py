#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:53:40 2019

@author: Benedict Wilkins
"""
import numpy as np
import itertools

from .debugutils import assertion

def exit_on(iterator, on):
    for x in iterator:
        yield x
        if on():
            return

def invert(index, shape):
    t = np.ones(shape, dtype=np.uint8)
    t[index] = 0
    return np.where(t)[0]
     
def repeat(iterator, n, *args, **kwargs):
    for i in range(n):
        for x in iterator(*args, **kwargs):
            yield (i, *x)
            
def limit(iterator, limit):
    return itertools.islice(iterator, limit)

#??? no idea what this was for..
def display_increments2(total):
    j = 0
    i = 0
    while j < total:
        j = i**2
        yield j
        i += 1        

def onehot(y, size=None):
    leny = size
    if size is None:
        leny = len(np.unique(y))
    y = y.reshape(y.shape[0])
    print(leny, y)
    r = np.zeros((y.shape[0], leny))
    r[:, y] = 1.
    return r

def onehot_int(y, size):
    r = np.zeros((size))
    r[y] = 1
    return r
    
def mnist(normalise=True):
    import tensorflow as tf
    mnist = tf.keras.datasets.mnist
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, 1)
    y_train = np.expand_dims(y_train, 1) 
    x_test = np.expand_dims(x_test, 1) 
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


    
class MeanAccumulator:
    
    def __init__(self, n=float('inf')):
        self._m = 0
        self._n = 0
        self._max_n = n
        
    def push(self, x):
        self._n += 1
        self._m = MeanAccumulator._moving_mean(self._m, x, min(self._n, self._max_n))
        
    def mean(self):
        assert(self._n > 0) # mean of no samples is undefined
        return self._m
    
    def _moving_mean(mean, x, n):
        '''
            Computes the mean in a single pass
            Args:
                mean: current mean
                x: next value
                n: number of values so far (including x)
        '''
        return (x + (n-1) * mean) / n
    
    def reset(self):
        self._m = 0
        self._n = 0
    
    def __str__(self):
        return str(self.mean())
    
    def __repr__(self):
        return MeanAccumulator.__name__ + '-' + str(self._m)

class MeanAccumulator2:
    
     def __init__(self):
        self._m = np.array([])
        self._n = np.array([])
        
     def push(self, x):
        self._m, self._n = MeanAccumulator2._variable_moving_mean(self._m, x, self._n)
        
     def mean(self):
        return self._m
    
     def _variable_moving_mean(mean, x, n):
        lx = len(x)
        lm = len(mean)
        if lx > lm:
            n = np.append(n, [0]*(lx-lm))
        n[:lx] += 1
        mean[:lx] = MeanAccumulator._moving_mean(mean[:lx], x[:lm], n[:min(lx,lm)])
        return np.concatenate((mean, x[lm:])), n    
    
class VarianceAccumulator:
    
    def __init__(self):
        self._m = 0
        self._s = 0
        self._n = 0
    
    def push(self, x):
        self._n += 1
        self._m, self._s = VarianceAccumulator._moving_variance(self._m, self._s, x, self._n)
    
    def mean(self):
        assert(self._n > 0) # mean of no samples is undefined
        return self._m
    
    def variance(self):
        assert(self._n > 1) # variance of a single sample is undefined
        return self._s / self._n
    
    def sample_variance(self):
        assert(self._n > 1) # variance of a single sample is undefined
        return self._s / (self._n - 1)    
    
    def _moving_variance(M, S, x, n):
        '''
            Computes the variance in a single pass
            Args:
                M: mean
                S: -
                x: next value
                n: number of values so far (including x)
        '''
        Mn = M + (x - M)/n 
        S = S + (x-M)*(x-Mn)
        return Mn, S 

     
class VarianceAccumulator2:
    
    def __init__(self):
        self._m = np.array([])
        self._s = np.array([])
        self._n = np.array([])
        self._nn = 0
    
    def push(self, x):
        self._m, self._s, self._n, self._nn = VarianceAccumulator2._variable_moving_variance(self._m, self._s, x, self._n)
        
    def mean(self):
        return self._m
    
    def variance(self):
        return self._s[:self._nn] / self._n[:self._nn]
    
    def sample_variance(self):
        return self._s[:self._nn] / (self._n[:self._nn] - 1)
        
    def standard_deviation(self):
        return np.sqrt(self.variance())
    
    def _variable_moving_variance(M,S,x,n):
        '''
            Computes the variance in a single pass for variable size x
            M: mean
            S: -
            x: next value (of varying size)
            n: number of values so far (including x)
        '''
        lx = len(x)
        lm = len(M)
        if lx > lm:
            a = [0]*(lx-lm)
            n = np.append(n, a)
        n[:lx] += 1
        M[:lx], S[:lx] = VarianceAccumulator._moving_variance(M[:lx], S[:lx], x[:lm], n[:min(lx,lm)])
        return np.concatenate((M, x[lm:])), np.concatenate((S, np.zeros(len(x[lm:])))), n, len(M)
        
    
    
class CMA:
    
    def __init__(self, *labels):
        self.labels = labels
        self.reset()
    
    def push(self, x):
        self._n += 1
        self._m = (x + (self._n-1) * self._m) / self._n
        
    def __call__(self):
        assert(self._n > 0) # mean of no samples is undefined
        return self._m
    
    def reset(self):
        if len(self.labels) > 0:
            self._m = np.zeros(len(self.labels))
        else:
            self._m = 0
        self._n = 0
        
    def labelled(self):
        return {self.labels[i]:self._m[i] for i in range(len(self.labels))}
    
    def __str__(self):
        if len(self.labels) > 0:
            return str(self.labelled())
        else:        
            return str(self._m)
    
    def __repr__(self):
        return CMA.__name__ + '-' + str(self)
    
class EMA:
    
    def __init__(self, n):
        self._alpha = 2. / (n + 1.)
        self.reset()
        
    def __push1(self, x):
        self._m = x
        self.push = self.__push

    def __push(self, x):
        self._m = self._alpha * x + (1-self._alpha) * self._m
    
    def __call__(self):
        assert(self._m is not None) # mean of no samples is undefined
        return self._m
    
    def reset(self):
        self._m = 0
        self.push = self.__push1
    
    def __str__(self):
        return str(self._m)
    
    def __repr__(self):
        return EMA.__name__ + '-' + str(self._m)


        
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
            assertion(not tuple(t.shape[1:]) == tuple(x[i].shape), 
                      'template shape {0} and example shape {1} do not match at position {2}'.format(t.shape[1:], x[i].shape, i))
        result = template
        
    for j in range(len(x)):
        result[j][0] = x[j]
    
    return iterator, result
    
def dataset(iterator, size=1000, template=None):
    iterator, result = __init_dataset(iterator, size, template)
    iterator = itertools.islice(iterator, 0, size-1)
        
    for i, x in enumerate(iterator, 1):
        for j in range(len(x)):
            result[j][i] = x[j]
    return result

#refactor at some point...    
def no_count(batch_iterator):
    for b in batch_iterator:
        yield b[1:]

def batch_iterator(*data, batch_size=16, shuffle=False, circular=False):
    if shuffle:
        data = __shuffle__(*data)
    if not circular:
        return __batch_iterator__(*data, batch_size=batch_size)
    else:
        return __batch_iterator_circular__(*data, batch_size=batch_size)

def __batch_iterator_circular__(*data, batch_size):
    i = 0
    m = max(len(d) for d in data)
    while True:
        indx = np.arange(i,i + batch_size) % m
        i += batch_size
        yield (i, *[d[indx] for d in data])
        
def __batch_iterator__(*data, batch_size):
    m = max(len(d) for d in data)
    j = 0
    for i in range(batch_size, m, batch_size):
        yield (i, *[d[j:i] for d in data])
        j = i
    yield (m, *[d[j:] for d in data]) #01/10/2019 swap argument order! to match enumerate
        
def __shuffle__(*data):
    m = max(len(d) for d in data)
    indx = np.arange(m)
    np.random.shuffle(indx)
    data = [d[indx] for d in data]
    return data

def collect(data, fun, unpack=True, batch_size=128):
    iterator = no_count(batch_iterator(data, batch_size=batch_size))
    z = pack_apply(iterator, fun, unpack)[0]
    return np.concatenate(z)

def apply(iterable, fun, unpack=True):
    if unpack:
        for i in iterable:
            yield fun(*i)
    else:
        for i in iterable:
            yield fun(i)
            
def pack_apply(iterable, fun, unpack=True):
    iterable = apply(iterable, fun, unpack)
    return pack(iterable)

def pack(iterator):
    '''
        Packs the content of an iterator into numpy array(s)
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
        
   


    

    
    

    
    
    