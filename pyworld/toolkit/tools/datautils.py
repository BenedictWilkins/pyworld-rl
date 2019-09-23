#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:53:40 2019

@author: Benedict Wilkins
"""
import numpy as np

def display_increments2(total):
    j = 0
    i = 0
    while j < total:
        j = i**2
        yield j
        i += 1        

def onehot(y):
    y = y.squeeze()
    r = np.zeros((y.shape[0], len(np.unique(y))))
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

def collect(model, x):
    j = 0
    result = np.empty((x.shape[0], 2))
    for x_batch, i in batch_iterator(x):
        result[j:i] = model(x_batch)
        j = i
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

def apply(iterator, fun):
    for i in iterator:
        yield fun(i)
 
def batch_iterator(*data, batch_size=16, shuffle=False):
    m = max(len(d) for d in data)
    if shuffle:
        indx = np.arange(m)
        np.random.shuffle(indx)
        data = [d[indx] for d in data]
    j = 0
    for i in range(batch_size, m, batch_size):
        yield (*[d[j:i] for d in data], i)
        j = i
    yield (*[d[j:] for d in data], m)
    
def batch_iterator2(iterator, batch_size, limit=np.inf, btype=list):
    limit = limit * batch_size
    i = 0 
    batch = []
    for b in iterator:
        i += 1
        batch.append(b)
        if i >= limit:
            break
        if not i % batch_size:
            if len(batch) > 0:
                yield (btype(batch), i)
            batch = []
    if len(batch) > 0:
        yield (btype(batch) , i)

def batch_iterator3(*data, batch_size=16 , p=None, iterations=None):
    assert p is None #not implemented yet
    m = max(len(d) for d in data)
    if iterations is None:
        iterations = int(m/batch_size)
    for i in range(iterations):
        indx = np.random.randint(0,m,batch_size)
        yield (*[d[indx] for d in data], i)

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

if __name__ == "__main__":
    x = np.arange(64)
    y = np.arange(64)
    for x_, y_, i in batch_iterator(x,y, shuffle=True):
        print(x_)
        print(y_)
        print()
        
   
    

    

    
    

    
    
    