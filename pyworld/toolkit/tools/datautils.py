#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:53:40 2019

@author: Benedict Wilkins
"""
import numpy as np
import itertools

def exit_on(iterator, on):
    for x in iterator:
        yield x
        if on():
            return
        
def repeat(iterator, n, *args, **kwargs):
    for i in range(n):
        for x in iterator(*args, **kwargs):
            yield (i, *x)

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

def collect(model, x):
    j = 0
    result = np.empty((x.shape[0], 2))
    for i, x_batch in batch_iterator(x):
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


def __init_dataset__(iterator, size):
    
    def non_singular_iterator(iterator):
        for x in iterator:
            yield (x,)
            
    x = next(iterator)
    
    singular = not isinstance(x, tuple)
    if singular:
        iterator = non_singular_iterator(iterator)
        x = (x,)
        
    result = tuple([np.empty((size, *__shape__(i)), dtype=__dtype__(i)) for i in x])
    for j in range(len(x)):
        result[j][0] = x[j]
    
    return iterator, result

def dataset(iterator, size=1000):
    iterator, result = __init_dataset__(iterator, size)
    iterator = itertools.islice(iterator, 0, size-1)
    
    for i, x in enumerate(iterator, 1):
        for j in range(len(x)):
            result[j][i] = x[j]
    return result

def dynamic_dataset(iterator, chunk=1, size=1000, random=False):
    iterator, result = __init_dataset__(iterator, size)
    iterator2 = itertools.islice(iterator, 0, size-1)
    
    for i, x in enumerate(iterator2, 1):
        for j in range(len(x)):
            result[j][i] = x[j]
    if not random:
        return __dynamic_dataset__(iterator, result, chunk, size)
    else:
        return __dynamic_dataset_random__(iterator, result, chunk, size)
   
def __dynamic_dataset__(iterator, result, chunk, size):
    for i, x in enumerate(iterator, 0):
        if not i % chunk:
            yield result
        for j in range(len(x)):
            result[j][i % size] = x[j]
            
def __dynamic_dataset_random__(iterator, result, chunk, size):
    for i, x in enumerate(iterator, 0):
        if not i % chunk:
            yield result
        indx = np.random.randint(0, size)
        for j in range(len(x)):
            result[j][indx] = x[j]
            

            
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
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.environments.counter as counter
    
    def test_dataset(): 
        env = counter.Counter()
        policy = gu.uniform_random_policy(env)
        iterator = gu.GymIterator(env, policy, episodic=False, mode=gu.sa)
        
        s,a = dataset(iterator, size=10)
        print(s,a)
    
    def test_dynamic_dataset(random=False): 
        env = counter.Counter()
        policy = gu.uniform_random_policy(env)
        iterator = gu.GymIterator(env, policy, episodic=False, mode=gu.s)
        
        gen = dynamic_dataset(iterator, size=10, random=random)
        d = next(gen)
        print("data", np.transpose(*d))
        for i,s in batch_iterator(*d, batch_size=6, circular=True):
            print(s)
            if i > 20:
                break
            print("data", np.transpose(*next(gen)))
    
    def test_circular():
        x = np.arange(12)
        y = np.arange(12)
        for i, x_, y_ in batch_iterator(x,y, batch_size=8, shuffle=False, circular=True):
            print("x", x_)
            print("y", y_)
            if i > 100:
                break
            
    test_dynamic_dataset(True)
        
   
    

    

    
    

    
    
    