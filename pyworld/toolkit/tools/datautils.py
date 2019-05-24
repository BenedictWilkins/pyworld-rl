#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:53:40 2019

@author: ben
"""
import numpy as np

def batch_to_numpy(batch, types, copy=False):
    return [np.array(batch[i], copy=copy, dtype=types[i]) for i in range(len(batch))]

def batch_to_tensor(batch, types, device='cpu'):
        return [types[i](batch[i]).to(device) for i in range(len(batch))]
    
def batch_iterator(*data, batch_size=16, shuffle=False):
    if shuffle:
        for d in data:   
            np.random.shuffle(d)
    j = 0
    m = max(len(d) for d in data)
    for i in range(batch_size, m, batch_size):
        yield [d[j:i] for d in data]
        j = i
    yield [d[j:] for d in data]

class MeanAccumulator:
    
    def __init__(self):
        self._m = 0
        self._n = 0
        
    def push(self, x):
        self._n += 1
        self._m = MeanAccumulator._moving_mean(self._m, x, self._n)
        
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
        
   
  
if __name__ == "__main__":
    
    ma = MeanAccumulator()
    x1 = 0
    ma.push(x1)
    print(ma.mean())
    x2 = 1
    ma.push(x2)
    print(ma.mean())
    x3 = 0
    ma.push(x3)
    print(ma.mean())
    
    ma = MeanAccumulator()
    x1 = np.array([0,0])
    ma.push(x1)
    print(ma.mean())
    x2 = np.array([1,1])
    ma.push(x2)
    print(ma.mean())
    x3 = np.array([1,0])
    ma.push(x3)
    print(ma.mean())

    print("MA2")
    ma = MeanAccumulator2()
    x1 = np.array([0,0,1])
    ma.push(x1)
    print(ma.mean())
    x2 = np.array([1])
    ma.push(x2)
    print(ma.mean())
    x3 = np.array([1,1])
    ma.push(x3)
    print(ma.mean())
    
    print("VA2")
    va = VarianceAccumulator2()
    x1 = np.array([0.])
    x2 = np.array([0.,1.])
    x3 = np.array([1.,1.,2.])
    x4 = np.array([1.])
    
    va.push(x1)
    print(va.variance(), va.mean())
    va.push(x2)
    print(va.variance(), va.mean())
    va.push(x3)
    print(va.variance(), va.mean())
    va.push(x4)
    print(va.variance(), va.mean())
    
    print("VA")
    va = VarianceAccumulator()
    x1 = 1
    va.push(x1)
    print(va.mean())
    x2 = 1
    va.push(x2)
    print(va.variance(), va.mean())
    x3 = 0
    va.push(x3)
    print(va.variance(), va.mean())
    
    

    
    
    