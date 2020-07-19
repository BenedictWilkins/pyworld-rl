import numpy as np

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
    """
        Cumulative moving average.
    """
    
    def __init__(self, *labels):
        self.labels = labels
        self.reset()
    
    def push(self, *x):
        x = np.array(x)
        self._x = x
        self._n += 1
        self._m = (x + (self._n-1) * self._m) / self._n
        
    def __call__(self):
        assert(self._n > 0) # mean of no samples is undefined
        return self._m
    
    def recent(self):
        if len(self.labels) > 0:
            return {self.labels[i]:self._x[i] for i in range(len(self.labels))} #get the most recent values that were pushed
        else:
            return self._x
    
    def reset(self):
        self._m = 0
        self._n = 0
        self._x = None
        
    def labelled(self):
        assert(self._n > 0)
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
