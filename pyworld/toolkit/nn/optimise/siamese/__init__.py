#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:26:27 2019

@author: ben
"""

import torch
import torch.nn
import torch.nn.functional as F
import numpy as np

from collections import namedtuple

import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu

from pyworld.toolkit.tools.datautils.accumulate import CMA #TODO remove

from ..base import TorchOptimiser
    
#TODO most of the self.loss computation of each optimiser should be moved into self.step

mode = namedtuple('mode', 'all top top_n, top_p')(0,1,2,3)  #enum?

class TripletOptimiser(TorchOptimiser):
         
    def __init__(self, model, margin = 0.2, mode = mode.all, k = 16, lr=0.0005):
        super(TripletOptimiser, self).__init__(model)
        self.mode = mode
        self.__top = [(False, False), (True, True), (True, False), (False, True)] #topk_n, topk_p
        self.k = int(k) #should be related to the batch size or number of p/n examples expected
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.cma = CMA('loss')
        self.margin = margin
    
    def step(self, x, y):
        #self.optim.zero_grad()
        loss = self.loss(x, y.squeeze(), *self.__top[self.mode])
        self.cma.push(loss.item())
        #loss.backward()
        #self.optim.step()
        return loss
        
    def loss(self, x, y, topk_n = False, topk_p = False):
        x_ = self.model(x)
        #d = self.distance_matrix(x_)
        unique = np.unique(y)
        loss = torch.FloatTensor(np.array([0.])).to(self.model.device)

        for u in unique:
            pi = np.nonzero(y == u)[0]
            ni = np.nonzero(y != u)[0]
            
            #xp_t = d[pi][:,pi]
            #xn_t = d[pi][:,ni]
            #slightly more efficient below
            xp_ = x_[pi]
            xn_ = x_[ni]
            xp = self.distance_matrix(xp_, xp_)
            xn = self.distance_matrix(xp_, xn_)

            if topk_p:
                xp = self.topk2(xp, self.k, large=True)
            if topk_n:
                xn = self.topk2(xn, self.k, large=False)
                
            #3D tensor, (a - p) - (a - n) 
            
            xf = xp.unsqueeze(2) - xn
            xf = F.relu(xf + self.margin) #triplet loss
            loss += xf.sum()

        return loss
        
    def distance_matrix(self, x1, x2=None): #L22 distance by default
        # TODO speed up...
        if x2 is None:
            x2 = x1
        n_dif = x1.unsqueeze(1) - x2.unsqueeze(0)
        return torch.sum(n_dif * n_dif, -1)
    
    ''' #speed up??!
    def dmatrix(x1,x2=None):
        if x2 is None:
            x2 = x1
        dists = -2 * np.dot(x1, x2.T) + np.sum(x1**2, axis=1) + np.sum(x2**2, axis=1)[:, np.newaxis]
        return dists
    '''
    
    def topk(self, x, k, large=False):
        # if we want the top k in the whole matrix, this makes later computations a bit tricky...
        # use topk2
        indx = torch.topk(x.view(-1), k, largest=large)[1]
        return indx / x.shape[1], indx % x.shape[1]

    def topk2(self, x, k, large=False):
        if k >= x.shape[1]:
            return x
        else:
            return torch.topk(x, k, dim=1, largest=large)[0]

class PairTripletOptimiser(TripletOptimiser):
    
    def __init__(self, model, margin = 0.2, mode = mode.all, k = 16, lr=0.0005):
        super(PairTripletOptimiser, self).__init__(model, margin, mode, k, lr)
        
    def step(self, x1, x2):
        '''
            Step with pairs of input. ``(x1_i, x2_i)`` are a pair - share the same label
            (x1_i, x2_{j\neq i}) are considered to have different labels (i.e. are not a pair).
        '''
        #self.optim.zero_grad()
        loss = self.loss(x1, x2, *self._TripletOptimiser__top[self.mode])
        self.cma.push(loss.item())
        #loss.backward()
        #self.optim.step()
        return loss
    
    def loss(self, x1, x2, topk_n = False, topk_p = False):
        
        x1_ = self.model(x1)
        x2_ = self.model(x2)
        
        d = self.distance_matrix(x1_, x2_)
        xp = torch.diag(d).unsqueeze(1)
        xn = d # careful with the diagonal?

        #if topk_n and self.k < xn.shape[0]:
        #    #remove xp - xn = 0, along  
        #    xn[range(d.shape[0]), range(d.shape[1])] = float('inf') #hopefully this doesnt mess up autograd...
        #    xn = self.topk2(d, self.k, large=False) #select the k best negative values for each anchor [batch_size x k]

        # is doesnt matter if xp is included in xn as xp_i - xn_i = 0, the original inequality is satisfied, the loss will be 0.
        # it may be more expensive to remove these values than to just compute as below.
        xf = xp.unsqueeze(2) - xn #should only consist of only ||A - P|| - ||A - N|| [batch_size x batch_size x k]
        
        #else: this is probably not needed...?
            #xf[:,range(d.shape[0]), range(d.shape[1])] = 0. #remove all ||A-P|| - ||A-P|| #todo
            
        xf = F.relu(xf + self.margin)
        return xf.sum()

class SASTripletOptimiser(TripletOptimiser):

    # monsterous... 
    # TODO refactor to make use of the methods in __SASModel for consistency...
    
    class __SAModel(torch.nn.Module):
        
        def __init__(self, s_model, a_model):
            super(SASTripletOptimiser._SASTripletOptimiser__SAModel, self).__init__()
            assert len(tu.as_shape(s_model.output_shape)) == 1
            assert len(tu.as_shape(a_model.output_shape)) == 1
            action_shape = tu.as_shape(a_model.input_shape)[0] - (2 * tu.as_shape(s_model.output_shape)[0])

            if action_shape < 1:
                raise ValueError('the input shape of the action model should be 2 x output_shape of the state model + the size of the action space, i.e. atleast > {0}'.format(2 * s_model.output_shape))
            
            self.state = s_model
            self.action = a_model

            #used to generate combinations S x S x A by the optimiser
            self.action_space = torch.as_tensor(np.identity(action_shape, dtype=np.float32), device=self.device)
        
        def forward(self, s1, a, s2, **kwargs):
            assert len(a.shape) > 1 and a.shape[1] > 1 # actions should be in 1-hot format
  
            # | x1 | x2 | a |.is used by the optimiser!
            x1_ = self.state(s1) #N x D
            x2_ = self.state(s2) #N x D
            a = a.to(self.device)

            x = torch.cat((x1_, x2_, a), 1) #N x 2D + A 
            z = self.action(x) # N x D?
            return z

        def distance(self, sas, pnorm=2, **kwargs): #this should be the same as in the optimiser...
            s1,a,s2 = sas
            z = self.forward(s1, a, s2, **kwargs)
            assert pnorm == 2 ##... look below **2
            return torch.norm(z, p=pnorm, dim=1) ** 2 #default L22 norm...

        @property
        def device(self):
            assert self.state.device == self.action.device
            return self.state.device
        
        def to(self, device):
            r = super(SASTripletOptimiser._SASTripletOptimiser__SAModel, self).to(device)
            self.action_space.to(self.device)
            return r

    def SASModel(s_model, a_model):
        return SASTripletOptimiser._SASTripletOptimiser__SAModel(s_model, a_model)
            
    def __init__(self, s_model, a_model, margin = 0.2, mode = mode.all, k = 16, lr=0.0005, pnorm=2):
        super(SASTripletOptimiser, self).__init__(SASTripletOptimiser.SASModel(s_model, a_model), margin, mode, k, lr)
        self.pnorm = 2
        
    def step(self, s1, a, s2):
        '''
           (s1_i, a_i, s2_i) are considered positive any other combination are negative 
           where a_i is taken from the set of possible actions. 
            Arguments:
                s1: a batch of images (N x H x W x C) float32
                a: a batch of actions (N x A) uint8
                s2: a batch of images (N x H x W x C) float32
        '''
        #self.optim.zero_grad()
        loss = self.loss(s1, a, s2, *self._TripletOptimiser__top[self.mode])
        self.cma.push(loss.item())
        #loss.backward()
        #self.optim.step()
        return loss
    
    def loss(self, s1, a, s2, topk_n = False, topk_p = False):
        #encode each state, of course this can be done more efficiently in the specific case!
        if a.dtype != np.int64:
            raise TypeError("a must be an N x 1 array of type int64 -- due to some weird pytorch indexing behaviour... (^_^)")

        x1_ = self.model.state(s1)
        x2_ = self.model.state(s2)

        #print(s1.shape, s2.shape, a.shape)
        #print(x1_.shape, x2_.shape)
        
        # batch_size x batch_size x |action_space|. distance matrix computed using the action model
        # the positive examples are those indexed by [[0-batch_size-1], [0-batch_size-1], a]
        d = self.distance_matrix(x1_, x2_) # N x N x |A|
        #print(d.shape)
        #print(d.shape, a.dtype)

        indx = (range(d.shape[0]), range(d.shape[1]), a.squeeze())  #only works if a is int64 type (dunno why...)

        xp = d[indx].unsqueeze(1) # batch_size x 1, ||A - P|| examples

        xn = d.reshape(d.shape[0], -1) # all others are negative, batch_size x (batch_size x actions)
        
        '''
        if topk_n and self.k < xn.shape[0]:
            #print("topk_n?")
            xn[range(d.shape[0]), range(d.shape[1])] = float('inf') #hopefully this doesnt mess up autograd... we are working with views... this is a bit harder to do?
            xn = self.topk2(d, self.k, large=False) #select the k best negative values for each anchor [batch_size x k]
            #print(xn.shape)
        '''
        
        # ||A-P|| - ||A-P|| will be 0 so wont contribute to the loss
        
        xf = xp.unsqueeze(2) - xn #should only consist of only ||A-P|| - ||A-N|| [batch_size x batch_size x k]
        #print(xf.shape)
        #else: this is probably not needed...?
            #xf = xp.unsqueeze(2) - xn
            #xf[:,range(d.shape[0]), range(d.shape[1])] = 0. #remove all ||A-P|| - ||A-P|| #todo
            
        xf = F.relu(xf + self.margin) 
        
        return xf.mean() #??sum??? ...
    
    def distance_matrix(self, x1, x2):
        ''' 
            Creates a similarity tensor of dimension N x N x |A| from two batches of states 
            each of dimension N x D. The similarity tensor is constructed from all possible 
            combinations of x1, x2, actions and the similarities are computed using the action 
            model. If the action model gives a scalar output, this value is used as the similarity,
            otherwise a p-norm is taken over the output vectors.
            
            Note that the action model must have input dimension D + D + |A| as the 
            elements of x1, x2 and a 1-hot vector representing the action are concatinated to 
            form the input. 
            
            Arguments:
                x1 : batch of states N x D
                x2 : batch of states N x D
            
            Returns:
                A similarity tensor of dimension N x N x |A|
        '''
        # create a HUGE tensor from a batch of encoded state pairs x1, x2
        # this will allow us to construct the N x N x |A| distance matrix
        x = self.pre_alt2(x1, x2) # N x N x |A| x D_in
            
        x_shape = x.shape
        x = x.view(-1, x.shape[-1]) # form a batch (N x N x |A|) x D_in, hopefully there is enough memory!
        
        #run through the action model
        z = self.model.action(x) # (N x N x |A|) x D_out
        
        if z.shape[1] > 1:
            #norm over D_out if vector output
            assert self.pnorm == 2 # look below ** 2
            z = torch.norm(z, p=self.pnorm, dim=1) ** 2 #default L2 norm...
        
        z = z.view(*x_shape[:-1]) # (N x N x |A|) x 1 to shape N x N x |A|
        
        return z # N x N x |A| similarity tensor
        
    def pre_alt1(self, x1, x2):
            raise NotImplementedError("use pre_alt2, TODO implement this if memory errors everywhere!")
            #alternative 1 (no copy) loop
            for action in self.action_space: #I could not think of a vectorised form that didnt involve doing a huge copy with torch.cat
                a_ = action.expand(-1, *action.shape) #TODO fix
                x = torch.cat((x1, x2, a_), 1)
                #blah blah    
        
    def pre_alt2(self, x1, x2):
        '''
            Constructs a tensor from the two batches of state encodings each of dimension N x D.
            The constructed tensor contains all possible combinations of (x1, x2, actions)
            the tensor has dimension N x N x |A| x M where |A| is the cardinality of the 
            action space (which is assumed to be discrete) and M = D + D + |A|.
            Each M vector is concatinated: | x1 | x2 | a |.
            
            Arguments:
                x1 : batch of states N x D
                x2 : batch of states N x D
            
            Returns:
                A tensor containing all possible combinations of x1, x2, actions 
                (concatinated along the final dimension)
        '''

        #alternative 2 copy with cat but no loop...
        x1 = x1.unsqueeze(0).expand(x1.shape[0], -1, -1)
        x2 = x2.unsqueeze(1).expand(-1, x2.shape[0], -1)
        x = torch.cat((x1,x2), 2) # all possible pairs of state encodings
        
        #x1 and x2 should have the same shape...
        a = self.model.action_space.unsqueeze(0).unsqueeze(0).expand(x1.shape[0], x2.shape[0], -1, -1)
        x = x.unsqueeze(2).expand(-1, -1, a.shape[-1], -1) 

        #all possible combinations of state-state-action given the action space we are working with
        x = torch.cat((x,a), 3) #SIGHHHH copying is the bane of my existence

        return x # N x N x |A| x M

if __name__ == "__main__":
    import pyworld.toolkit.tools.torchutils as tu
    from pyworld.toolkit.nn.MLP import MLP
    
    state_shape = 3
    action_shape = 4
    batch_size = 2
    
    state_out_shape = 3
    action_out_shape = 2
    
    action_model = MLP(state_out_shape * 2 + action_shape, action_out_shape)
    state_model = MLP(state_shape, state_out_shape)
    
    x1 = torch.as_tensor(np.random.uniform(size=(batch_size,state_shape)).astype(np.float32))
    x2 = torch.as_tensor(np.random.uniform(size=(batch_size, state_shape)).astype(np.float32))
    a = torch.as_tensor(np.random.randint(0, action_shape, size=batch_size))
    
    print(x1)
    print(x2)
    print(a)
    
    tro = SASTripletOptimiser(state_model, action_model)
    tro.step(x1, a, x2)
    
    #action_model(x1, x2)



    #print(x1.expand(10, *x1.shape)) #NICE!
    #print(torch.cat((x1, x2, a), 1))
    
    
    #tro = PairTripletOptimiser(model, k=1, mode=mode.top)
    #tro.step(x1, x2)
    

