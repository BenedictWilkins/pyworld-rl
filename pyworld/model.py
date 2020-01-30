#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 10:56:23 2019

@author: ben
"""


        
''' ************************************ MODEL ************************************ '''
from abc import ABC, abstractmethod
import torch
import copy
from . import common as pwcom

class Model(ABC):
    
    def __init__(self):
        self.train = True
    
    @abstractmethod
    def step(self, batch, *args, **kwargs):
        pass
    
    @abstractmethod
    def train(self, train=True):
        pass
    
class NNModel(Model):
    
    def __init__(self, net, **params):
        print("MODEL INFO:")
        self.device = params.get('device', 'cpu')
        self.net = net.to(self.device)
        self.optim = params.get('optim', None)
        if self.optim is None:
             self.optim = torch.optim.Adam(net.parameters(), lr=params.get('lr', 0.0001))
        print("-- USING OPTIMIZER: " + type(self.optim).__name__)
        print("-- USING NETWORK: " + type(self.net).__name__)
    
    def train(self, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()
        
        
class DQNNModel(Model):
    
    def __init__(self, net, **params):
        super(DQNNModel, self).__init__()
        print("MODEL INFO:")
        self.device = params.get('device', 'cpu')
        print("-- USING DEVICE: " + self.device)
        self.net = net.to(self.device)
        print("-- USING NETWORK: " + type(self.net).__name__)
        self.optim = params.get('optim', None)
        if self.optim is None:
             self.optim = torch.optim.Adam(net.parameters(), lr=params.get('lr', 0.001))
        print("-- USING OPTIMIZER: " + type(self.optim).__name__)
        self.gamma = params.get('gamma', 0.99)
        self.step = self._step
        if params.get('grad_clip', False):
            self.gradient_clip = params['grad_clip']
        if params.get('target', False):
            print("-- USING TARGET NETWORK")
            self.step = self._step_with_target
            if params.get('sync', False):
                self.sync = params['sync']
                self.sync_params = params.get('sync_params', None)
            else:
                self.sync = simple_sync
                self.sync_params = ()
            self.sync_at = params.get('sync_at', 1000)
            self.target_net = copy.deepcopy(net) #.to(self.device) ... keep on cpu? no optimization is done on the target network
        if params.get('loss', False):
            self.loss = params.get('loss') 
        print("-- USING LOSS: ", self.loss.__name__)
        
    def train(self, train=True):
        if train:
            self.net.train()
        else:
            self.net.eval()

def simple_sync(net, tgt_net, *_):
        """
            update the parameters of the target network with the real network.
        """
        tgt_net.load_state_dict(net.state_dict())

def alpha_sync(net, target_net, *args):
        """
        Blend params of target net with params from the model
        """
        alpha, = args
        state = net.state_dict()
        tgt_state = target_net.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        target_net.load_state_dict(tgt_state)

def batch_to_tensors(batch, batch_types_numpy, batch_types_torch, device='cpu'):
        batch = pwcom.batch_to_numpy(batch,  batch_types_numpy)
        batch = pwcom.batch_to_tensor(batch, batch_types_torch, device=device)
        return batch


