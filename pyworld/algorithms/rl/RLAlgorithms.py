#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 10:56:31 2018

@author: ben
"""

from collections import defaultdict
from abc import abstractmethod
import random
import operator

def default_step(t):
    return 1./t


class TD:
    
    def __init__(self, policy, initial_state=None, initial_reward=None, step=default_step, discount=0.99):
        self.policy = policy
        self.ut = {} #utility estimates
        self.fr = defaultdict(lambda: 0) #frequencies of each state
        self.step = step
        self.discount = discount
        self.s = initial_state
        self.r = initial_reward
        self.t = 0
    
    def update(self, reward:float, state):
        self.t += 1
        if state not in self.ut.keys():
            self.ut[state] = reward
        if self.s is not None:
            self.fr[state] += 1
            uts = self.ut[self.s]
            self.ut[self.s] = uts + self.step(self.t) * (self.r + self.discount * self.ut[state] - uts)
        self.s = state
        self.r = reward
        return self.policy.evaluate(state, self.ut, self.fr)
    
    
class QLearning:
    
    def __init__(self, policy, initial_state=None, initial_reward=None, step=default_step, discount=0.99):
        self.policy = policy
        self.q = defaultdict(lambda : {k:0 for k in policy.possible_actions}) #utility estimates
        self.fr = defaultdict(lambda : {k:0 for k in policy.possible_actions}) #frequencies of each state
        self.step = step
        self.discount = discount
        self.s = initial_state
        self.r = initial_reward
        self.t = 0

    def update(self, reward:float, state):
        self.t += 1
        if self.s is not None:
            self.fr[self.s][self.a] += 1
            qs = self.q[self.s][self.a]
            self.q[self.s][self.a] = qs + self.step(self.fr[self.s][self.a]) * (self.r + self.discount * max(self.q[state].values()) - qs)
        self.s = state
        self.r = reward
        self.a = self.policy.evaluate(state, self.q[state], self.fr[state])
        return self.a
    
    def evaluate(self, state):
        return self.policy.evaluate(state, self.q[state], self.fr[state])
        
class Policy:
    
    def __init__(self, possible_actions):
        self.possible_actions = possible_actions
    
    @abstractmethod
    def evaluate(self, state, u, fr):
        pass        

class GreedyPolicy(Policy):
    
    def __init__(self, possible_actions):
        super().__init__(possible_actions)

    def evaluate(self, s, qs, frs):
        if not len(qs) == 0:
            return max(qs, key=qs.get)
        else:
            return random.choice(self.possible_actions)
   
class EGreedyPolicy(Policy):

    def __init__(self, possible_actions, e=0.01):
        super().__init__(possible_actions)
        self.e = e

    def evaluate(self, s, qs, frs):
        if not len(qs) == 0 and random.random() > self.e:
            return max(qs, key=qs.get)
        else:
            return random.choice(self.possible_actions)
        
class URandomPolicy(Policy):
    
    def __init__(self, possible_actions):
        super().__init__(possible_actions)
    
    def evaluate(self, state, ut, fr):
        return random.choice(self.possible_actions)
    
    