#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 14:41:39 2018

@author: ben
"""

import math

def l22(a,b):
    r = 0
    for z in zip(a,b):
        r += (z[0] - z[1])**2
    return r

def l2(a,b):
    return math.sqrt(l22(a,b))
    

class ValueIteration:
    
    def __init__(self, rfun, states, actions, transitions):
        self.rfun = rfun
        self.states = states
        self.actions = actions
        self.transitions = transitions
        self.V = {k:(0,None) for k in states}
        
    def greedy_policy(self, s):
        return self.V[s]
        
            
    def synch_value_iterator(self, g = 0.9, e = 10e-2, maxiterations=100):
        eb = e * (1 - g) / g
        i = 0
        err = float('inf')
        while i < maxiterations and err > eb:
            Vn = {}
            i += 1
            for s in self.states: 
                m = float('-inf')
                for a in self.actions:
                    mn = self.rfun(s, a) + g * self.expected_future_reward(s, a)
                    if(mn > m): 
                        m = mn
                        Vn[s] = (m, a)
            err = l2([v[0] for v in self.V.values()], [v[0] for v in Vn.values()])
            print("Next: ",i," - ", err)
            self.V = Vn

    def asynch_value_iterator(self, g = 0.9, maxiterations=100):
        i = 0
        while i < maxiterations:
            i += 1
            for s in self.states: 
                m = float('-inf')
                mn = 0.
                for a in self.actions:
                    mn = self.rfun(s, a) + g * self.expected_future_reward(s,a)
                    if(mn > m): 
                        m = mn 
                        self.V[s] = (m, a)
            print("Next: ",i)

    def expected_future_reward(self, s, a):
        efr = 0
        for sp, t in self.transitions[a][s].items():
            efr += t * self.V[sp][0]
        return efr
    
    def __str__(self):
        return "ValueIteration:\n" + "\n".join(str(v[0]) + " -> " + str(v[1]) for v in self.V.items())
    
    def print_transitions(self):
        for a in self.transitions:
            print(a)
            for s in self.transitions[a]:
                print("    ", s, "->", self.transitions[a][s])
    
