#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 10:04:12 2019

@author: ben
"""
import time
import traceback
            
def print_stack_trace():
    print("\n".join([line.strip() for line in traceback.format_stack()]))
        
class __assertion:
    
    def __init__(self):
        pass
    
    def __call__(self, condition, error):
        if condition:
            if isinstance(error, str):
                raise ValueError(error)
            raise error
        
assertion = __assertion()

def should_continue():
    yes =  ['y','yes']
    no =  ['n','no']
    print("continue? (y/n)")
    inp = input()
    while inp not in yes + no:
        print("continue? (y/n)")
        inp = input()
    return inp in yes

class Sleep:
    
    t = lambda: int(round(time.time() * 1000))
    
    def __init__(self, wait):
        '''
            Args:
                wait in ms
        '''
        self.wait = wait
       
    def __enter__(self):
        self.start = Sleep.t()
        self.finish = self.start + self.wait
    
    def __exit__(self, type, value, traceback):
        while Sleep.t() < self.finish:
            time.sleep(1./1000.)  
            
class Time:
    
    t = lambda: time.time()
    
    def __init__(self,  message=''):
         self.message = message
         self.start = None
    
    def __enter__(self):
        self.start = Time.t()
    
    def __exit__(self, type, value, traceback):
        time = Time.t() - self.start
        print(self.message, time) 
    