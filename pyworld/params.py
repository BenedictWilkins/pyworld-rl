#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 17:01:05 2019

@author: ben
"""

GAMMA = 0.99
DEVICE = 'cpu'
FRAME_STACK = 4
FRAME_SKIP = 3
UNROLL_STEPS = 3
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
ENTROPY_BETA = 0.01
CLIP_GRAD = 0 # dont use it...

def gamma(params):
    return params.get('gamma', GAMMA)

def device(params):
    return params.get('device', DEVICE)

def frame_stack(params):
    return params.get('frame_stack', FRAME_STACK)

def frame_skip(params):
    return params.get('frame_skip', FRAME_SKIP)

def unroll_steps(params):
    return params.get('unroll_steps', UNROLL_STEPS)

def batch_size(params):
    return params.get('batch_size', BATCH_SIZE)

def learning_rate(params):
    return params.get('learning_rate', LEARNING_RATE)

def entropy_beta(params):
    return params.get('entropy_beta', ENTROPY_BETA)

def clip_grad(params):
    return params.get('clip_grad', CLIP_GRAD)