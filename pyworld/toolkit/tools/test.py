#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 11:29:42 2019

author: Benedict Wilkins
"""
import numpy as np
import torch

import pyworld.toolkit.tools.datautils as du
import pyworld.toolkit.tools.torchutils as tu
import pyworld.toolkit.tools.gymutils as gu
import pyworld.environments

env = gu.make('ObjectMover-v1')

class 

policy = gu.policy.uniform_random_policy(env.action_space)

input_shape = env.observation_space.shape
output_shape = 2

#data = gu.episodes(env, policy, mode=gu.mode.ss, epochs=config['epochs']) #- doesnt work so well....? hmm
data = gu.datasets(env, policy, mode=gu.mode.ss, size=10, epochs=10)