#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:46:19 2019

author: Benedict Wilkins
"""

if __name__ == "__main__":
    import pyworld.toolkit.tools.gymutils as gu
    import pyworld.toolkit.tools.visutils as vu
    import numpy as np


    env = gu.env(binary=0.45, stack=3)
    
    #iterator = gu.iterators.s_iterator(env, gu.policy.uniform_random_policy(env))
    vu.play(gu.video(env))

        