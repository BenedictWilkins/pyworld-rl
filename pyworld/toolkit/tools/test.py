#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:32:54 2019

@author: ben
"""

if __name__ == "__main__":
    import pyworld.environments.objectmover as om
    import pyworld.toolkit.tools.visutils as vu
    import pyworld.toolkit.tools.gymutils as gu
    
    env = om.default()
    policy =  gu.uniform_random_policy(env)
    giter = gu.GymIterator(env, policy)
    
    data = gu.dynamic_dataset(env, policy, mode = gu.sa, size=1, onehot=True)
    print(data)

    

    
