#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:22:57 2019

@author: Benedict Wilkins
"""

try:
    
    from . import MLP
    from . import CNet
    from . import autoencoder
    
    __all__ = ('MLP', 'CNet', "autoencoder")
    
except Exception as e:
    print("WARNING: pyworld.nn is unavailable as a dependancy was not found.")
    print(str(e))
    