#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 15-09-2020 12:46:56

    [Description]
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np

def __is_channels__(axes):
    return axes == 1 or axes == 3 or axes == 4

def image_format(shape): # guess the shape, defaults to HWC
    if len(shape) == 3:
        if __is_channels__(shape[-1]):
            return "HWC"
        elif __is_channels__(shape[0]):
            return "CHW"
    elif len(shape) == 2:
        return "HW"
    else:
        raise ValueError("Invalid image format: {0}".format(shape))
# ???
#def format(image, format_old, format_new):
#    old = np.array([format_old.index(i) for i in "CHW"])
#    new = np.array([format_new.index(i) for i in "CHW"])
#    swap = new[old]
#    return image.transpose(*swap)

def nform(image):
    # assumes NHWC, HWC, NHW or HW
    # guess the image format and normalise it (NHWC)
    if len(image.shape) == 2: #HW format
        return image[np.newaxis,:,:,np.newaxis], lambda x: x[0]
    elif len(image.shape) == 3: #HWC or NHW
        if image.shape[-1] in [1,3,4]:
            return image[np.newaxis,...], lambda x: x[0] #HWC ?
        else:
            return image[...,np.newaxis], lambda x: x #NHW ?
    elif len(image.shape) == 4:
        return image, lambda x: x
    else:
        raise ValueError("Unknown image format: {0}".format(image.shape))

def isCHW(image): # weak check
    '''
        Is the given image in CHW or NCHW.
        Arguments:
            image: to check
    '''
    C_index = 4 - len(image.shape)
 
    if C_index in [0,1] and __is_channels__(image.shape[1-C_index]):
        return True
    return False

def isHWC(image): #weak check
    '''
        Is the given image in HWC or NHWC.
        Arguments:
            image: to check
    '''
    C_index = 4 - len(image.shape)
    if C_index in [0,1] and __is_channels__(image.shape[-1]):
        return True
    return False