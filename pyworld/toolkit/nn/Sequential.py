#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 16-06-2020 11:50:11

    Simple sequential network that adds some additional functionality over torch.nn.Sequential.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from collections import OrderedDict

import torch.nn as nn
import torch.nn.functional as F

from .inverse import inverse
from .shape import shape, as_shape

class LayerDict(OrderedDict):

    def __init__(self, module, layers):
        self.module = module
        super(LayerDict, self).__init__(layers)

    def __setitem__(self, k, v):
        super(LayerDict, self).__setitem__(k, v)
        if isinstance(v, nn.Module):
            self.module.add_module(k, v)
        else:
            self.__dict__[k] = v
            
class Sequential(nn.Module):

    def __init__(self, **layers):
        super(Sequential, self).__init__()
        self.layers = LayerDict(self, layers)

    def forward(self, *args, **kwargs):
        liter = iter(self.layers.values())
        y = next(liter)(*args, **kwargs)
        for layer in liter:
            y = layer(y)
        return y

    def inverse(self, share_weights=False):
        if isinstance(share_weights, bool):
            share_weights = {k:share_weights for k in self.layers.keys()}

        ilayers = OrderedDict()
        for k,v in reversed(self.layers.items()):
            if isinstance(v, nn.Module):
                v = inverse(v, share_weights=share_weights.get(k, False))[0]
            ilayers[k] = v

        return Sequential(**ilayers).to(self.device)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def shape(self, input_shape):
        input_shape = as_shape(input_shape)
        result = OrderedDict()
        for k,v in self.layers.items():
            if isinstance(v, nn.Module):
                input_shape = result[k] = shape(v, input_shape)
        return result

    def __str__(self):
        slayers = "    " + "\n    ".join(["({0}): {1}".format(k,str(v)) for k,v in self.layers.items()])
        return "{0}(\n{1}\n)".format(self.__class__.__name__, slayers)

    def __repr__(self):
        return str(self)
        
class View:

    def __init__(self, *shape):
        self.shape = shape
    
    def __call__(self, x):
        return x.view(x.shape[0], *self.shape)

    def __str__(self):
        attrbs = ",".join(["{0}={1}".format(k,v) for k,v in dict(shape=self.shape).items()])
        return "{0}({1})".format(self.__class__.__name__, attrbs)

    def __repr__(self):
        return str(self)

def view(*shape):
    return View(*shape)


