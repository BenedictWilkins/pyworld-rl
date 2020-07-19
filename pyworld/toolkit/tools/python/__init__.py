#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-06-2020 13:52:23

    Useful Python functions, typically for development/debugging purposes.
    Heavy use of the \"inspect\" module.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import inspect
import sys
 
def get_classes(file):
    """ Get all classes defined in a file (module). 
    
    Args:
        file (str): name of file (module), commonly __name__.

    Returns:
        dict: all classes defined in the file.
    """
    return dict(inspect.getmembers(sys.modules[file], inspect.isclass))