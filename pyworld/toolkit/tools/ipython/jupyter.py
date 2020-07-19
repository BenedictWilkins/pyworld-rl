#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 12-06-2020 17:14:51

    Part of IPython utiltiies with a focus on Jupyter.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import os
import sys
import pprint

from IPython.core.display import HTML

class HTMLOut(object):

    def __init__(self, html_template="{0}"):
        self.__template = html_template
        self.__text = HTML()

    def write(self, *args, **kwargs):
        text = pprint.pformat(*args, **kwargs)
        self.__text.value = text
        display(HTML(self.__template(text)))

def local_import(root=1):
    """ Adds local directory to the system path.
        Example: 
        
        Project directory Structure: 
            project
                lib
                    __init__.py
                notebook_folder
                    notebook.ipynb

        in notebook.ipynb:

            local_import(1)
            import project.lib
            ...

    Args:
        root (int, optional): location of the root directory. Defaults to 0 (the current directory)
    """
    module_path = os.path.abspath('.')
    module_path = "/".join(module_path.split("/")[:-root])
    if module_path not in sys.path:
        sys.path.append(module_path)
    return module_path
    

def cell_variables():
    """ Get all of the (global) variables in the current (or previous) Jupyter Notebook cell.

    Returns:
        dict: all global variables in the cell.
    """
    import inspect
    from inspect import getframeinfo
    import io
    from contextlib import redirect_stdout
    
    ipy = get_ipython()
    out = io.StringIO()
    
    with redirect_stdout(out): #get all cell inputs
        ipy.magic("history {0}".format(ipy.execution_count))
    cell_inputs = out.getvalue()
    
    #get caller globals ---- LOL HACKz
    frame = inspect.stack()[1][0]
    c_line = getframeinfo(frame).lineno
    g = frame.f_globals
    if not "_" in g:
        raise ValueError("The function \"cell_variables\" must be called from within a Jupyter Notebook.")
    
    IGNORE = "#ignore"
    #process each line...
    x = cell_inputs.replace(" ", "").split("\n")
    x.pop(c_line - 1) #lines are 1 indexed, remove the calling line 
    x = [a.split("=")[0] for a in x if "=" in a and IGNORE not in a] #all of the variables in the cell
    result = {k:g[k] for k in x if k in g}

    return result
