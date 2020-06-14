#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 12-06-2020 17:14:51

    Part of IPython utiltiies with a focus on Jupyter.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

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
    
    #process each line...
    x = cell_inputs.replace(" ", "").split("\n")
    x.pop(c_line - 1) #lines are 1 indexed, remove the calling line 
    x = [a.split("=")[0] for a in x if "=" in a] #all of the variables in the cell
    result = {k:g[k] for k in x if k in g}

    return result
