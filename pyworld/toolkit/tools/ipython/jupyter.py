#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 12-06-2020 17:14:51

    Part of IPython utiltiies with a focus on Jupyter.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

def cell_variables(offset=0):
    """ Get all of the (global) variables in the current (or previous) Jupyter Notebook cell.

    Args:
        offset (int, optional): off set to previously executed cells. Defaults to 0 (the current cell)

    Returns:
        dict: all global variables in the cell.
    """
    import io
    from contextlib import redirect_stdout
    
    ipy = get_ipython()
    out = io.StringIO()
    
    with redirect_stdout(out):
        ipy.magic("history {0}".format(ipy.execution_count - offset))
    
    #process each line...
    x = out.getvalue().replace(" ", "").split("\n")
    x = [a.split("=")[0] for a in x if "=" in a] #all of the variables in the cell
    g = globals()
    result = {k:g[k] for k in x if k in g}
    return result
