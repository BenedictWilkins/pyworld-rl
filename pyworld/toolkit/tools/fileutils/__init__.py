#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 17-05-2019 16:56:21 

    Saving and loading many different formats all in one place, as well as lots of helpful functions.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import os
import pickle
import datetime
import json
import re
import fnmatch #unix file name matcher

from . import __import__

from .formats import text
from .formats import image
from .formats import video
from .formats import sound
from .formats import misc

def load(path, **kwargs):
    """ Load a file.

    Args:
        path (str): file path.

    Returns:
        : loaded data
    """
    path = expand_user(path)
    if has_extension(path):
       _, ext = os.path.splitext(path)
       return __import__.fileio.io[ext](path, **kwargs)

def save(path, data, force=True, overwrite=False, **kwargs):
    """ Save a file.

    Args:
        path (str): file path.
        data : to save
        force (bool, optional): Create directories if they dont exist already. Defaults to True.
        overwrite (bool, optional): Overwrite the file (if it exists), see fileutils.file. Defaults to False.

    Returns:
        object: result of the save (file type dependant).
    """
    path = file(path, force=force, overwrite=overwrite)
    if has_extension(path):
       _, ext = os.path.splitext(path)
       return __import__.fileio.io[ext](path, data, **kwargs)
    
def expand_user(path):
    """ Expands \"~\" in a file path (see os.path.expanduser).

    Args:
        path (str): path

    Returns:
        str: expanded path
    """
    if path.startswith("~"):
        return os.path.expanduser(path)
    return path

def file(file, force=True, overwrite=False):
    """ Format a file path. Creates new directories if they dont exist (if force is True).

    Args:
        file (str): file path.
        force (bool, optional): Create directories if they dont exist. Defaults to True.
        overwrite (bool, optional): if False, the file will be made unique: \"file(i).ext\". Defaults to False.

    Returns:
        str: formatted file path.
    """
    file = expand_user(file)

    if force:
        path, _ = os.path.split(file)
        if not os.path.exists(path):
            os.makedirs(path)

    if not overwrite:
        o_file, ext = os.path.splitext(file)
        i = 0
        while os.path.isfile(file):
            i += 1
            file = o_file + "(" + str(i) + ")" + ext

    return file


#TODO rename
def next(file, force=True):
    o_file, ext = os.path.splitext(file)

    i = 0
    while os.path.exists(file):
        i += 1
        file = o_file + "(" + str(i) + ")" + ext
        
    if force:
        os.makedirs(file)

    return file

def files(path, full=False):
    """ Get all files in a directory.

    Args:
        path (str): directory path.
        full (bool, optional): fully qualified file paths, or just files names. Defaults to False.

    Raises:
        ValueError: if the path doesnt exist 

    Returns:
        list: files names (or fully qualified paths)
    """
    path = expand_user(path)
    if not os.path.isdir(path):
        raise ValueError("path {0} does not exist".format(path))

    if not full:
        return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def sort_files(file s):
    """ 
        Sorts file names by their unique tag (X):

        foo.txt
        foo(1).txt
        ...
        foo(N).txt

    Args:
        files (str): file names
    Returns:
        [type]: [description]
    """
    '''
        
    '''
    def number(file):
        f = os.path.splitext(os.path.basename(file))[0]
        x = re.findall("\(([0-9]+)\)", f)
        if len(x) == 1:
            return int(x[0])
        elif len(x) == 0:
            return 0
        else:
            return float('inf')

    return sorted(files, key = lambda x: number(x))


def dirs(path, full=False):
    if not os.path.isdir(path):
        raise ValueError("path {0} does not exist".format(path))
    if not full:
        return [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    else:
        return [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]


def file_datetime():
    return str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))

def has_extension(file):
    return len(file.split('.')) > 1

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def files_with_extention(path, extension, full=False):
    return [file for file in files(path, full=full) if file.endswith(extension)]

def filter(files, blacklist=[], whitelist=[]):
    for pattern in whitelist:
        files = fnmatch.filter(files, pattern)

    rfiles = set()
    for pattern in blacklist:
        rfiles = rfiles.union(set(fnmatch.filter(files, pattern)))

    return list(set(files) - rfiles)
