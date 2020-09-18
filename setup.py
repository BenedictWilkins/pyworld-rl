#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:12:18 2019

@author: ben
"""

from setuptools import setup, find_packages

setup(name='pyworld',
      version='0.0.1',
      description='',
      url='',
      author='Benedict Wilkins',
      author_email='benrjw@gmail.com',
      packages=find_packages(),
      install_requires=["opencv-python", 
                        "scikit-image",
                        "matplotlib", 
                        "numpy",
                        "plotly", 
                        "gym[atari]",
                        "ipywidgets", 
                        "ipycanvas", 
                        "jupyterlab",
                        "h5py",
                        "torch"],
      zip_safe=False)

# NOTE: opencv requires ffmpeg


