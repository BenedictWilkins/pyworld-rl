#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-06-2020 14:01:28

    File IO for common image formats.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from pyworld.toolkit.tools.fileutils.__import__ import fileio

class CVImageIO(fileio):

    def __init__(self, ext):
        super(CVImageIO, self).__init__(ext, 'cv2')

    def save(self, file, data):
        self.cv2.imwrite(file, data)

    def load(self, file):
        return self.cv2.imread(file)

class pngIO(CVImageIO):
    def __init__(self):
        super(pngIO, self).__init__('.png')

class jpegIO(CVImageIO):
    def __init__(self):
        super(jpegIO, self).__init__('.jpeg')
    
class jpegIO2(CVImageIO):
    def __init__(self):
        super(jpegIO, self).__init__('.jpg')

class bmpIO(CVImageIO):
    def __init__(self):
        super(bmpIO, self).__init__('.bmp')
    
