#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 14-06-2020 13:11:25
    
    File IO for common text formats.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pyworld.toolkit.tools.fileutils.__import__ as I

class TextIO(I.fileio):

    def __init__(self):
        super(TextIO, self).__init__('.txt')
    
    def save(self, file, data, mode='w'):
        if mode == 'w':
            with open(file, mode) as f:
                f.write(data)
        else:
            raise NotImplementedError("TODO write in append mode...?")

    def load(self, file, mode='r'):
        with open(file, mode) as f:
            return f.readlines() 

class JsonIO(I.fileio):

    def __init__(self):
        super(JsonIO, self).__init__('.json', 'json')
    
    def save(self, file, data):
        with open(file, 'w') as fp:
            self.json.dump(data, fp)

    def load(self, file):
        with open(file, 'r') as fp:
            data = self.json.load(fp)
        return data

class YamlIO(I.fileio):

    def __init__(self):
        super(YamlIO, self).__init__('.yaml', 'yaml')

    def save(self, file, data):
        with open(file, 'w') as fp:
            self.yaml.dump(data, fp)

    def load(self, file):
        with open(file, 'r') as fp:
            data = self.yaml.full_load(fp)
        return data