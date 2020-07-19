#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 18-06-2020 13:56:16

    Test the get_classes function - requires its own file.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import unittest

import pyworld.toolkit.tools.python as P

class Test(unittest.TestCase):

    def test(self):
        clss = P.get_classes(__name__)
        self.assertEqual(clss, {"Test":Test})

if __name__ == "__main__":
    unittest.main()