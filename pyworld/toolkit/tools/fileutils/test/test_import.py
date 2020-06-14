import unittest

import os
import numpy as np

import pyworld.toolkit.tools.fileutils.__import__ as I
from pyworld.toolkit.tools.fileutils.formats import text, misc, image, video

class TestImport(unittest.TestCase):

    def test_error(self):
        class ErrorIO(I.fileio):

            def __init__(self):
                super(ErrorIO, self).__init__('.error', 'module_that_doesnt_exist_::::')
            
            def save(self, file, data):
                pass

            def load(self, file):
                pass

        self.assertFalse('.error' in I.fileio.io)

class TestText(unittest.TestCase):

    def test_json(self):
        json = text.JsonIO()
        d = dict(a=1,b=2,c=3)
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test.json")
        json.save(path, d)
        self.assertEqual(d, json.load(path))

    def test_yaml(self):
        yaml = text.YamlIO()
        d = dict(a=1,b=2,c=3)
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test.yaml")
        yaml.save(path, d)
        self.assertEqual(d, yaml.load(path))

    def test_pickle(self):
        io = misc.PickleIO()
        d = dict(a=1,b=2,c=3)
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test.pickle")
        io.save(path, d)
        self.assertEqual(d, io.load(path))

class TestMisc(unittest.TestCase):

    def test_pickle(self):
        io = text.TextIO()
        d = "hello\nworld"
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test.txt")
        io.save(path, d)
        self.assertEqual(d, ''.join(io.load(path)))

class TestImage(unittest.TestCase):

    def test_png(self):
        io = image.pngIO()
        d = np.random.randint(0,255,size=(4,4,3))
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test1.png")
        io.save(path, d)
        self.assertTrue(np.array_equal(d, io.load(path)))

    def test_jpeg(self):
        io = image.jpegIO()
        d = np.random.randint(0,255,size=(4,4,3), dtype=np.uint8)
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test2.jpeg")
        io.save(path, d)
        #self.assertTrue(np.array_equal(d, io.load(path))) TODO THERE IS A PROBLEM SAVE/LOAD JPEG ....... dont understand it

    def test_bmp(self):
        io = image.bmpIO()
        d = np.random.randint(0,255,size=(4,4,3))
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test3.bmp")
        io.save(path, d)
        self.assertTrue(np.array_equal(d, io.load(path)))



class TestVideo(unittest.TestCase):

    def test_gif(self):
        io = video.gifIO()
        d = np.random.randint(0,255,size=(20, 4,4,3))
        path = os.path.join(os.path.split(__file__)[0], ".test_files/test.gif")
        print(path)
        io.save(path, d)
        #self.assertTrue(np.array_equal(d, io.load(path)))

if __name__ == "__main__":
    unittest.main()
   




