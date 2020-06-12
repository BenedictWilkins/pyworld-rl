import unittest

import numpy as np

import pyworld.toolkit.tools.gymutils.spaces as spaces

class TestOneHot(unittest.TestCase):

    def test_0(self):
        with self.assertRaises(AssertionError):
            spaces.OneHot(0)
    
    def test_1(self):
        self.assertEqual(spaces.OneHot(1).sample(), np.array([1.]))
        
    def test_sample_contains(self):
        space = spaces.OneHot(3)
        for i in range(10):
            self.assertTrue(space.contains(space.sample()))
        
    def test_not_contains(self):
        space = spaces.OneHot(3)
        self.assertFalse(space.contains(np.array([1,0,0,0])))
        self.assertFalse(space.contains(np.array([[1,0],[0,0],[0,0]])))
        self.assertFalse(space.contains(np.array([1,1,0])))
        self.assertFalse(space.contains(np.array([1,2,0])))
        self.assertFalse(space.contains(np.array([])))

    def test_shape(self):
        space = spaces.OneHot(4)
        self.assertEqual(space.shape, (4,))
    
    def test_dtype(self):
        space = spaces.OneHot(3, dtype=np.uint8)
        self.assertEqual(space.dtype, np.uint8)
        self.assertEqual(space.sample().dtype, np.uint8)

    def test_contains_dtype(self):
        space = spaces.OneHot(3, dtype=np.uint8)
        self.assertTrue(space.contains(np.array([1,0,0], dtype=np.float32))) #maybe we want this behaviour?

if __name__ == "__main__":
    unittest.main()



