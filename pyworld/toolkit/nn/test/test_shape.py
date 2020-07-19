import unittest

import numpy as np
import torch
import torch.nn as nn
import pyworld.toolkit.nn.shape as S

class TestShape(unittest.TestCase):

    def test_conv2d(self):
        layer = nn.Conv2d(3,16,kernel_size=3,stride=2,padding=1)
        input_shape = (3,10,10)
        shape = S.conv2d_shape(layer, input_shape)
        t = torch.from_numpy(np.zeros((1, *input_shape), dtype=np.float32))
        t = layer(t)
        self.assertEqual(shape, S.as_shape(t.shape)[1:])

    def test_linear(self):
        layer = nn.Linear(10, 20)
        self.assertEqual(S.linear_shape(layer), (20,))

if __name__ == "__main__":
    unittest.main()