import numpy as np
import torch

import pyworld.toolkit.tools.torchutils as tu

def test_collect():
    x = torch.from_numpy(np.random.randint(0,8,size=100))
    def model(x):
        return x + 1
    y = tu.collect(model, x, batch_size=8)
    print(x)
    print(y)

test_collect()


