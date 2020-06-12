import gym


import gym
import numpy as np


class OneHot(gym.Space):

    def __init__(self, size, dtype=np.float32):
        assert isinstance(size, int) and size > 0
        self.size = size
        super(OneHot, self).__init__((size,), dtype)

    def sample(self):
        r = np.zeros(self.size, dtype=self.dtype)
        r[np.random.randint(self.size)] = 1
        return r

    def contains(self, x):
        if isinstance(x, (list, tuple, np.ndarray)) and len(x) == self.size:
            u, c = np.unique(x, return_counts=True)
            return u[0] == 0 and u[1] == 1 and c[int(u[0])] == self.size - 1
        else:
            return False

    def __repr__(self):
        return "OneHot(%d)" % self.size