import unittest

import numpy as np
import gym


import pyworld.toolkit.tools.gymutils as gu

ITER_LIMIT = 25

class TestEnv(gym.Env):

    def __init__(self):
        super(TestEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(np.float32(0), np.float32(1), shape=(1,5,5))
        self.i = 0

    def step(self, action):
        self.i += 1
        return self.observation_space.sample(), 0., self.i >= ITER_LIMIT, None
    
    def reset(self):
        self.i = 0
        return self.observation_space.sample()


class TestIterator(unittest.TestCase):

    def test_GymIterator(self):
        env = TestEnv()
        policy = gu.policy.uniform(env.action_space)
        iterator = gu.iterators.GymIterator(env, policy)
        iterator = gu.iterators.itertools.islice(iterator, 0, ITER_LIMIT + 2)
        self.assertEqual(len([i for i in iterator]), ITER_LIMIT+1)


    

        

        






if __name__ == "__main__":
    unittest.main()




