import unittest

import numpy as np
import gym


import pyworld.toolkit.tools.gymutils as gu

class TestEnv(gym.Env):

    def __init__(self):
        super(TestEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(np.float32(0), np.float32(1), shape=(1,5,5))
        self.i = 0
        self.limit = 25

    def step(self, action):
        self.i += 1
        return self.observation_space.sample(), 0., self.i >= 25, None
    
    def reset(self):
        self.i = 0
        return self.observation_space.sample()


class TestEpisode(unittest.TestCase):

    def test_pack(self):
        env = TestEnv()
        iterator = gu.iterators.GymIterator(env, mode=gu.mode.sar)
        iterator = gu.iterators.itertools.islice(iterator, 0, 5)
        s,a,r = gu.pack(iterator)
        self.assertEqual(s.shape, (5,1,5,5))
        self.assertEqual(a.shape, (5,))
        self.assertEqual(r.shape, (5,))
        
    def test_episode(self):
        env = TestEnv()
        policy = gu.policy.uniform(env.action_space)
        
        s,a,r = gu.episode(env, policy, mode=gu.mode.sar)

        print(s.shape)
        print(a.shape)
        print(r.shape)

        #TODO test

    def test_episodes(self):
        env = TestEnv()
        policy = gu.policy.uniform(env.action_space)
        for s,a,r in gu.episodes(env, policy, mode=gu.mode.sar):
            print(s.shape)
            print(a.shape)
            print(r.shape)

        #TODO test


        






if __name__ == "__main__":
    unittest.main()




