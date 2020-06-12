import unittest

import numpy as np

import gym

import pyworld.toolkit.tools.gymutils.spaces as spaces
import pyworld.toolkit.tools.gymutils.policy as policy

class TestDiscretePolicy(unittest.TestCase):

    def test_0(self):
        with self.assertRaises(AssertionError):
            policy.DiscretePolicy(0)

    def test_sampler(self):
        with self.assertRaises(NotImplementedError):
            p = policy.DiscretePolicy(1)
            p(None)

    def test_discrete(self):
        policy.DiscretePolicy(gym.spaces.Discrete(1))
    
    def test_int(self):
        p = policy.DiscretePolicy(1)
        self.assertEqual(p.action_space.n, 1)
        self.assertEqual(p.action_space.dtype, np.int64)

    def test_dtype(self):
        p = policy.DiscretePolicy(1, dtype=np.float32)
        self.assertEqual(p.action_space.dtype, np.float32)
        p = policy.DiscretePolicy(gym.spaces.Discrete(1), dtype=np.uint8)
        self.assertEqual(p.action_space.dtype, np.uint8)

class TestUniform(unittest.TestCase):

    def test_1(self):
        p = policy.uniform(1)
        self.assertEqual(p.sample(), 0)

    def test_n(self):
        space = gym.spaces.Discrete(5)
        p = policy.uniform(space)
        for i in range(10):
            space.contains(p(None))

    def test_dtype(self):
        p = policy.uniform(1, dtype=np.float32)
        self.assertEqual(type(p(None)), np.float32)

class TestOneHot(unittest.TestCase):

    def test_1(self):
        p = policy.onehot(policy.uniform(1))
        self.assertEqual(p(None), np.array([1]))

    def test_n(self):
        space = gym.spaces.Discrete(5)
        p = policy.onehot(policy.uniform(space))
        onehot_space = spaces.OneHot(5)
        for i in range(10):
            space.contains(p(None))

    def test_dtype(self):
        p = policy.onehot(policy.uniform(1), dtype=np.uint8)
        self.assertEqual(p(None).dtype, np.uint8)

    def test_policy_int(self):
        with self.assertRaises(AssertionError):
            policy.onehot(policy.uniform(1, dtype=np.float32))

if __name__ == "__main__":
    unittest.main()


