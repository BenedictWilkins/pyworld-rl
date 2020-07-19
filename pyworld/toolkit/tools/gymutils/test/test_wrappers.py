#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Created on 17-06-2020 17:29:19

    Test gym wrappers.
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import unittest

import numpy as np
import gym
import pyworld.toolkit.tools.gymutils.wrappers as wrappers 

class TestEnv(gym.Env):

    def __init__(self):
        super(TestEnv, self).__init__()
        self.iter_limit = 10
        self.i = 0

    def step(self, action):
        self.i += 1
        return self.observation_space.sample(), 0., self.i >= self.iter_limit, None
    
    def reset(self):
        self.i = 0
        return self.observation_space.sample()


class Test(unittest.TestCase):

    def test_Float(self):
        class TestFloat(TestEnv):

            def __init__(self):
                super(TestFloat,self).__init__()
                self.observation_space = gym.spaces.Box(0,255, shape=(1,5,5), dtype=np.uint8)

        env = TestFloat()
        env = wrappers.Float(env)
        self.assertTrue((env.observation_space.high == 1.).all())
        self.assertTrue((env.observation_space.low == 0.).all())
        self.assertEqual(env.observation_space.shape, (1,5,5))
        self.assertEqual(env.observation_space.dtype, np.float32)

        self.assertEqual(env.step(0)[0].shape, env.observation_space.shape)
        self.assertEqual(env.step(0)[0].dtype, np.float32)
        self.assertLessEqual(env.step(0)[0].max(), 1.)
        self.assertGreaterEqual(env.step(0)[0].min(), 0.)


    def test_Integer(self):
        class TestInteger(TestEnv):

            def __init__(self):
                super(TestInteger,self).__init__()
                self.observation_space = gym.spaces.Box(np.float32(0.), np.float32(1.), shape=(1,5,5), dtype=np.float32)

        env = TestInteger()
        env = wrappers.Integer(env)
        self.assertTrue((env.observation_space.high == 255).all())
        self.assertTrue((env.observation_space.low == 0).all())
        self.assertEqual(env.observation_space.shape, (1,5,5))
        self.assertEqual(env.observation_space.dtype, np.uint8)

        self.assertEqual(env.step(0)[0].shape, env.observation_space.shape)
        self.assertEqual(env.step(0)[0].dtype, np.uint8)
        self.assertLessEqual(env.step(0)[0].max(), 255)
        self.assertGreaterEqual(env.step(0)[0].min(), 0)


    def test_CHW(self):
        class TestCHW(TestEnv):

            def __init__(self):
                super(TestCHW,self).__init__()
                self.observation_space = gym.spaces.Box(np.float32(0.), np.float32(1.), shape=(4,5,1), dtype=np.float32)

        env = TestCHW()
        env = wrappers.CHW(env)
        self.assertTrue((env.observation_space.high == 1).all())
        self.assertTrue((env.observation_space.low == 0).all())
        self.assertEqual(env.observation_space.shape, (1,4,5))
        self.assertEqual(env.observation_space.dtype, np.float32)

        self.assertEqual(env.step(0)[0].shape, env.observation_space.shape)
        self.assertEqual(env.step(0)[0].dtype, np.float32)
        self.assertLessEqual(env.step(0)[0].max(), 1)
        self.assertGreaterEqual(env.step(0)[0].min(), 0)


    
    def test_HWC(self):
        class TestHWC(TestEnv):

            def __init__(self):
                super(TestHWC,self).__init__()
                self.observation_space = gym.spaces.Box(np.float32(0.), np.float32(1.), shape=(1,5,4), dtype=np.float32)

        env = TestHWC()
        env = wrappers.HWC(env)
        self.assertTrue((env.observation_space.high == 1).all())
        self.assertTrue((env.observation_space.low == 0).all())
        self.assertEqual(env.observation_space.shape, (5,4,1))
        self.assertEqual(env.observation_space.dtype, np.float32)

        self.assertEqual(env.step(0)[0].shape, env.observation_space.shape)
        self.assertEqual(env.step(0)[0].dtype, np.float32)
        self.assertLessEqual(env.step(0)[0].max(), 1)
        self.assertGreaterEqual(env.step(0)[0].min(), 0)




if __name__ == "__main__":
    unittest.main()