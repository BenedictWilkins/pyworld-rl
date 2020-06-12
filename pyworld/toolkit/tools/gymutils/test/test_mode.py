import unittest

import numpy as np

import pyworld.toolkit.tools.gymutils.mode as mode

class TetObservation(unittest.TestCase):

    def test_observation(self):
        obs = mode.observation(1,2,3)
        self.assertEqual(obs[0], 1)
        self.assertEqual(obs[1], 2)
        self.assertEqual(obs[2], 3)

        self.assertEqual([1,2,3], [i for i in obs])

    def test_state(self):
        obs = mode.s(1)
        self.assertEqual(obs.state, 1)

    def test_pack(self):

        # test with list
        observations = [mode.observation(np.array([i, i*i]), i) for i in range(10)]
        s, a = mode.pack(observations)
        xa = np.arange(0,10)
        xs = np.concatenate((x[:,np.newaxis], (x*x)[:,np.newaxis]), axis=1)

        self.assertTrue(np.all(a == xa))
        self.assertTrue(np.all(s == xs))

        # test with generator
        def iter():
            for i in range(10):
                yield np.array([i, i*i]), i

        observations = iter()
        s, a = mode.pack(observations)
        self.assertTrue(np.all(a == xa))
        self.assertTrue(np.all(s == xs))


if __name__ == "__main__":
    unittest.main()



