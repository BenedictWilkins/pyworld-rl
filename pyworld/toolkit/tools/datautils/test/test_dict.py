import unittest

import pyworld.toolkit.tools.datautils.dict as D


class TestFDict(unittest.TestCase):

    def test_fdict(self):
        d = D.fdict(a=1,b=2,c=3)
        self.assertEqual(d, dict(a=1,b=2,c=3))
    
    def test_fdict_set(self):
        d = D.fdict(a=1,b=2,c=3)
        with self.assertRaises(KeyError):
            d['a'] = 2
        d['d'] = 4
        self.assertEqual(d, dict(a=1,b=2,c=3,d=4))

        d.update(dict(e=5,f=6))
        self.assertEqual(d, dict(a=1,b=2,c=3,d=4, e=5, f=6))

        with self.assertRaises(KeyError):
            d.update(dict(g=7, b=2))

if __name__ == "__main__":
    unittest.main()



