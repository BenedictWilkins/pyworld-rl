
import numpy as np

def unitvector(d, n=1):
    """ Random unit vector(s).

    Args:
        d (int): dimension of the unit vector.
        n (int, optional): number of vectors to sample. Defaults to 1.

    Returns:
        ndarray: unit vectors (n x d)
    """
    r = np.random.normal(size=(n,d))
    return (r / np.linalg.norm(r, axis=1)[:,np.newaxis]).squeeze()

def walk(n, d, alpha=1.):
    """
    Random walk of length n in n dimensions.

    Args:
         n (int): length of the walk > 1
         d (int): dimension of the walk > 0
         alpha (float,  optional): step size Defaults to 1..

     Returns:
        ndarray: the random walk (n x d)
    """
    assert n > 1 
    assert d > 0
    return np.cumsum(alpha * unitvector(d, n=n), axis=0)