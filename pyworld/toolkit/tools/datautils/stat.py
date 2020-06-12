import numpy as np
import scipy.stats as stats

def p_greater(a, b):
    """ Assuming a and b are a collection of realisations of random variables A and B respectively, estimate Pr(A > B).

    Args:
        a ([list, np.array]): A collection of samples.
        b ([list, np.array]): A collection of samples.

    Returns:
        float: Probability Pr(A > B)
    """
    assert len(a.shape) == 1 and len(b.shape) == 1
    ranks = stats.rankdata(np.concatenate((a,b)))
    n, m = a.shape[0], b.shape[0]
    if n < m:
        return (ranks[:n] - np.arange(1, n+1)).sum() / (m*n)
    else:
        return 1 - ((ranks[n:] - np.arange(1, m+1)).sum() / (m*n))