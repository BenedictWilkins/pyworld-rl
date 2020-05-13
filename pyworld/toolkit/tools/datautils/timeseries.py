import numpy as np


from scipy import fftpack

# ========================================================================================== #

def spectrum(y, samplerate=1.):
    """ Computes the positive frequency spectrum of y via fourier transform. 

    Args:
        x (ndarray): sample rates
        y (ndarray): signal

    Returns:
        tuple[ndarray, ndarray]: frequency, amplitude
    """
    yf = fftpack.fft(y, y.size)
    amp = np.abs(yf) # get amplitude spectrum 
    freq = fftpack.fftfreq(y.size, samplerate)
    return freq[0:freq.size//2], (2/amp.size)*amp[0:amp.size//2]

def detrend(x, y):
    """ Detrends the time series y.

    Args:
        x (ndarray): sample rates
        y (ndarray): time series to detrend

    Returns:
        tuple[ndarray, ndarray]: x', y'
    """
    return x[1:], y[1:] - y[:-1]