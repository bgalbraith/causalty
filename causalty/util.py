from typing import Optional

import numpy as np


def mrdivide(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """
    MATLAB's matrix right division function
    https://www.mathworks.com/help/matlab/ref/mrdivide.html
    Solve systems of linear equations xA = B for x
    """
    x, _, _, _ = np.linalg.lstsq(a.T, b.T, rcond=None)
    return x.T


def demean(x: np.ndarray, normalize: bool = False) -> np.ndarray:
    """
    Remove the mean and optionally normalize the input signal
    """
    x_ = np.atleast_3d(x)
    n_vars, n_obs, n_trials = x_.shape
    x_ = x_.reshape((n_vars, n_obs*n_trials))
    x_ -= np.mean(x_, axis=1, keepdims=True)
    if normalize:
        x_ / np.std(x_, axis=1, keepdims=True)
    return x_.reshape((n_vars, n_obs, n_trials))


def block_fft(a: np.ndarray, q: Optional[int] = None):
    """
    Block FFT
    """
    n1, n2, p = a.shape
    if q is None:
        q = p
    x = np.fft.fft(a.reshape((-1, p), order='F').T, q, axis=0).T
    return x.reshape((n1, n2, q), order='F')
