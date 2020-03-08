from typing import Tuple

import numpy as np

from causalty.util import block_fft, demean, mrdivide


def var_specrad(a: np.ndarray) -> float:
    """
    Compute the spectral radius for the coefficient matrix
    """
    n_vars, _, lags = a.shape
    pn1 = (lags-1)*n_vars
    a_ = np.vstack((
        a.reshape(n_vars, -1),
        np.hstack((np.eye(pn1),
                   np.zeros((pn1, n_vars))
                   ))
    ))
    w, _ = np.linalg.eig(a_)
    rho = np.max(np.abs(w))
    return rho


def tsdata_to_var(x: np.ndarray, lags: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the coefficient matrix give time series data in the form
    (sources, observations, trials)
    """
    x_ = np.atleast_3d(x)
    n_vars, n_obs, n_trials = x_.shape
    m = n_trials * (n_obs - lags)

    x_ = demean(x_)
    x0 = x_[:, lags:].reshape((n_vars, m))
    xl = np.zeros((n_vars, lags, m))
    for i in range(1, lags+1):
        xl[:, i-1] = x_[:, lags-i:-i].reshape((n_vars, m))
    xl = xl.reshape((n_vars * lags, m))
    a = mrdivide(x0, xl)
    e = x0 - a @ xl
    sig = (e @ e.T) / (m - 1)
    return a.reshape((n_vars, n_vars, lags)), sig


def var_to_transfer_function(a: np.ndarray, fres: int) -> np.ndarray:
    """
    Convert the coefficient matrix to its equivalent spectral transfer function
    form
    """
    n_vars = a.shape[0]
    i = np.eye(n_vars).reshape((n_vars, n_vars, 1))
    af = block_fft(np.concatenate((i, -a), axis=2), 2*fres)
    i = i.squeeze()
    k = fres + 1
    h = np.zeros((n_vars, n_vars, k), dtype=np.complex128)
    for j in range(k):
        h[:, :, j] = mrdivide(i, af[:, :, j])
    return h


def var_to_cpsd(a: np.ndarray, sig: np.ndarray,
                fres: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the cross-power spectral density from the coefficient matrix
    """
    n_trials = a.shape[0]
    k = fres + 1
    h = var_to_transfer_function(a, fres)
    s = np.zeros((n_trials, n_trials, k), dtype=np.complex128)
    for i in range(k):
        s[:, :, i] = h[:, :, i] @ sig @ h[:, :, i].conj().T
    return s, h
