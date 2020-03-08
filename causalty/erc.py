from typing import Tuple

import numpy as np

from causalty.mvar import tsdata_to_var, var_to_cpsd


def partial_coherence(s: np.ndarray) -> np.ndarray:
    """
    Compute partial coherence given the cross-power spectral density
    """
    chi = np.zeros_like(s)
    for i in range(s.shape[2]):
        c = np.linalg.inv(s[:, :, i])
        cd = np.diag(c)
        chi[:, :, i] = c / np.sqrt(np.outer(cd, cd))
    return chi


def sddtf(h: np.ndarray, chi: np.ndarray) -> np.ndarray:
    """
    Short d Directed Transfer Function
    """
    h_ = np.abs(h)
    chi_ = np.abs(chi)
    z = (h_ * chi_) / np.sqrt(np.sum(h_**2 * chi_**2))
    return z


def erc(x: np.ndarray, lags: int,
        fres: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Event Related Causality
    """
    a, sig = tsdata_to_var(x, lags)
    s, h = var_to_cpsd(a, sig, fres)
    chi = partial_coherence(s)
    z = sddtf(h, chi)
    return z, s
