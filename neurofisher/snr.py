"""Module for computing signal-to-noise ratios (SNR)

This module provides functions for computing SNR using Fisher information.
"""

import numpy as np

from neurofisher.utils import compute_firing_rate


def SNR_bound_instantaneous(x, C, b):
    """Compute SNR about the latent trajectoryusing instantaneous Fisher information.

    Parameters
    ----------
    x : ndarray
        Latent trajectory (unit variance)
    C : ndarray
        Loading matrix
    b : ndarray
        Bias vector

    Returns
    -------
    float
        SNR in dB
    """
    firing_rates = compute_firing_rate(x, C, b)

    SNR = 0
    for i, firing_rate in enumerate(firing_rates):
        CC = C @ np.diag(firing_rate) @ C.T + np.eye(C.shape[0]) * 1e-6
        invCC = np.linalg.inv(CC)
        invCC[invCC > 1e6] = 0
        SNR += np.trace(invCC)
    SNR = SNR / firing_rates.shape[0]
    SNR = 10 * np.log10(C.shape[0] / SNR)

    return SNR
