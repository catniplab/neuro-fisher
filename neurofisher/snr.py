"""Module for computing signal-to-noise ratios (SNR)

This module provides functions for computing SNR using Fisher information.
"""

import numpy as np
from neurofisher.utils import compute_firing_rate, update_bias


def compute_instantaneous_snr(x, C, b, tgt_rate):
    """Compute SNR from instantaneous fisher information.

    Parameters
    ----------
    x : ndarray
        Latent trajectory (unit variance)
    C : ndarray
        Loading matrix
    b : ndarray
        Bias vector
    tgt_rate : float
        Target firing rate

    Returns
    -------
    tuple
        (SNR in dB, updated bias)
    """
    rates = compute_firing_rate(x, C, b)
    b = update_bias(np.mean(rates, axis=0), b, tgt_rate=tgt_rate)
    rates = compute_firing_rate(x, C, b)

    snr = 0
    reg = 1e-6  # Small regularization term to prevent singularity
    for i, rate in enumerate(rates):
        # Scale rates to prevent overflow
        rate_scale = np.max(rate)
        if rate_scale > 0:
            rate = rate / rate_scale
        # Add small regularization to diagonal to prevent singularity
        F = C @ np.diag(rate) @ C.T
        F = F + reg * np.eye(F.shape[0])
        try:
            snr += np.trace(np.linalg.inv(F)) / rate_scale if rate_scale > 0 else 0
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            snr += np.trace(np.linalg.pinv(F)) / rate_scale if rate_scale > 0 else 0
    snr = snr / rates.shape[0]
    snr = 10 * np.log10(C.shape[0] / snr)

    return snr, b
