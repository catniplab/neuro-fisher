"""Module for computing signal-to-noise ratios (SNR)

This module provides functions for computing SNR using Fisher information.
"""

import numpy as np

from neurofisher.utils import compute_firing_rate


def SNR_bound_instantaneous(
    x: np.ndarray,
    C: np.ndarray,
    b: np.ndarray
) -> float:
    """Compute SNR about the latent trajectory using instantaneous Fisher information.

    How accurately can we estimate the latent state x from the spikes y of a log-linear Poisson model?

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

    # total power should be d_latent if normalized latents are used
    x_power = np.sum(np.mean(x ** 2, axis=0))
    d_latent = x.shape[1]
    assert d_latent == C.shape[0]

    invFI = 0.0
    for firing_rate in firing_rates:
        CC = C @ np.diag(firing_rate) @ C.T + np.eye(d_latent) * 1e-6
        invCC = np.linalg.inv(CC)
        invCC[invCC > 1e6] = 0
        invFI += np.trace(invCC)
    invFI = invFI / firing_rates.shape[0]  # average over time

    SNR_dB = 10 * np.log10(x_power / invFI)

    return SNR_dB
