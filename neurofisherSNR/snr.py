"""Module for computing signal-to-noise ratios (SNR)

This module provides functions for computing SNR using Fisher information.
"""

from typing import Any

import numpy as np

from neurofisherSNR.utils import compute_firing_rate, power_to_dB


def SNR_bound_instantaneous(x: np.ndarray, CT: np.ndarray, b: np.ndarray) -> float:
    """Compute SNR about the latent trajectory using instantaneous Fisher information.

    How accurately can we estimate the latent state x from the spikes y of a log-linear Poisson model?

    Parameters
    ----------
    x : ndarray
        Latent trajectory (zero-mean, shape (n_timepoints, d_latent))
    CT : ndarray
        Loading matrix (d_latent, d_neurons)
    b : ndarray
        Bias vector (1, d_neurons)

    Returns
    -------
    float
        SNR bound in dB
    """
    assert (
        isinstance(x, np.ndarray) and x.ndim == 2
    ), "x must be 2D ndarray (n_timepoints, d_latent)"
    assert (
        isinstance(CT, np.ndarray) and CT.ndim == 2
    ), "CT must be 2D ndarray (d_latent, d_neurons)"
    if isinstance(b, np.ndarray) and b.ndim == 1:
        b = b.reshape(1, -1)
    assert (
        isinstance(b, np.ndarray) and b.ndim == 2 and b.shape[0] == 1
    ), "b must be 2D ndarray (1, d_neurons)"
    firing_rates = compute_firing_rate(x, CT, b)
    # total power should be d_latent if normalized latents are used
    x_power = np.sum(np.mean(x**2, axis=0))
    d_latent = x.shape[1]
    assert d_latent == CT.shape[0]

    invFI = 0.0
    for firing_rate in firing_rates:
        CC = CT @ np.diag(firing_rate) @ CT.T + np.eye(d_latent) * 1e-6
        invCC = np.linalg.inv(CC)
        invCC[invCC > 1e6] = 0
        invFI += np.trace(invCC)
    invFI = invFI / firing_rates.shape[0]  # average over time
    SNR_dB = power_to_dB(x_power / invFI)

    return SNR_dB
