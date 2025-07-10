"""Utility functions for neural data processing."""

import numpy as np


def compute_firing_rate(x, CT, b):
    """Compute firing rate of a log-linear Poisson neuron model.

    Parameters
    ----------
    x : ndarray
        Input matrix
    CT : ndarray
        Loading matrix
    b : ndarray
        Bias vector

    Returns
    -------
    ndarray
        Firing rates
    """
    return np.exp(x @ CT + b)


def bias_matching_firing_rate(x, CT, b, tgt_rate=0.05):
    """Update bias to match target firing rate.

    Parameters
    ----------
    rates : ndarray
        Firing rates
    b : ndarray
        Bias vector
    tgt_rate : float, optional
        Target firing rate, by default 0.05

    Returns
    -------
    ndarray
        Updated bias, updated firing rates
    """
    mean_rate = compute_firing_rate(x, CT, b).mean(axis=0)
    assert b.size == mean_rate.size
    # Handle zero rates
    mask = mean_rate > 0
    b[..., mask] = b[..., mask] + np.log(tgt_rate / mean_rate[mask])

    # Reshape back to original shape
    new_rates = compute_firing_rate(x, CT, b)
    return b, new_rates


def safe_normalize(C, axis=0):
    """Safely normalize matrix along specified axis.

    Parameters
    ----------
    C : ndarray
        Matrix to normalize
    axis : int, optional
        Axis to normalize along, by default 0

    Returns
    -------
    ndarray
        Normalized matrix
    """
    norm = np.linalg.norm(C, axis=axis)
    # Handle zero columns
    mask = norm > 0
    C_norm = np.zeros_like(C)
    C_norm[:, mask] = C[:, mask] / norm[mask]
    return C_norm
