"""Utility functions for neural data processing."""

import numpy as np


def compute_firing_rate(x, C, b):
    """Compute firing rate of a log-linear Poisson neuron model.

    Parameters
    ----------
    x : ndarray
        Input matrix
    C : ndarray
        Loading matrix
    b : ndarray
        Bias vector

    Returns
    -------
    ndarray
        Firing rates
    """
    # Clip values to prevent overflow
    log_rate = x @ C + b
    log_rate = np.clip(log_rate, -700, 700)
    return np.exp(log_rate)


def update_bias(curr_rate, curr_b, tgt_rate=0.05):
    """Update bias to match target firing rate.

    Parameters
    ----------
    curr_rate : ndarray
        Current firing rates
    curr_b : ndarray
        Current bias
    tgt_rate : float, optional
        Target firing rate, by default 0.05

    Returns
    -------
    ndarray
        Updated bias
    """
    assert curr_b.size == curr_rate.size
    # Handle zero rates
    mask = curr_rate > 0
    new_b = curr_b.copy()
    # Ensure we're working with 1D arrays for indexing
    new_b = new_b.ravel()
    curr_b = curr_b.ravel()
    curr_rate = curr_rate.ravel()
    new_b[mask] = curr_b[mask] + np.log(tgt_rate / curr_rate[mask])
    # Reshape back to original shape
    return new_b.reshape(curr_b.shape)


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
