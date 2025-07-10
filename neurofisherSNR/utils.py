"""Utility functions for neural data processing."""

import numpy as np
from typing import Tuple


def compute_firing_rate(x: np.ndarray, CT: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute firing rate of a log-linear Poisson neuron model.

    Parameters
    ----------
    x : ndarray
        Input matrix (n_timepoints, d_latent)
    CT : ndarray
        Loading matrix (d_latent, d_neurons)
    b : ndarray
        Bias vector (1, d_neurons)

    Returns
    -------
    ndarray
        Firing rates (n_timepoints, d_neurons)
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
    return np.exp(x @ CT + b)


def bias_matching_firing_rate(
    x: np.ndarray, CT: np.ndarray, b: np.ndarray, tgt_rate: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Update bias to match target firing rate.

    Parameters
    ----------
    x : ndarray
        Input matrix (n_timepoints, d_latent)
    CT : ndarray
        Loading matrix (d_latent, d_neurons)
    b : ndarray
        Bias vector (1, d_neurons)
    tgt_rate : float, optional
        Target firing rate, by default 0.05

    Returns
    -------
    tuple
        Updated bias, updated firing rates
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
    mean_rate = compute_firing_rate(x, CT, b).mean(axis=0)
    assert b.size == mean_rate.size
    # Handle zero rates
    mask = mean_rate > 0
    b[0, mask] = b[0, mask] + np.log(tgt_rate / mean_rate[mask])
    # Reshape back to original shape
    new_rates = compute_firing_rate(x, CT, b)
    return b, new_rates


def safe_normalize(C: np.ndarray, axis: int = 0) -> np.ndarray:
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
