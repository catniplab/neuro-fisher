"""Module for generating latent neural trajectories.

This module provides functions for generating different types of latent trajectories.
"""

# Module for generating latent trajectory

import numpy as np


def generate_gp_trajectory(
    time_range: np.ndarray, d_latent: int, lengthscale: float = 1.0
) -> np.ndarray:
    """Generate latent trajectory using Gaussian Process with RBF kernel.
    Parameters
    ----------
    time_range : ndarray
        1D array of time points (n_timepoints,)
    d_latent : int
        Number of latent dimensions
    lengthscale : float, optional
        Length scale parameter for RBF kernel, by default 1.0
    Returns
    -------
    ndarray
        Normalized latent trajectory of shape (n_timepoints, d_latent)
    """
    assert time_range.ndim == 1, "time_range must be 1D array"
    assert (
        isinstance(d_latent, int) and d_latent > 0
    ), "d_latent must be positive integer"
    assert isinstance(lengthscale, float) or isinstance(
        lengthscale, int
    ), "lengthscale must be float or int"
    n_timepoints = time_range.shape[0]
    # Compute RBF kernel matrix
    t1, t2 = np.meshgrid(time_range, time_range)
    K = np.exp(-0.5 * ((t1 - t2) / lengthscale) ** 2)
    # Add small jitter for numerical stability
    K += 1e-6 * np.eye(n_timepoints)
    # Generate samples from multivariate normal
    L = np.linalg.cholesky(K)
    x = np.dot(L, np.random.randn(n_timepoints, d_latent))
    # Normalize
    x -= np.mean(x, axis=0)
    x = x / np.std(x, axis=0)
    assert x.shape == (n_timepoints, d_latent), "Output shape mismatch"
    return x


def generate_oscillation_trajectory(
    time_range: np.ndarray, w: float = 1.0
) -> np.ndarray:
    """Generate latent trajectory from a perfect oscillation.
    Parameters
    ----------
    time_range : ndarray
        1D array of time points (n_timepoints,)
    w : float, optional
        Angular frequency, by default 1.0
    Returns
    -------
    ndarray
        Latent trajectory of shape (n_timepoints, 2)
    """
    assert time_range.ndim == 1, "time_range must be 1D array"
    assert isinstance(w, float) or isinstance(w, int), "w must be float or int"
    n_timepoints = time_range.shape[0]
    # Generate random angles
    latent_trajectory = np.vstack(
        [np.sin(2 * np.pi * w * time_range), np.cos(2 * np.pi * w * time_range)]
    ).T
    latent_trajectory = latent_trajectory / np.std(latent_trajectory, axis=0)
    assert latent_trajectory.shape == (n_timepoints, 2), "Output shape mismatch"
    return latent_trajectory
