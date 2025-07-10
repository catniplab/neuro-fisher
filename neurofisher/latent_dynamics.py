"""Module for generating latent neural trajectories.

This module provides functions for generating different types of latent trajectories.
"""

# Module for generating latent trajectory

import numpy as np


def generate_gp_trajectory(time_range, d_latent, lengthscale=1.0):
    """Generate latent trajectory using Gaussian Process with RBF kernel.

    Parameters
    ----------
    n_timepoints : int
        Number of time points
    d_latent : int
        Number of latent dimensions
    lengthscale : float, optional
        Length scale parameter for RBF kernel, by default 1.0

    Returns
    -------
    ndarray
        Normalized latent trajectory of shape (n_timepoints, d_latent)
    """
    # Compute RBF kernel matrix
    t1, t2 = np.meshgrid(time_range, time_range)
    K = np.exp(-0.5 * ((t1 - t2) / lengthscale) ** 2)

    # Add small jitter for numerical stability
    K += 1e-6 * np.eye(time_range.shape[0])

    # Generate samples from multivariate normal
    L = np.linalg.cholesky(K)
    latent_trajectory = np.dot(
        L, np.random.randn(time_range.shape[0], d_latent))

    # Normalize
    latent_trajectory -= np.mean(latent_trajectory, axis=0)
    latent_trajectory = latent_trajectory / np.std(latent_trajectory, axis=0)

    return latent_trajectory


def generate_oscillation_trajectory(time_range, w=1.0):
    """Generate latent trajectory from a perfect oscillation.

    Parameters
    ----------
    time_range : ndarray
        Time range
    w : float, optional
        Angular frequency, by default 1.0

    Returns
    -------
    ndarray
        Latent trajectory of shape (n_timepoints, d_latent)
    """
    # Generate random angles
    latent_trajectory = np.vstack(
        [np.sin(2 * np.pi * w * time_range),
         np.cos(2 * np.pi * w * time_range)]
    ).T

    latent_trajectory = latent_trajectory / np.std(latent_trajectory, axis=0)

    return latent_trajectory
