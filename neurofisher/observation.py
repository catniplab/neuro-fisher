"""Module for generating log-linear Poisson neural observations.

This module provides functionality for generating Poisson-distributed neural observations
from latent trajectories with target firing rates and signal-to-noise ratios (SNR).
"""

import numpy as np
from neurofisher.optimize import initialize_C, optimize_C
from neurofisher.snr import compute_instantaneous_snr
from neurofisher.utils import compute_firing_rate


def gen_poisson_observations(
    x,
    C=None,
    d_neurons=100,
    tgt_rate=0.01,
    p_coh=0.5,
    p_sparse=0.1,
    tgt_snr=10.0,
    snr_fn=compute_instantaneous_snr,
):
    """Generate Poisson observations with controlled SNR and firing rate.

    Parameters
    ----------
    x : ndarray
        Latent trajectory
    C : ndarray, optional
        Loading matrix, by default None
    d_neurons : int, optional
        Number of neurons, by default 100
    tgt_rate : float, optional
        Target firing rate, by default 0.01
    p_coh : float, optional
        Coherence, by default 0.5
    p_sparse : float, optional
        Sparsity, by default 0.1
    tgt_snr : float, optional
        Target SNR, by default 10.0
    snr_fn : callable, optional
        SNR function, by default comp_snr

    Returns
    -------
    tuple
        (observations, C, b, rates, snr)
    """
    assert p_sparse >= 0, "p_sparse must be between 0 and 1"
    assert p_sparse <= 1, "p_sparse must be between 0 and 1"
    assert p_coh >= np.sqrt(
        (d_neurons - x.shape[1]) / (x.shape[1] * (d_neurons - 1))
    ), f"p_coh must be greater than sqrt((d_neurons - d_latent) / (d_latent * (d_neurons - 1))) = {np.sqrt((d_neurons - x.shape[1]) / (x.shape[1] * (d_neurons - 1))):.2f}"
    assert d_neurons > 0, "d_neurons must be positive"
    assert tgt_rate > 0, "tgt_rate must be positive"

    if not np.all(np.isclose(np.std(x, axis=0), 1)):
        print("WARNING: latent trajectory must have unit variance. Normalizing...")
        x = x / np.std(x, axis=0)

    d_latent = x.shape[1]
    if C is None:
        C = initialize_C(d_latent, d_neurons, p_coh, p_sparse)
    else:
        assert (
            C.shape[0] == x.shape[1]
        ), "Loading matrix must have same number of rows as latent trajectory dimensions"
        assert (
            C.shape[1] == d_neurons
        ), "Loading matrix must have same number of columns as number of neurons"

    b = 1.0 * np.random.rand(1, d_neurons) - np.log(tgt_rate)
    C, b, snr = optimize_C(x, C, b, tgt_rate, tgt_snr=tgt_snr, snr_fn=snr_fn)
    rates = compute_firing_rate(x, C, b)

    # Clip rates to prevent overflow in Poisson sampling
    max_rate = 1e9  # Maximum rate that NumPy's Poisson can handle
    rates = np.clip(rates, 0, max_rate)

    obs = np.random.poisson(rates)

    return obs, C, b, rates, snr
