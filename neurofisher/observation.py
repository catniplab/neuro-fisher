"""Module for generating log-linear Poisson neural observations.

This module provides functionality for generating Poisson-distributed neural observations
from latent trajectories with target firing rates and signal-to-noise ratios (SNR).
"""

from typing import Callable, Optional

import numpy as np

from neurofisher.optimize import initialize_C, optimize_C
from neurofisher.snr import SNR_bound_instantaneous
from neurofisher.utils import bias_matching_firing_rate, compute_firing_rate


def gen_poisson_observations(
    x: np.ndarray,
    C: Optional[np.ndarray] = None,
    d_neurons: int = 100,
    tgt_rate_per_bin: float = 0.01,
    max_rate_per_bin: float = 1,
    priority: str = "mean",
    p_coh: float = 0.5,
    p_sparse: float = 0.1,
    tgt_snr: float = 10.0,
    snr_fn: Callable = SNR_bound_instantaneous,
    verbose: bool = False,
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
    tgt_rate_per_bin : float, optional
        Target mean firing rate per bin, by default 0.01
    max_rate_per_bin : float, optional
        Maximum firing rate per bin, by default 1
    priority : str, optional
        Priority for the optimization: "mean" or "max", by default "mean"
    p_coh : float, optional
        Coherence, by default 0.5
    p_sparse : float, optional
        Sparsity, by default 0.1
    tgt_snr : float, optional
        Target SNR, by default 10.0
    snr_fn : callable, optional
        SNR function, by default compute_instantaneous_snr
    verbose : bool, optional
        Whether to print debug information, by default False
    Returns
    -------
    tuple
        (observations, C, b, rates, snr)
    """
    assert p_sparse >= 0.0, "p_sparse must be between 0 and 1"
    assert p_sparse <= 1.0, "p_sparse must be between 0 and 1"
    assert p_coh >= np.sqrt(
        (d_neurons - x.shape[1]) / (x.shape[1] * (d_neurons - 1))
    ), (
        f"p_coh must be greater than sqrt((d_neurons - d_latent) / "
        f"(d_latent * (d_neurons - 1))) = "
        f"{np.sqrt((d_neurons - x.shape[1]) / (x.shape[1] * (d_neurons - 1))):.2f}"
    )
    assert d_neurons > 0, "d_neurons must be positive"
    assert tgt_rate_per_bin > 0.0, "tgt_rate_per_bin must be positive"
    assert max_rate_per_bin > 0.0, "max_rate_per_bin must be positive"

    if not np.all(np.isclose(np.std(x, axis=0), 1)):
        print("WARNING: latent trajectory must have unit variance. Normalizing...")
        x = x / np.std(x, axis=0)

    d_latent = x.shape[1]
    if C is None:
        CT = initialize_C(d_latent, d_neurons, p_coh, p_sparse)
    else:
        CT = C.T
        assert (
            CT.shape[0] == x.shape[1]
        ), "Loading matrix must have same number of rows as latent trajectory dimensions"
        assert (
            CT.shape[1] == d_neurons
        ), "Loading matrix must have same number of columns as number of neurons"

    # b = 1.0 * np.random.rand(1, d_neurons) - np.log(tgt_rate_per_bin)
    b = np.zeros((1, d_neurons))
    rates = compute_firing_rate(x, CT, b)
    b, rates = bias_matching_firing_rate(x, CT, b, tgt_rate=tgt_rate_per_bin)

    CT, b, snr = optimize_C(
        x=x,
        CT=CT,
        b=b,
        tgt_rate_per_bin=tgt_rate_per_bin,
        max_rate_per_bin=max_rate_per_bin,
        tgt_snr=tgt_snr,
        snr_fn=snr_fn,
        priority=priority,
        verbose=verbose,
    )
    rates = compute_firing_rate(x, CT, b)
    obs = np.random.poisson(rates)

    return obs, CT.T, b, rates, snr
