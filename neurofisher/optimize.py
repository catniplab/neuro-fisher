"""Module for optimizing loading matrix of the observation model.

This module provides functions for optimizing the loading matrix of the observation model,
including computing coherence, and scaling loading matrices to achieve target signal-to-noise ratios.
"""

import numpy as np

from neurofisher.utils import bias_matching_firing_rate, safe_normalize


def compute_coherence(C):
    """Compute coherence of loading matrix.

    Coherence is defined as the maximum absolute value of the off-diagonal entries of the normalized correlation matrix
    of the columns of C. Mathematically, for a matrix C with columns c_i, the coherence μ is:
        μ(C) = max_{i ≠ j} |<c_i, c_j>| / (||c_i|| * ||c_j||)
    where <c_i, c_j> is the inner product between columns i and j, and ||c_i|| is the norm of column i.
    See: https://en.wikipedia.org/wiki/Coherence_(signal_processing)

    Parameters
    ----------
    C : ndarray
        Loading matrix

    Returns
    -------
    float
        Maximum off-diagonal correlation
    """
    C_norm = safe_normalize(C)
    CC = C_norm.T @ C_norm
    return np.max(np.abs(CC - np.diag(np.diag(CC))))


def project_l1ball(v, s=1):
    """Project vector onto L1-ball.

    Solves: min_w 0.5 * || w - v ||_2^2 , s.t. || w ||_1 <= s

    Parameters
    ----------
    v : ndarray
        Vector to project
    s : float, optional
        Ball radius, by default 1

    Returns
    -------
    ndarray
        Projected vector
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    u = np.abs(v)
    if u.sum() <= s:
        return v
    w = project_simplex(u, s=s)
    w *= np.sign(v)
    return w


def project_simplex(v, s=1):
    """Project vector onto positive simplex.

    Solves: min_w 0.5 * || w - v ||_2^2 , s.t. sum_i w_i = s, w_i >= 0

    Parameters
    ----------
    v : ndarray
        Vector to project
    s : float, optional
        Simplex radius, by default 1

    Returns
    -------
    ndarray
        Euclidean projection of v on the simplex

    References
    ----------
    [1] Duchi, John, et al. "Efficient projections onto the L1-ball for learning in high dimensions."
        Proceedings of the 25th international conference on Machine learning. 2008.
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    if v.sum() == s and np.all(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w


def adjust_gain(x, C, b, current_gain, tgt_rate_per_bin, max_rate_per_bin):
    """Limit firing rate to max_rate_per_bin.

    Parameters
    ----------
    x : ndarray
        Latent trajectory
    C : ndarray
        Loading matrix
    b : ndarray
        Bias vector
    tgt_rate_per_bin : float
        Target firing rate per bin
    max_rate_per_bin : float
        Maximum firing rate per bin

    Returns
    -------
    ndarray
        Mask of neurons to limit
    """
    Cx = (x @ C).max(axis=0)
    C = C * current_gain
    b, firing_rates = bias_matching_firing_rate(
        x, C, b, tgt_rate=tgt_rate_per_bin)

    adjusted_idx = firing_rates.max(axis=0) > max_rate_per_bin
    adjusted_gain = np.ones(C.shape[1]) * current_gain
    adjusted_gain[adjusted_idx] = (
        current_gain
        + np.log(max_rate_per_bin / firing_rates.max(axis=0)[adjusted_idx])
        / Cx[adjusted_idx]
    )

    return adjusted_gain, adjusted_idx


def optimize_C(
    x,
    C,
    b,
    tgt_rate_per_bin,
    max_rate_per_bin,
    tgt_snr,
    snr_fn,
    priority="max",
    max_iter=40,
    tol=0.1,
    min_gain=0.5,
    max_gain=1.0,
    verbose=False,
):
    """Uniformly scale the loading matrix to match the target SNR using bisection search.

    Args:
        x: The latent trajectory matrix
        C: The loading matrix to scale
        b: The bias vector
        tgt_rate: Target firing rate per bin
        tgt_snr: Target signal-to-noise ratio in dB
        snr_fn: Function to compute SNR
        max_rate: Maximum firing rate
        max_iter: Maximum number of bisection iterations
        tol: Relative tolerance for SNR matching (e.g., 0.1 means 10% tolerance)
        min_gain: Initial minimum gain for search
        max_gain: Initial maximum gain for search
        verbose: Whether to print debug information

    Returns:
        Tuple of (scaled_C, updated_b, achieved_snr)

    Raises:
        ValueError: If tgt_snr is invalid or if search fails to converge
    """
    if tol <= 0.0 or tol >= 1.0:
        raise ValueError("Tolerance must be between 0 and 1")

    # Initial bounds for gain
    curr_min_gain = min_gain
    curr_max_gain = max_gain

    prev_snr = -float("inf")
    # Find initial bounds that contain the solution
    for _ in range(10):  # Limit initial search iterations
        try:
            b_tmp, _ = bias_matching_firing_rate(x, C * curr_max_gain,
                                                 b, tgt_rate=tgt_rate_per_bin)
            snr = snr_fn(x, C * curr_max_gain, b_tmp)
            if snr > tgt_snr or snr < prev_snr:
                break
        except np.linalg.LinAlgError:
            curr_max_gain *= 0.8
            break
        curr_max_gain *= 2.0
        prev_snr = snr

    prev_snr = float("inf")
    for _ in range(10):  # Limit initial search iterations
        try:
            b_tmp, _ = bias_matching_firing_rate(x, C * curr_max_gain,
                                                 b, tgt_rate=tgt_rate_per_bin)
            snr = snr_fn(x, C * curr_min_gain, b_tmp)
            if snr < tgt_snr or snr > prev_snr:
                break
        except np.linalg.LinAlgError:
            pass  # If singular, try smaller gain
        curr_min_gain *= 0.5
        prev_snr = snr

    # Bisection search
    best_snr = float("inf")
    best_gain = None
    best_b = None
    curr_b = b
    for i in range(max_iter):
        # curr_gain = (curr_min_gain + curr_max_gain) / 2
        curr_gain = np.sqrt(curr_min_gain * curr_max_gain)
        try:
            adjusted_gain, adjusted_idx = adjust_gain(
                x, C, curr_b, curr_gain, tgt_rate_per_bin, max_rate_per_bin
            )
            curr_C = C * adjusted_gain
            curr_b, _ = bias_matching_firing_rate(
                x, curr_C, b, tgt_rate=tgt_rate_per_bin)
            snr = snr_fn(x, curr_C, curr_b)
            if priority == "max":
                recalibrated_gain, _ = adjust_gain(
                    x, curr_C, curr_b, 1, tgt_rate_per_bin, max_rate_per_bin
                )
                adjusted_gain = adjusted_gain * recalibrated_gain

            if verbose:
                print(
                    f"SNR: {snr:.2f} dB, Gain: {curr_gain:.2f}, Adjusted neurons: {adjusted_idx.sum()}"
                )
            # Keep track of best solution
            if abs(snr - tgt_snr) < abs(best_snr - tgt_snr):
                best_snr = snr
                best_gain = adjusted_gain
                best_b = curr_b

            # Check for convergence
            rel_err = abs(snr - tgt_snr) / abs(tgt_snr)
            if rel_err <= tol:
                if verbose:
                    print(
                        f"Converged after {i + 1} iterations with relative error {rel_err:.2%}"
                    )
                return C * adjusted_gain, curr_b, snr

            # Update search bounds
            curr_min_gain = curr_gain
        except np.linalg.LinAlgError:
            # If singular, try smaller gain
            curr_max_gain = curr_gain
            continue

        # Check if search bounds are too close
        if (curr_max_gain - curr_min_gain) / curr_min_gain < 1e-6:
            if verbose:
                print(f"Search bounds converged after {i + 1} iterations")
            break
        if adjusted_gain.max() < curr_gain:
            curr_max_gain = adjusted_gain.max()

    # If we didn't find a solution within tolerance, use the best one found
    if best_gain is not None:
        print(
            f"Could not find solution for target SNR {tgt_snr} dB, using best solution found: SNR = {best_snr:.2f} dB"
        )
        if priority == "mean":
            best_b, _ = bias_matching_firing_rate(
                x, C * best_gain, best_b, tgt_rate_per_bin)

        return C * best_gain, best_b, best_snr

    raise ValueError(f"Failed to find solution for target SNR {tgt_snr} dB")


def initialize_C(d_latent, d_neurons, p_coh, p_sparse=0.0, C=None):
    """Generate loading matrix with controllable coherence and sparsity.

    Parameters
    ----------
    d_latent : int
        Latent dimension
    d_neurons : int
        Number of neurons
    p_coh : float
        Target coherence
    p_sparse : float, optional
        Sparsity probability, by default 0
    C : ndarray, optional
        Initial matrix, by default None

    Returns
    -------
    ndarray
        Generated loading matrix
    """
    if C is None:
        C = np.random.randn(d_latent, d_neurons)
        C = C * (np.random.rand(d_latent, d_neurons) > p_sparse)
        C = safe_normalize(C)
        C[np.isnan(C)] = 0

    n_iter = 15
    n_inner = 1000
    rho = 0.5
    eta = 1.1
    lbda = 0.9
    alpha = lbda * rho

    for _ in range(n_iter):
        for _ in range(n_inner):
            coh = compute_coherence(C)
            if coh < p_coh:
                C = safe_normalize(C)
                C[np.isnan(C)] = 0
                return C

            vv = (C.T @ C - np.eye(d_neurons)) / rho
            v = project_l1ball(vv.flatten(), s=1)
            v_mat = np.reshape(v, (d_neurons, d_neurons))

            mm = C - alpha * C @ (v_mat + v_mat.T)
            C = safe_normalize(mm)
            C[np.isnan(C)] = 0
        rho = rho / eta
        alpha = lbda * rho

    if coh >= p_coh:
        print(
            f"WARNING: target Coherence {p_coh} not reached, Current Coherence {coh}")

    C = safe_normalize(C)
    C[np.isnan(C)] = 0
    return C
