# %%
import numpy as np
import warnings
from scipy.linalg import lstsq


def comp_rate(x, C, b):
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
    log_rate = np.clip(log_rate, -700, 700)  # exp(700) is close to float64 max
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


def scale_C(
    x,
    C,
    b,
    tgt_rate,
    tgt_snr,
    snr_fn,
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
    if tol <= 0 or tol >= 1:
        raise ValueError("Tolerance must be between 0 and 1")

    # Initial bounds for gain
    curr_min_gain = min_gain
    curr_max_gain = max_gain

    # Find initial bounds that contain the solution
    for _ in range(10):  # Limit initial search iterations
        try:
            snr, _ = snr_fn(x, C * curr_max_gain, b, tgt_rate)
            if snr > tgt_snr:
                break
        except np.linalg.LinAlgError:
            pass  # If singular, try larger gain
        curr_max_gain *= 2.0

    for _ in range(10):  # Limit initial search iterations
        try:
            snr, _ = snr_fn(x, C * curr_min_gain, b, tgt_rate)
            if snr < tgt_snr:
                break
        except np.linalg.LinAlgError:
            pass  # If singular, try smaller gain
        curr_min_gain *= 0.5

    # Bisection search
    best_snr = float("inf")
    best_gain = None
    best_b = None

    for i in range(max_iter):
        curr_gain = (curr_min_gain + curr_max_gain) / 2
        try:
            snr, curr_b = snr_fn(x, C * curr_gain, b, tgt_rate)

            # Keep track of best solution
            if abs(snr - tgt_snr) < abs(best_snr - tgt_snr):
                best_snr = snr
                best_gain = curr_gain
                best_b = curr_b

            # Check for convergence
            rel_err = abs(snr - tgt_snr) / abs(tgt_snr)
            if rel_err <= tol:
                if verbose:
                    print(
                        f"Converged after {i + 1} iterations with relative error {rel_err:.2%}"
                    )
                return C * curr_gain, curr_b, snr

            # Update search bounds
            if snr > tgt_snr:
                curr_max_gain = curr_gain
            else:
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

    # If we didn't find a solution within tolerance, use the best one found
    if best_gain is not None:
        if verbose:
            print(
                f"Using best solution found: SNR = {best_snr:.2f} dB (target: {tgt_snr:.2f} dB)"
            )
        return C * best_gain, best_b, best_snr

    raise ValueError(f"Failed to find solution for target SNR {tgt_snr} dB")


def comp_instantaneous_snr(x, C, b, tgt_rate):
    """Compute SNR from instantaneous fisher information.

    Parameters
    ----------
    x : ndarray
        Latent trajectory (unit variance)
    C : ndarray
        Loading matrix
    b : ndarray
        Bias vector
    tgt_rate : float
        Target firing rate

    Returns
    -------
    tuple
        (SNR in dB, updated bias)
    """
    rates = comp_rate(x, C, b)
    b = update_bias(np.mean(rates, axis=0), b, tgt_rate=tgt_rate)
    rates = comp_rate(x, C, b)

    snr = 0
    reg = 1e-6  # Small regularization term to prevent singularity
    for i, rate in enumerate(rates):
        # Scale rates to prevent overflow
        rate_scale = np.max(rate)
        if rate_scale > 0:
            rate = rate / rate_scale
        # Add small regularization to diagonal to prevent singularity
        F = C @ np.diag(rate) @ C.T
        F = F + reg * np.eye(F.shape[0])
        try:
            snr += np.trace(np.linalg.inv(F)) / rate_scale if rate_scale > 0 else 0
        except np.linalg.LinAlgError:
            # If still singular, use pseudo-inverse
            snr += np.trace(np.linalg.pinv(F)) / rate_scale if rate_scale > 0 else 0
    snr = snr / rates.shape[0]
    snr = 10 * np.log10(C.shape[0] / snr)

    return snr, b


def comp_coherence(C):
    """Compute coherence of loading matrix.

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


def proj_l1ball(v, s=1):
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
    w = proj_simplex(u, s=s)
    w *= np.sign(v)
    return w


def proj_simplex(v, s=1):
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
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    (n,) = v.shape
    if v.sum() == s and np.alltrue(v >= 0):
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
    theta = (cssv[rho] - s) / (rho + 1.0)
    w = (v - theta).clip(min=0)
    return w


def gen_C(d_latent, d_neurons, p_coh, p_sparse=0, C=None):
    """Generate loading matrix with controlled coherence and sparsity.

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
            coh = comp_coherence(C)
            if coh < p_coh:
                C = safe_normalize(C)
                C[np.isnan(C)] = 0
                return C

            vv = (C.T @ C - np.eye(d_neurons)) / rho
            v = proj_l1ball(vv.flatten(), s=1)
            v_mat = np.reshape(v, (d_neurons, d_neurons))

            mm = C - alpha * C @ (v_mat + v_mat.T)
            C = safe_normalize(mm)
            C[np.isnan(C)] = 0
        rho = rho / eta
        alpha = lbda * rho

    if coh >= p_coh:
        print(f"WARNING: target Coherence {p_coh} not reached, Current Coherence {coh}")

    C = safe_normalize(C)
    C[np.isnan(C)] = 0
    return C


def gen_poisson(
    x,
    C=None,
    d_neurons=100,
    tgt_rate=0.01,
    p_coh=0.5,
    p_sparse=0.1,
    tgt_snr=10.0,
    snr_fn=comp_instantaneous_snr,
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
        C = gen_C(d_latent, d_neurons, p_coh, p_sparse)

    b = 1.0 * np.random.rand(1, d_neurons) - np.log(tgt_rate)
    C, b, snr = scale_C(x, C, b, tgt_rate, tgt_snr=tgt_snr, snr_fn=snr_fn)
    rates = comp_rate(x, C, b)

    # Clip rates to prevent overflow in Poisson sampling
    max_rate = 1e9  # Maximum rate that NumPy's Poisson can handle
    rates = np.clip(rates, 0, max_rate)

    obs = np.random.poisson(rates)

    return obs, C, b, rates, snr


def generate_gp_latent_trajectory(time_range, d_latent, lengthscale=1.0):
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
    latent_trajectory = np.dot(L, np.random.randn(time_range.shape[0], d_latent))

    # Normalize
    latent_trajectory -= np.mean(latent_trajectory, axis=0)
    latent_trajectory = latent_trajectory / np.std(latent_trajectory, axis=0)

    return latent_trajectory


if __name__ == "__main__":
    # Example usage
    x = np.random.randn(1000, 2)
    obs, C, b, rates, snr = gen_poisson(
        x,
        C=None,
        d_neurons=50,
        tgt_rate=0.1,
        p_coh=0.8,
        p_sparse=0.1,
        tgt_snr=10.0,
        snr_fn=comp_instantaneous_snr,
    )

    print("Generated observations shape:", obs.shape)
    print("Loading matrix shape:", C.shape)
    print("Bias shape:", b.shape)
    print("Firing rate per bin shape:", rates.shape)
    print("SNR:", snr)
