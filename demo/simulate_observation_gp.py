# %%
# Generating log-linear Poisson observations from GP latent trajectory given Fisher Information SNR bound
import numpy as np
from neurofisher.latent_dynamics import generate_gp_trajectory
from neurofisher.observation import gen_poisson_observations
from neurofisher.vis_utils import *


# Default parameters
time_duration = 10
num_steps = 1000
time_range = np.linspace(0, time_duration, num_steps)
d_latent = 5
dt = time_range[1] - time_range[0]

# Generating GP latent trajectory
latent_trajectory = generate_gp_trajectory(
    time_range=time_range,
    d_latent=d_latent,
    lengthscale=0.5,
)

# Generate observations
p_sparse = 0.1
p_coherence = 1
target_rate = 20.0 * dt
target_snr = 5.0  # dB
num_neurons = 50

observations, loading_matrix, bias, firing_rate_per_bin, snr = gen_poisson_observations(
    x=latent_trajectory,
    C=None,
    d_neurons=num_neurons,
    tgt_rate=target_rate,
    p_coh=p_coherence,
    p_sparse=p_sparse,
    tgt_snr=target_snr,
    snr_fn=compute_instantaneous_snr,
)

# Plotting observations with latent trajectory
num_neurons = 10
plot_spike_train(
    x=latent_trajectory,
    y=observations,
    num_neurons=num_neurons,
    dt=dt,
    time_range=time_range,
    firing_rates=firing_rate_per_bin,
    output_filename=None,
)
