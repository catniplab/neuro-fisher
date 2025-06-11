# %%
# Generating log-linear Poisson observations from 2D Ring latent trajectory given Fisher Information SNR bound

from neurofisher.visualize import *
from neurofisher.simulate import *

import numpy as np


# Default parameters
time_duration = 10
num_steps = 1000
time_range = np.linspace(0, time_duration, num_steps)
dt = time_range[1] - time_range[0]

# Generating Ring latent trajectory
latent_trajectory = np.vstack(
    [np.sin(2 * np.pi * time_range), np.cos(2 * np.pi * time_range)]
).T
latent_trajectory = latent_trajectory / np.std(latent_trajectory, axis=0)

# Generate observations
p_sparse = 0.1
p_coherence = 1
target_rate = 20.0 * dt
target_snr = 5.0  # dB
num_neurons = 10

observations, loading_matrix, bias, firing_rate_per_bin, snr = gen_poisson(
    x=latent_trajectory,
    C=None,
    d_neurons=num_neurons,
    tgt_rate=target_rate,
    p_coh=p_coherence,
    p_sparse=p_sparse,
    tgt_snr=target_snr,
    snr_fn=comp_instantaneous_snr,
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
