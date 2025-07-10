# Generating log-linear Poisson observations from 2D oscillation latent trajectory given Fisher Information SNR bound
import numpy as np

from neurofisher.latent_dynamics import generate_oscillation_trajectory
from neurofisher.observation import gen_poisson_observations
from neurofisher.snr import SNR_bound_instantaneous
from neurofisher.vis_utils import plot_spike_train

np.random.seed(2)

# Default parameters
time_duration = 10  # seconds
num_steps = 1000
time_range = np.linspace(0, time_duration, num_steps)
dt = time_range[1] - time_range[0]

# Generating Ring latent trajectory
latent_trajectory = generate_oscillation_trajectory(
    time_range=time_range,
    w=1.0
)

# Generate observations
p_sparse = 0.1
p_coherence = 1
target_rate_per_bin = 20.0 * dt  # (target_firing_rate in Hz * dt)
target_snr = 3.0  # dB
num_neurons = 10

observations, C, bias, firing_rate_per_bin, snr = gen_poisson_observations(
    x=latent_trajectory,
    C=None,
    d_neurons=num_neurons,
    tgt_rate_per_bin=target_rate_per_bin,
    max_rate_per_bin=100 * dt,
    p_coh=p_coherence,
    p_sparse=p_sparse,
    tgt_snr=target_snr,
    snr_fn=SNR_bound_instantaneous,
    priority="mean",
    verbose=True,
)

# Plotting observations with latent trajectory
plot_spike_train(
    x=latent_trajectory,
    y=observations,
    num_neurons=num_neurons,
    dt=dt,
    time_range=time_range,
    firing_rates=firing_rate_per_bin / dt,  # Hz
    output_filename="oscillation_spike_train.png",
)
