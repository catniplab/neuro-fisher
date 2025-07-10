"""Visualization utilities"""

# Auxiliary functions for visualizing the data

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Matplotlib settings
plt.rcParams["text.usetex"] = False
plt.rcParams["text.latex.preamble"] = r"\usepackage{amsmath}"
plt.rcParams["font.sans-serif"] = "Helvetica"
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.serif"] = "cmr10"
plt.rcParams["font.size"] = 12
plt.rcParams["pdf.fonttype"] = 42  # TrueType fonts


def plot_spike_train(
    x: np.ndarray,
    y: np.ndarray,
    num_neurons: int,
    dt: float,
    time_range: np.ndarray,
    firing_rates: np.ndarray,
    output_filename: Optional[str] = None,
):
    spike_times = [
        np.where(y[:, neuron_idx] > 0)[0] * dt for neuron_idx in range(num_neurons)
    ]

    fig, (spike_ax, latent_ax, rate_ax) = plt.subplots(
        3, 1, figsize=(12, 6), gridspec_kw={"height_ratios": [3, 1, 1]}
    )

    for neuron_idx, neuron_spikes in enumerate(spike_times):
        spike_ax.eventplot(
            neuron_spikes, colors="black", lineoffsets=neuron_idx, linelengths=0.8
        )

    spike_ax.set_ylim(-1, num_neurons)
    spike_ax.set_ylabel("Neurons")
    spike_ax.xaxis.set_ticklabels([])
    spike_ax.set_xlim(time_range[0], time_range[-1])

    latent_ax.plot(time_range, x)
    latent_ax.set_xlabel("Time (s)")
    latent_ax.set_ylabel("Latents")
    latent_ax.xaxis.set_ticklabels([])
    latent_ax.set_xlim(time_range[0], time_range[-1])

    rate_ax.plot(time_range, firing_rates)
    rate_ax.set_xlabel("Time (s)")
    rate_ax.set_ylabel("Firing Rate (Hz)")
    rate_ax.set_xlim(time_range[0], time_range[-1])

    plt.subplots_adjust(hspace=0.05)

    if output_filename is not None:
        plt.savefig(f"{output_filename}", dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_filename}")
        plt.close()
    else:
        return fig
