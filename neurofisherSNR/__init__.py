"""NeuroFisher: A package for generating log-linear Poisson neural observations."""

# SNR computation
# Latent dynamics generation
from .latent_dynamics import (generate_gp_trajectory,
                              generate_oscillation_trajectory)
# Poisson observations generation
from .observation import gen_poisson_observations
# Optimization
from .optimize import optimize_C
from .snr import SNR_bound_instantaneous
# Utility functions
from .utils import power_from_dB, power_to_dB, powerDb_to_R2

__version__ = "0.2"

__all__ = [
    "SNR_bound_instantaneous",
    "optimize_C",
    "generate_gp_trajectory",
    "generate_oscillation_trajectory",
    "gen_poisson_observations",
    "power_to_dB",
    "power_from_dB",
    "powerDb_to_R2",
]
