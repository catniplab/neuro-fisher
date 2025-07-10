import numpy as np
import pytest
from neurofisherSNR.snr import SNR_bound_instantaneous
from neurofisherSNR.optimize import initialize_C


class TestSNRBoundInstantaneous:
    """Test class for checking input and output dimensionality of SNR_bound_instantaneous function."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        # Set random seed for reproducible tests
        np.random.seed(20250710)

        # Test parameters
        self.time_steps = 100
        self.d_latent = 3
        self.d_neurons = 10
        self.tgt_rate_per_bin = 0.1
        self.max_rate_per_bin = 1.0
        self.tgt_snr = 5.0

        # Create test data
        self.x = np.random.randn(self.time_steps, self.d_latent)
        self.C = initialize_C(self.d_latent, self.d_neurons, p_coh=0.5, p_sparse=0.0)
        self.b = np.random.randn(1, self.d_neurons)

    def test_snr_bound_instantaneous_simple(self):
        snr = SNR_bound_instantaneous(self.x, self.C.T, self.b)
        assert np.isfinite(snr), "SNR should be finite."
        assert snr > 0, "SNR should be positive for this input."
