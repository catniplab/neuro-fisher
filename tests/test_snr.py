import numpy as np
import pytest
from neurofisherSNR.snr import SNR_bound_instantaneous


def test_snr_bound_instantaneous_simple():
    x = np.eye(2)
    CT = np.ones((2, 2))
    b = np.zeros(2)
    snr = SNR_bound_instantaneous(x, CT, b)
    assert np.isfinite(snr), "SNR should be finite."
    assert snr > 0, "SNR should be positive for this input."
