import numpy as np

from neurofisherSNR.utils import powerDb_to_R2


def power_to_dB(power_ratio):
    """Convert power ratio to dB: SNR_dB = 10 * log10(power_ratio)"""
    return 10 * np.log10(power_ratio)


def test_scalar_cases():
    test_cases = [
        (1.0, 0.0, "1:1 ratio"),
        (10.0, 10.0, "10:1 ratio"),
        (100.0, 20.0, "100:1 ratio"),
        (0.1, -10.0, "1:10 ratio"),
        (0.01, -20.0, "1:100 ratio"),
        (2.0, 3.010299956639812, "2:1 ratio"),
        (0.5, -3.010299956639812, "1:2 ratio"),
    ]
    for power_ratio, expected_dB, desc in test_cases:
        computed_dB = power_to_dB(power_ratio)
        error_msg = f"Failed {desc}: {power_ratio} → {computed_dB} dB (expected {expected_dB} dB)"
        assert np.abs(computed_dB - expected_dB) < 1e-10, error_msg


def test_vector_cases():
    # Test 1D arrays
    power_ratios = np.array([0.1, 1.0, 10.0, 100.0, 1000.0])
    expected_dB = np.array([-10.0, 0.0, 10.0, 20.0, 30.0])
    computed_dB = power_to_dB(power_ratios)
    assert np.allclose(computed_dB, expected_dB,
                       atol=1e-10), f"1D array test failed: {computed_dB} != {expected_dB}"
    # Test 2D array
    power_2d = np.array([[1.0, 10.0], [0.1, 100.0]])
    expected_2d = np.array([[0.0, 10.0], [-10.0, 20.0]])
    computed_2d = power_to_dB(power_2d)
    assert np.allclose(computed_2d, expected_2d,
                       atol=1e-10), f"2D array test failed: {computed_2d} != {expected_2d}"


def test_edge_cases():
    edge_cases = [
        (1e-12, -120.0),
        (1e-6, -60.0),
        (1e6, 60.0),
        (1e12, 120.0)
    ]
    for power_ratio, expected_dB in edge_cases:
        computed_dB = power_to_dB(power_ratio)
        error_msg = f"Edge case failed: {power_ratio} → {computed_dB} dB (expected {expected_dB} dB)"
        assert np.abs(computed_dB - expected_dB) < 1e-10, error_msg


def test_snr_formula():
    x_power = np.array([1.0, 2.0, 5.0])
    invFI = np.array([0.1, 0.5, 1.0])
    power_ratios = x_power / invFI
    snr_dB = power_to_dB(power_ratios)
    expected_snr = np.array([10.0, 6.0206, 6.9897])
    assert np.allclose(snr_dB, expected_snr,
                       atol=1e-3), f"SNR formula test failed: {snr_dB} != {expected_snr}"


def test_powerDb_to_R2_vector():
    """Test converting dB values to R² values for vector inputs."""
    # Test cases: 0dB, -40dB, 20dB
    snr_dB = np.array([0.0, -40.0, 20.0])

    # Expected R² values:
    # 0dB → SNR_linear = 1 → R² = 1 - 1/1 = 0
    # -40dB → SNR_linear = 10^(-4) = 0.0001 → R² = 1 - 1/0.0001 = 1 - 10000 = -9999
    # 20dB → SNR_linear = 10^2 = 100 → R² = 1 - 1/100 = 0.99
    expected_R2 = np.array([0.0, -9999.0, 0.99])

    computed_R2 = powerDb_to_R2(snr_dB)

    # Check that the computed R² values match expected values
    assert np.allclose(computed_R2, expected_R2, atol=1e-10), (
        f"R² conversion failed: {computed_R2} != {expected_R2}"
    )
