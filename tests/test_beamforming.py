import numpy as np

from src.beamforming import (half_power_beamwidth_deg, misalignment_deg,
                             mrt_weights, snr, snr_db, spectral_efficiency)


def _channel(n_tx=8, seed=0):
    rng = np.random.default_rng(seed)
    h = rng.normal(size=(1, n_tx, 1)) + 1j * rng.normal(size=(1, n_tx, 1))
    return h


def test_mrt_weights_have_unit_norm():
    w = mrt_weights(_channel())

    assert np.isclose(np.linalg.norm(w), 1.0)


def test_mrt_maximises_snr_over_any_other_unit_norm_beamformer():
    """MRT is the SNR-optimal single-user precoder; nothing should beat it."""
    H = _channel(n_tx=8, seed=1)
    best = snr(H, mrt_weights(H), noise_power=1.0)

    rng = np.random.default_rng(2)
    for _ in range(200):
        w = rng.normal(size=8) + 1j * rng.normal(size=8)
        w /= np.linalg.norm(w)
        assert snr(H, w, noise_power=1.0) <= best + 1e-9


def test_matched_beam_gives_channel_power_over_noise():
    H = _channel(n_tx=8, seed=3)
    h = H[0, :, 0]

    value = snr(H, mrt_weights(H), noise_power=0.5)

    assert np.isclose(value, np.sum(np.abs(h) ** 2) / 0.5)


def test_spectral_efficiency_is_shannon_capacity():
    assert np.isclose(spectral_efficiency(3.0), np.log2(4.0))
    assert np.isclose(snr_db(100.0), 20.0)


def test_beamwidth_halves_when_the_array_doubles():
    """~102/N degrees for a half-wavelength ULA: 12.75 deg at 8 elements,
    0.80 deg at 128. This is why an Nt=8 study cannot see prediction error."""
    assert np.isclose(half_power_beamwidth_deg(8), 102 / 8)
    assert np.isclose(half_power_beamwidth_deg(128), 102 / 128)
    assert half_power_beamwidth_deg(64) == 2 * half_power_beamwidth_deg(128)


def test_misalignment_is_measured_from_the_base_station_not_the_origin():
    """3_beamforming.ipynb used arctan2(x, y), which assumes the BS sits at
    (0,0). Users span x in [242,278], y in [297,847], so that angle was wrong."""
    bs = np.array([250.0, 400.0])
    truth = bs + np.array([0.0, 100.0])        # due +y from the BS
    predicted = bs + np.array([100.0, 100.0])  # 45 degrees off

    assert np.isclose(misalignment_deg(predicted, truth, bs), 45.0)
    assert np.isclose(misalignment_deg(truth, truth, bs), 0.0)


def test_misalignment_in_beamwidths_scales_with_array_size():
    """The same metres of error costs 16x more beamwidths at Nt=128 than Nt=8."""
    bs = np.array([250.0, 400.0])
    truth = bs + np.array([0.0, 200.0])
    predicted = truth + np.array([1.6, 0.0])

    small = misalignment_deg(predicted, truth, bs) / half_power_beamwidth_deg(8)
    large = misalignment_deg(predicted, truth, bs) / half_power_beamwidth_deg(128)

    assert np.isclose(large / small, 16.0)
    assert small < 0.1 and large > 0.5
