import numpy as np

from experiments.codebook_eval import dft_codebook, hybrid_precoder, precoder_snr
from experiments.phase_noise_eval import (apply_phase_error, evaluate_phase_noise,
                                          expected_loss_db, mrt_precoder)
from experiments.snr_eval import evaluate, lookup_channels
from tests.test_snr_eval import _corpus, _los_channels, _nearest


def _random_channels(m, n_tx, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, n_tx)) + 1j * rng.normal(size=(m, n_tx))


def test_phase_error_keeps_the_beam_unit_norm():
    f = mrt_precoder(_random_channels(10, 64))
    phi = np.random.default_rng(1).normal(0, 0.5, f.shape)
    assert np.allclose(np.linalg.norm(apply_phase_error(f, phi), axis=1), 1.0)


def test_mean_array_gain_loss_matches_theory():
    """With a_k = |h_k|^2 / ||h||^2, E|sum_k a_k e^{j phi_k}|^2 is exactly
    e^{-s^2} + (1 - e^{-s^2}) sum_k a_k^2; the second term is O(1/N), so the loss
    tends to expected_loss_db(sigma) as the array grows."""
    for n_tx in (64, 1024):
        h = _random_channels(2000, n_tx, seed=2)
        f = mrt_precoder(h)
        a = np.abs(h) ** 2 / np.sum(np.abs(h) ** 2, axis=1, keepdims=True)
        for sigma in (0.1, 0.3, 0.5):
            phi = np.random.default_rng(3).normal(0, sigma, f.shape)
            ratio = precoder_snr(apply_phase_error(f, phi), h, 1.0) / precoder_snr(f, h, 1.0)
            c = np.exp(-sigma ** 2)
            assert np.isclose(ratio.mean(), np.mean(c + (1 - c) * np.sum(a ** 2, axis=1)), rtol=5e-3)
    assert np.isclose(-10 * np.log10(ratio.mean()), expected_loss_db(0.5), atol=0.02)  # N = 1024


def test_common_phase_rotation_does_not_change_snr():
    """Why per-element error, not oscillator phase noise: a shared LO rotation is harmless."""
    h = _random_channels(5, 32, seed=4)
    f = mrt_precoder(h)
    assert np.allclose(precoder_snr(apply_phase_error(f, np.full(f.shape, 1.3)), h, 1.0),
                       precoder_snr(f, h, 1.0))


def test_zero_sigma_reproduces_the_mrt_evaluation():
    truth, regimes = _corpus()
    preds = {"noisy": truth + [0.0, 0.6]}
    chans, valid, _ = lookup_channels(preds, truth, 10, _nearest, _los_channels(64))
    res = evaluate_phase_noise(chans, valid, regimes, 1e-6, mrt_precoder, [0.0], draws=5)
    ref, _ = evaluate(preds, truth, 10, regimes, _nearest, _los_channels(64), 1e-6)
    for name in ref:
        assert np.isclose(res["0"][name]["snr_db"], ref[name]["snr_db"])


def test_same_draw_for_every_method_and_snr_falls_with_sigma():
    truth, regimes = _corpus()
    preds = {"perfect": truth, "noisy": truth + [0.0, 0.6]}
    chans, valid, _ = lookup_channels(preds, truth, 10, _nearest, _los_channels(64))
    W = dft_codebook(64)
    res = evaluate_phase_noise(chans, valid, regimes, 1e-6, lambda h: hybrid_precoder(h, W, 1),
                               [0.0, 0.3, 0.5], draws=4)
    for sigma in ("0", "0.3", "0.5"):
        # identical beam + identical phase draw -> identical SNR
        assert np.isclose(res[sigma]["perfect"]["snr_db"], res[sigma]["oracle"]["snr_db"])
    assert res["0"]["oracle"]["snr_db"] > res["0.3"]["oracle"]["snr_db"] > res["0.5"]["oracle"]["snr_db"]
    assert np.isclose(res["0.5"]["expected_loss_db"], expected_loss_db(0.5))
