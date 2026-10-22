import numpy as np

from experiments.codebook_eval import (dft_codebook, evaluate_codebook, hybrid_precoder,
                                       precoder_snr, refine_top_k)
from experiments.snr_eval import lookup_channels
from tests.test_snr_eval import _corpus, _los_channels, _nearest


def _random_channels(m, n_tx, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(m, n_tx)) + 1j * rng.normal(size=(m, n_tx))


def test_dft_codebook_is_unit_norm_constant_modulus_and_orthogonal_at_k_equals_n():
    W = dft_codebook(16)
    assert W.shape == (16, 16)
    assert np.allclose(np.abs(W), 1 / 4)                     # phase shifters only
    assert np.allclose(W.conj().T @ W, np.eye(16), atol=1e-12)
    assert dft_codebook(16, oversample=2).shape == (16, 32)


def test_single_rf_chain_is_the_best_single_codebook_beam():
    h = _random_channels(20, 16)
    W = dft_codebook(16, 2)
    f = hybrid_precoder(h, W, n_rf=1)
    best = np.max(np.abs(h @ W) ** 2, axis=1)
    assert np.allclose(precoder_snr(f, h, 1.0), best)
    assert np.allclose(np.linalg.norm(f, axis=1), 1.0)


def test_more_rf_chains_never_hurt_and_full_rank_hybrid_equals_mrt():
    """With orthonormal F_RF, adding beams can only add gain; with all N of them
    F_RF is unitary and the hybrid precoder is exactly fully-digital MRT."""
    h = _random_channels(50, 16, seed=1)
    W = dft_codebook(16)
    gains = [precoder_snr(hybrid_precoder(h, W, r), h, 1.0) for r in (1, 2, 4, 16)]
    for lo, hi in zip(gains, gains[1:]):
        assert np.all(hi >= lo - 1e-9)
    assert np.allclose(gains[-1], np.sum(np.abs(h) ** 2, axis=1))


def test_top_k_refinement_recovers_the_true_best_beam_as_k_grows():
    h_true = _random_channels(200, 16, seed=2)
    h_src = h_true + 0.8 * _random_channels(200, 16, seed=3)     # imperfect prediction
    W = dft_codebook(16)
    best = np.argmax(np.abs(h_true @ W), axis=1)
    match = [np.mean(refine_top_k(h_src, h_true, W, k) == best) for k in (1, 3, 16)]
    assert match[0] < match[1] < match[2] == 1.0


def test_evaluate_codebook_end_to_end_on_los_channels():
    truth, regimes = _corpus()
    preds = {"perfect": truth, "noisy": truth + [0.0, 0.6]}
    chans, valid, _ = lookup_channels(preds, truth, 10, _nearest, _los_channels(64))
    W = dft_codebook(64)
    res = evaluate_codebook(chans, valid, regimes, 1e-6, W, [1, 2], [1, 3], [0.0, 0.01])

    analog = res["hybrid"]["1"]
    assert analog["perfect"]["beam_match"] == 1.0
    assert np.isclose(analog["perfect"]["snr_db"], analog["oracle"]["snr_db"])
    # A 0.6 m error rarely changes a 64-beam codebook choice on a LoS channel -- the
    # robustness the codebook buys -- so the two can tie up to rounding.
    assert analog["perfect"]["gain_db"] >= analog["noisy"]["gain_db"] - 1e-9

    ex = res["training"]["exhaustive"]
    assert ex["beams_measured"] == 64 and ex["beam_match"] == 1.0
    assert np.isclose(ex["se_eff"]["0.01"], (1 - 0.64) * ex["se"])
    top1 = res["training"]["perfect"]["1"]
    assert np.isclose(top1["se"], ex["se"])                # same beam, ...
    assert top1["se_eff"]["0.01"] > ex["se_eff"]["0.01"]  # ... a fraction of the training
