import numpy as np

from experiments.snr_eval import beam_snr, evaluate
from src.beamforming import mrt_weights, snr

BS = np.array([287.5, 651.5])
GRID = np.stack(np.meshgrid(np.arange(250, 270, 0.2), np.arange(380, 440, 0.2)), -1).reshape(-1, 2)


def _nearest(positions):
    return np.argmin(np.linalg.norm(GRID[None] - positions[:, None], axis=2), axis=1)


def _los_channels(n_tx):
    """Line-of-sight ULA response from the BS: phase set by the bearing, power by range."""
    def fn(indices):
        delta = GRID[indices] - BS
        theta = np.arctan2(delta[:, 1], delta[:, 0])
        amp = 1.0 / np.linalg.norm(delta, axis=1)
        return amp[:, None] * np.exp(1j * np.pi * np.arange(n_tx) * np.sin(theta)[:, None])
    return fn


def _corpus():
    """Two users walking down y at 1.5 m per slot."""
    y = np.linspace(390, 430, 30)
    truth = np.stack([np.stack([np.full(30, x), y], 1) for x in (255.0, 262.0)])
    return truth, np.array(["cyclist", "vehicular"])


def test_beam_snr_matches_the_reference_mrt_implementation():
    rng = np.random.default_rng(0)
    h_beam = rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))
    h_true = rng.normal(size=(5, 16)) + 1j * rng.normal(size=(5, 16))
    expected = [snr(t[None, :, None], mrt_weights(b[None, :, None]), 0.3)
                for b, t in zip(h_beam, h_true)]
    assert np.allclose(beam_snr(h_beam, h_true, 0.3), expected)


def test_beam_from_an_empty_channel_scores_zero():
    assert beam_snr(np.zeros((1, 8)), np.ones((1, 8)), 1.0)[0] == 0.0


def test_perfect_predictor_equals_oracle_and_beats_stale_reactive_beam():
    truth, regimes = _corpus()
    preds = {"perfect": truth, "noisy": truth + [0.0, 0.6]}
    out, info = evaluate(preds, truth, 10, regimes, _nearest, _los_channels(64), 1e-6)

    assert np.isclose(out["perfect"]["snr_db"], out["oracle"]["snr_db"])
    assert np.isclose(out["perfect"]["oracle_gap_recovered"], 1.0)
    assert out["reactive"]["gain_db"] == 0.0
    assert out["perfect"]["gain_db"] > out["noisy"]["gain_db"]
    assert out["perfect"]["se_gain"] > 0
    assert info["n_dropped"] == 0 and info["n_slots"] == 2 * 20


def test_reactive_beam_is_built_from_the_previous_slot():
    """A predictor that outputs slot t's true position is exactly the reactive beam."""
    truth, regimes = _corpus()
    stale = np.concatenate([truth[:, :1], truth[:, :-1]], axis=1)
    out, _ = evaluate({"stale": stale}, truth, 10, regimes, _nearest, _los_channels(64), 1e-6)
    assert np.isclose(out["stale"]["snr_db"], out["reactive"]["snr_db"])


def test_slots_with_no_channel_at_the_true_position_are_dropped_for_everyone():
    truth, regimes = _corpus()
    dead = set(_nearest(truth[0, 15:16]).tolist())
    los = _los_channels(8)

    def with_hole(indices):
        H = los(indices)
        H[np.isin(indices, list(dead))] = 0
        return H

    out, info = evaluate({"perfect": truth}, truth, 10, regimes, _nearest, with_hole, 1e-6)
    assert info["n_dropped"] == 1
    assert np.isfinite(out["perfect"]["snr_db"])


def test_gain_vs_imm_reported_for_other_methods_only():
    truth, regimes = _corpus()
    preds = {"imm": truth + [0.0, 0.3], "lstm": truth}
    out, _ = evaluate(preds, truth, 10, regimes, _nearest, _los_channels(64), 1e-6)
    assert out["lstm"]["gain_vs_imm_db"] > 0
    assert "gain_vs_imm_db" not in out["imm"] and "gain_vs_imm_db" not in out["oracle"]
    assert set(out["lstm"]["gain_db_by_regime"]) == {"cyclist", "vehicular"}
