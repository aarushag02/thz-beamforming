import numpy as np
import pytest

from src.data import (WARMUP, build_sequences, masked_mae_abs, masked_mae_metres,
                      sequence_truth, stream_to_sequence)
from src.trajectories import generate_corpus

UNIT = {"x": np.ones(2), "y": np.ones(2)}


def _line(T=6):
    t = np.arange(T, dtype=float)
    return {"observed": np.stack([t, np.zeros(T)], 1),
            "true": np.stack([t, np.full(T, 10.0)], 1), "regime": "pedestrian"}


def test_direct_alignment_hand_computed():
    d = build_sequences([_line(6)], mode="direct", scales=UNIT, warmup=0)
    L = 4
    assert d["X"].shape == (1, L, 2) and d["Y"].shape == (1, L, 2)
    assert np.allclose(d["X"][0], [[1, 0]] * L)                    # obs[t]-obs[t-1]
    assert np.allclose(d["anchors"][0], [[k + 1, 0] for k in range(L)])  # obs[t]
    assert np.allclose(d["truth"][0], [[k + 2, 10] for k in range(L)])   # true[t+1]
    assert np.allclose(d["Y"][0], [[1, 10]] * L)


def test_anchor_plus_scaled_target_round_trips_to_truth():
    corpus = generate_corpus(n=4, steps=60, rng=np.random.default_rng(0))
    for mode, kw in (("direct", {}), ("hybrid", {"imm_params": {"dt": 0.1, "sigma": 0.5}})):
        d = build_sequences(corpus, mode=mode, **kw)
        back = d["anchors"] + d["Y"] * d["scales"]["y"]
        assert np.allclose(back, d["truth"])


def test_hybrid_has_four_features_and_is_causal():
    corpus = generate_corpus(n=1, steps=60, rng=np.random.default_rng(1))
    imm = {"dt": 0.1, "sigma": 0.5}
    a = build_sequences(corpus, mode="hybrid", imm_params=imm, scales={"x": np.ones(4), "y": np.ones(2)})
    assert a["X"].shape[-1] == 4
    changed = [dict(corpus[0], observed=corpus[0]["observed"].copy())]
    changed[0]["observed"][31:] += 20.0      # perturb obs from slot t=31 (k=30) on
    b = build_sequences(changed, mode="hybrid", imm_params=imm, scales={"x": np.ones(4), "y": np.ones(2)})
    assert np.allclose(a["X"][0, :30], b["X"][0, :30])


def test_hybrid_requires_imm_params():
    with pytest.raises(ValueError):
        build_sequences([_line()], mode="hybrid")


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        build_sequences([_line()], mode="nope")


def test_mixed_lengths_rejected():
    """Review focus 2."""
    with pytest.raises(ValueError, match="same length"):
        build_sequences([_line(6), _line(8)], scales=UNIT)


def test_scales_fitted_on_given_data_and_reused():
    corpus = generate_corpus(n=4, steps=60, rng=np.random.default_rng(2))
    tr = build_sequences(corpus[:3])
    te = build_sequences(corpus[3:], scales=tr["scales"])
    assert te["scales"] is tr["scales"]
    assert tr["scales"]["x"].shape == (2,) and tr["scales"]["y"].shape == (2,)


def test_masked_metrics_ignore_warmup_steps():
    d = build_sequences([_line(20)], scales=UNIT, warmup=5)
    pred = d["Y"].copy()
    pred[:, :5] += 100.0                      # garbage inside warm-up only
    assert masked_mae_metres(pred, d) == pytest.approx(0.0)
    truth = d["truth"]
    wrong = truth + np.array([3.0, 4.0])
    assert masked_mae_abs(wrong, truth, warmup=5) == pytest.approx(5.0)


def test_stream_helpers_align_with_sequences():
    traj = _line(7)
    stream = np.arange(14, dtype=float).reshape(7, 2)
    assert np.allclose(stream_to_sequence(stream), stream[1:6])
    assert np.allclose(sequence_truth(traj), traj["true"][2:])
    assert WARMUP == 10
