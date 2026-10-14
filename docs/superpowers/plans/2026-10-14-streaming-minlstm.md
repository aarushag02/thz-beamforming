# Streaming minLSTM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a streaming (whole-trajectory, O(1)-per-slot) formulation of the LSTM/minLSTM next-position predictors, streaming KF/IMM baselines, compute benchmarks and an end-to-end study script that runs on a GCP GPU.

**Architecture:** Library additions in `src/` (baselines, data, models, train, bench), each covered by fast CPU tests; one orchestration script `experiments/run_study.py` that writes `results/streaming_study.json` + `results/streaming_predictions.npz`; small post-processing scripts for beam misalignment and LaTeX tables.

**Tech Stack:** Python 3.10+, PyTorch ≥ 2.3, NumPy, SciPy, Optuna ≥ 3.6, Matplotlib, pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-streaming-minlstm-design.md`

## Global Constraints

- Run tests with `python -m pytest -q` from the repo root (`conftest.py` puts the root on `sys.path`; import as `from src.x import y`).
- All tests CPU-only and fast (whole suite < 3 min including the smoke run).
- Do not change behaviour of existing functions/classes; existing 52 tests must keep passing.
- Streaming index convention: sequence index `k = 0..L-1`, `L = T-2`, slot `t = k+1`; predicts `true[t+1]`.
- `WARMUP = 10`: loss and every metric exclude `k < 10`, for every method.
- Corpus defaults: `steps=200`, `sigma=0.5`, `dt=0.1`.
- Code style: match `src/` — module docstring explaining *why*, short function docstrings, no type-annotation noise, numpy/torch idioms already used.
- Every commit message ends with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. A huge single-step jump in observations (outlier) must not make IMM model probabilities NaN — outputs stay finite. (Task 1)
2. A corpus whose trajectories have different lengths must fail loudly in `build_sequences`, not silently mis-stack. (Task 2)
3. A NaN validation loss (diverging trial) must not crash training or get saved as "best"; it counts as non-improvement and training stops early. (Task 4)
4. `torch.compile` failing (Windows, missing compiler) must be recorded in the bench result, not crash the study. (Task 5)
5. `beam_eval.py` without DeepMIMO and without `--bs-pos` must exit with a clear message, not a traceback. (Task 7)

---

### Task 1: Streaming classical trackers

**Files:**
- Modify: `src/baselines.py` (append; keep everything existing)
- Test: `tests/test_baselines_stream.py`

**Interfaces:**
- Consumes: existing `hold`, `constant_velocity`, `ls_linear` (window `(W,2)` → `(2,)`).
- Produces:
  - `kalman_cv_stream(obs, dt=0.1, sigma=0.5, q=1.0) -> np.ndarray (T, 2)`; row `t` = predicted position at `t+1` from `obs[0..t]`.
  - `imm_stream(obs, dt=0.1, sigma=0.5, q=1.0, omega=0.25, p_stay=0.95) -> np.ndarray (T, 2)`; same row convention.
  - `stream_window_predictor(obs, fn, window) -> np.ndarray (T, 2)`; same row convention.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_baselines_stream.py
import numpy as np

from src.baselines import (constant_velocity, imm_stream, kalman_cv_stream, ls_linear,
                           stream_window_predictor)
from src.trajectories import coordinated_turn


def _noisy(path, sigma=0.5, seed=0):
    return path + np.random.default_rng(seed).normal(0.0, sigma, size=path.shape)


def _one_step_mae(pred, truth, skip=20):
    # pred[t] predicts truth[t+1]
    return float(np.mean(np.linalg.norm(pred[skip:-1] - truth[skip + 1:], axis=1)))


def test_stream_outputs_have_trajectory_shape():
    obs = _noisy(coordinated_turn(50, 10.0, np.zeros(50), 0.1, (0.0, 0.0), 0.0))
    for fn in (lambda o: kalman_cv_stream(o), lambda o: imm_stream(o)):
        out = fn(obs)
        assert out.shape == (50, 2)
        assert np.all(np.isfinite(out))


def test_stream_trackers_are_causal():
    """Row t may depend on obs[0..t] only."""
    obs = _noisy(coordinated_turn(60, 10.0, np.full(60, 0.2), 0.1, (0.0, 0.0), 0.0))
    perturbed = obs.copy()
    perturbed[30:] += 25.0
    for fn in (kalman_cv_stream, imm_stream):
        a, b = fn(obs), fn(perturbed)
        assert np.allclose(a[:30], b[:30])
        assert not np.allclose(a[30:], b[30:])


def test_kalman_stream_tracks_a_straight_line():
    truth = coordinated_turn(200, 10.0, np.zeros(200), 0.1, (0.0, 0.0), 0.3)
    pred = kalman_cv_stream(_noisy(truth), dt=0.1, sigma=0.5, q=1.0)
    # Far better than the 0.5 m raw observation noise once converged.
    assert _one_step_mae(pred, truth) < 0.35


def test_imm_close_to_kalman_on_straight_motion():
    truth = coordinated_turn(200, 10.0, np.zeros(200), 0.1, (0.0, 0.0), 0.3)
    obs = _noisy(truth)
    kf = _one_step_mae(kalman_cv_stream(obs, q=1.0), truth)
    imm = _one_step_mae(imm_stream(obs, q=1.0, omega=0.3, p_stay=0.97), truth)
    # IMM pays a model-switching penalty on straight lines (measured 1.14-1.17x).
    assert imm < 1.3 * kf


def test_imm_beats_kalman_on_a_sustained_turn():
    truth = coordinated_turn(300, 10.0, np.full(300, 0.3), 0.1, (0.0, 0.0), 0.0)
    obs = _noisy(truth)
    kf = _one_step_mae(kalman_cv_stream(obs, q=1.0), truth)
    imm = _one_step_mae(imm_stream(obs, q=1.0, omega=0.3, p_stay=0.97), truth)
    assert imm < kf


def test_imm_survives_an_outlier_jump():
    """Review focus 1: all model likelihoods underflow on a 500 m jump."""
    obs = _noisy(coordinated_turn(80, 10.0, np.zeros(80), 0.1, (0.0, 0.0), 0.0))
    obs[40] += 500.0
    out = imm_stream(obs)
    assert np.all(np.isfinite(out))


def test_stream_window_predictor_matches_windowed_fn():
    obs = _noisy(coordinated_turn(40, 5.0, np.full(40, 0.1), 0.1, (0.0, 0.0), 0.0))
    W = 5
    for fn in (constant_velocity, ls_linear):
        out = stream_window_predictor(obs, fn, W)
        for t in range(W - 1, 40):
            assert np.allclose(out[t], fn(obs[t - W + 1:t + 1]))
        assert np.allclose(out[0], obs[0])   # one point: falls back to hold
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_baselines_stream.py -q`
Expected: FAIL with `ImportError: cannot import name 'imm_stream'`

- [ ] **Step 3: Implement (append to `src/baselines.py`)**

```python
# --- Streaming trackers -----------------------------------------------------
#
# The windowed kalman_cv above restarts from zero velocity in every window,
# which is not how a base station would run a tracker and hands the learned
# models an easy win. These run once along the whole trajectory. Row t of every
# output is the prediction of position t+1 from obs[0..t].

_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])  # state [x, vx, y, vy]


def _cv_transition(dt):
    F = np.eye(4)
    F[0, 1] = F[2, 3] = dt
    return F


def _ct_transition(dt, omega):
    """Coordinated turn at a known rate omega (rad/s), counter-clockwise for omega > 0."""
    s, c = np.sin(omega * dt), np.cos(omega * dt)
    return np.array([
        [1.0, s / omega, 0.0, -(1.0 - c) / omega],
        [0.0, c, 0.0, -s],
        [0.0, (1.0 - c) / omega, 1.0, s / omega],
        [0.0, s, 0.0, c],
    ])


def _process_noise(dt, q):
    block = np.array([[dt**4 / 4, dt**3 / 2], [dt**3 / 2, dt**2]])
    zero = np.zeros((2, 2))
    return q * np.block([[block, zero], [zero, block]])


def _initial_state(first, sigma):
    return (np.array([first[0], 0.0, first[1], 0.0]),
            np.diag([sigma**2, 100.0, sigma**2, 100.0]))


def _kf_update(x, P, z, R):
    """Measurement update; also returns the innovation log-likelihood."""
    residual = z - _H @ x
    S = _H @ P @ _H.T + R
    S_inv = np.linalg.inv(S)
    K = P @ _H.T @ S_inv
    x = x + K @ residual
    P = (np.eye(4) - K @ _H) @ P
    _, logdet = np.linalg.slogdet(2.0 * np.pi * S)
    return x, P, -0.5 * (residual @ S_inv @ residual + logdet)


def kalman_cv_stream(obs, dt=0.1, sigma=0.5, q=1.0):
    """Constant-velocity Kalman filter run once along the trajectory."""
    obs = np.asarray(obs, dtype=float)[:, :2]
    F, Q, R = _cv_transition(dt), _process_noise(dt, q), sigma**2 * np.eye(2)
    x, P = _initial_state(obs[0], sigma)
    out = np.empty((len(obs), 2))
    for t, z in enumerate(obs):
        if t > 0:
            x, P = F @ x, F @ P @ F.T + Q
        x, P, _ = _kf_update(x, P, z, R)
        out[t] = (F @ x)[[0, 2]]
    return out


def _imm_mix(xs, Ps, mu, Pi):
    """IMM interaction step: predicted model probabilities and mixed priors."""
    c = Pi.T @ mu                                   # c[j] = P(model j next)
    weights = Pi * mu[:, None] / np.maximum(c[None, :], 1e-300)   # [i, j] = P(i | j)
    x_mix = [sum(weights[i, j] * xs[i] for i in range(len(xs))) for j in range(len(xs))]
    P_mix = []
    for j in range(len(xs)):
        P_j = np.zeros((4, 4))
        for i in range(len(xs)):
            d = xs[i] - x_mix[j]
            P_j += weights[i, j] * (Ps[i] + np.outer(d, d))
        P_mix.append(P_j)
    return c, x_mix, P_mix


def imm_stream(obs, dt=0.1, sigma=0.5, q=1.0, omega=0.25, p_stay=0.95):
    """Interacting-multiple-model tracker: CV, CT(+omega), CT(-omega).

    The standard maneuvering-target tracker and the comparator a tracking
    reviewer will ask for. All three models share the [x, vx, y, vy] state, so
    mixing is exact. Model likelihoods are combined in log space so a wild
    measurement cannot underflow every probability to zero.
    """
    obs = np.asarray(obs, dtype=float)[:, :2]
    Fs = [_cv_transition(dt), _ct_transition(dt, omega), _ct_transition(dt, -omega)]
    M = len(Fs)
    Q, R = _process_noise(dt, q), sigma**2 * np.eye(2)
    Pi = np.full((M, M), (1.0 - p_stay) / (M - 1))
    np.fill_diagonal(Pi, p_stay)

    x0, P0 = _initial_state(obs[0], sigma)
    xs, Ps = [x0.copy() for _ in Fs], [P0.copy() for _ in Fs]
    mu = np.full(M, 1.0 / M)
    out = np.empty((len(obs), 2))
    x_mix, P_mix = xs, Ps

    for t, z in enumerate(obs):
        if t > 0:
            xs = [F @ x for F, x in zip(Fs, x_mix)]
            Ps = [F @ P @ F.T + Q for F, P in zip(Fs, P_mix)]
            mu = c
        log_lik = np.empty(M)
        for j in range(M):
            xs[j], Ps[j], log_lik[j] = _kf_update(xs[j], Ps[j], z, R)
        log_post = np.log(np.maximum(mu, 1e-300)) + log_lik
        log_post -= log_post.max()
        mu = np.exp(log_post)
        mu /= mu.sum()

        c, x_mix, P_mix = _imm_mix(xs, Ps, mu, Pi)
        out[t] = sum(c[j] * (Fs[j] @ x_mix[j])[[0, 2]] for j in range(M))
    return out


def stream_window_predictor(obs, fn, window):
    """Run a (W, 2) -> (2,) window predictor causally along a trajectory.

    Early slots use whatever history exists; a single point falls back to hold.
    """
    obs = np.asarray(obs, dtype=float)[:, :2]
    out = np.empty((len(obs), 2))
    for t in range(len(obs)):
        recent = obs[max(0, t - window + 1):t + 1]
        out[t] = recent[-1] if len(recent) < 2 else fn(recent)
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_baselines_stream.py -q` then `python -m pytest -q`
Expected: all pass. If `test_imm_close_to_kalman_on_straight_motion` or `test_imm_beats_kalman_on_a_sustained_turn` fails, do NOT loosen thresholds — report the measured MAEs back.

- [ ] **Step 5: Commit**

```bash
git add src/baselines.py tests/test_baselines_stream.py
git commit -m "Add streaming Kalman and IMM trackers"
```

---

### Task 2: Whole-trajectory sequence dataset

**Files:**
- Modify: `src/data.py` (append)
- Test: `tests/test_data_seq.py`

**Interfaces:**
- Consumes: `imm_stream(obs, **imm_params)` from Task 1; existing `fit_scale`.
- Produces:
  - `WARMUP = 10`
  - `build_sequences(corpus, mode="direct", imm_params=None, scales=None, warmup=WARMUP) -> dict` with keys `X (N,L,F)` scaled, `Y (N,L,2)` scaled, `anchors (N,L,2)`, `truth (N,L,2)`, `scales {"x": (F,), "y": (2,)}`, `warmup`. `F = 2` direct, `4` hybrid.
  - `masked_mae_metres(pred_scaled, data) -> float`
  - `masked_mae_abs(pred_abs, truth, warmup=WARMUP) -> float`
  - `stream_to_sequence(stream_pred) -> (L, 2)` (= `stream_pred[1:-1]`)
  - `sequence_truth(traj) -> (L, 2)` (= `traj["true"][2:, :2]`)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_seq.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_data_seq.py -q`
Expected: FAIL with `ImportError: cannot import name 'WARMUP'`

- [ ] **Step 3: Implement (append to `src/data.py`; add `from src.baselines import imm_stream` to the imports)**

```python
# --- Whole-trajectory (streaming) sequences ---------------------------------
#
# The windowed dataset above re-processes W positions every slot and trains on
# the last step only, which throws away both of minLSTM's advantages. Here one
# sample is one whole trajectory with a target at every step. Sequence index k
# is slot t = k + 1: the model has seen obs[0..t] and predicts true[t+1].
# Inputs are per-step displacements, so nothing depends on where in the
# corridor the user is and no window anchor is needed.

WARMUP = 10  # steps excluded from loss and metrics, for every method alike


def _trajectory_arrays(traj, mode, imm_params):
    obs = np.asarray(traj["observed"], dtype=float)[:, :2]
    true = np.asarray(traj["true"], dtype=float)[:, :2]
    t = np.arange(1, len(obs) - 1)
    step = obs[t] - obs[t - 1]
    if mode == "direct":
        X, anchors = step, obs[t]
    elif mode == "hybrid":
        if imm_params is None:
            raise ValueError("mode='hybrid' needs imm_params")
        imm = imm_stream(obs, **imm_params)
        # The model sees the tracker's prediction and learns its correction.
        X, anchors = np.concatenate([step, imm[t] - obs[t]], axis=1), imm[t]
    else:
        raise ValueError(f"unknown mode {mode!r}; choose 'direct' or 'hybrid'")
    truth = true[t + 1]
    return X, truth - anchors, anchors, truth


def build_sequences(corpus, mode="direct", imm_params=None, scales=None, warmup=WARMUP):
    """One sample per trajectory, a target at every step.

    Scales are fitted per axis on this corpus when absent (pass the training
    split's scales for val/test). Returns a dict; see the module notes above.
    """
    if mode == "hybrid" and imm_params is None:
        raise ValueError("mode='hybrid' needs imm_params")
    lengths = {len(t["observed"]) for t in corpus}
    if len(lengths) != 1:
        raise ValueError(f"all trajectories must have the same length, got {sorted(lengths)}")
    parts = [_trajectory_arrays(t, mode, imm_params) for t in corpus]
    X, Y, anchors, truth = (np.stack([p[i] for p in parts]) for i in range(4))
    if scales is None:
        scales = {"x": fit_scale(X), "y": fit_scale(Y[:, warmup:])}
    return {"X": X / scales["x"], "Y": Y / scales["y"], "anchors": anchors,
            "truth": truth, "scales": scales, "warmup": warmup}


def masked_mae_abs(pred_abs, truth, warmup=WARMUP):
    """Mean Euclidean error in metres over steps k >= warmup."""
    error = np.asarray(pred_abs) - np.asarray(truth)
    return float(np.mean(np.linalg.norm(error, axis=-1)[..., warmup:]))


def masked_mae_metres(pred_scaled, data):
    """Model output (scaled, relative to anchor) -> metres -> masked MAE."""
    pred_abs = data["anchors"] + np.asarray(pred_scaled) * data["scales"]["y"]
    return masked_mae_abs(pred_abs, data["truth"], data["warmup"])


def stream_to_sequence(stream_pred):
    """Row t of a streaming tracker predicts t+1; keep slots t = 1..T-2."""
    return np.asarray(stream_pred)[1:-1]


def sequence_truth(traj):
    return np.asarray(traj["true"], dtype=float)[2:, :2]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_data_seq.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data_seq.py
git commit -m "Add whole-trajectory sequence dataset with hybrid IMM mode"
```

---

### Task 3: Sequence models with O(1) streaming step

**Files:**
- Modify: `src/minrnn.py` (add `MinLSTMLayer.step`, class `MinLSTMSeq`)
- Modify: `src/models.py` (add `PlainLSTMLayer.step`, `LSTMSeq`, `PlainLSTMSeq`, registry entries)
- Test: `tests/test_seq_models.py`

**Interfaces:**
- Produces (all seq models): `forward(x (B,L,F), mode=...) -> (B,L,out)`; `init_state(batch, device=None) -> state`; `step(x_t (B,F), state) -> (y_t (B,out), state)`.
- Registry names: `lstm_seq`, `plain_lstm_seq`, `minlstm_seq`, `minlstm_seq_res`. Constructor kwargs: `input_size, hidden_size, num_layers, output_size, dropout` (+ `log_space`, `residual_norm` for minLSTM).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seq_models.py
import pytest
import torch

from src.minrnn import MinLSTMLayer, MinLSTMSeq
from src.models import LSTMSeq, PlainLSTMSeq, build_model

SEQ = ["lstm_seq", "plain_lstm_seq", "minlstm_seq", "minlstm_seq_res"]


def _streamed(model, x):
    state = model.init_state(x.shape[0])
    outs = []
    for k in range(x.shape[1]):
        y, state = model.step(x[:, k], state)
        outs.append(y)
    return torch.stack(outs, dim=1)


def test_layer_step_matches_sequential():
    torch.manual_seed(0)
    for log_space in (False, True):
        layer = MinLSTMLayer(3, 8, log_space=log_space)
        x = torch.randn(2, 12, 3)
        h = torch.zeros(2, 8)
        steps = []
        for k in range(12):
            h = layer.step(x[:, k], h)
            steps.append(h)
        assert torch.allclose(torch.stack(steps, 1), layer.forward_sequential(x), atol=1e-6)


@pytest.mark.parametrize("name", SEQ)
def test_every_step_output_shape(name):
    model = build_model(name, input_size=4, hidden_size=16, num_layers=2, dropout=0.0)
    assert model(torch.randn(3, 25, 4)).shape == (3, 25, 2)


@pytest.mark.parametrize("name", SEQ)
def test_streaming_step_matches_full_forward(name):
    torch.manual_seed(1)
    model = build_model(name, input_size=2, hidden_size=16, num_layers=2, dropout=0.0).eval()
    x = torch.randn(3, 40, 2)
    with torch.no_grad():
        assert torch.allclose(_streamed(model, x), model(x), atol=1e-5)


def test_minlstm_seq_parallel_matches_sequential_both_variants():
    torch.manual_seed(2)
    for residual_norm in (False, True):
        model = MinLSTMSeq(input_size=2, hidden_size=16, num_layers=3, dropout=0.0,
                           residual_norm=residual_norm).eval()
        x = torch.randn(2, 64, 2)
        with torch.no_grad():
            assert torch.allclose(model(x, mode="parallel"), model(x, mode="sequential"), atol=1e-5)


@pytest.mark.parametrize("name", SEQ)
def test_outputs_are_causal(name):
    torch.manual_seed(3)
    model = build_model(name, input_size=2, hidden_size=16, num_layers=2, dropout=0.0).eval()
    x = torch.randn(2, 30, 2)
    y = x.clone()
    y[:, 15:] += 5.0
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[:, :15], b[:, :15], atol=1e-6)
    assert not torch.allclose(a[:, 15:], b[:, 15:])


def test_plain_lstm_seq_matches_cudnn_weights():
    torch.manual_seed(4)
    fused = LSTMSeq(input_size=2, hidden_size=16, num_layers=2, dropout=0.0).eval()
    plain = PlainLSTMSeq(input_size=2, hidden_size=16, num_layers=2, dropout=0.0).eval()
    plain.load_state_dict_from_cudnn(fused)
    x = torch.randn(3, 20, 2)
    with torch.no_grad():
        assert torch.allclose(plain(x), fused(x), atol=1e-5)


def test_minlstm_seq_rejects_unknown_mode():
    with pytest.raises(ValueError):
        MinLSTMSeq()(torch.randn(1, 4, 2), mode="bogus")


def test_existing_registry_entries_unchanged():
    for name in ("lstm", "plain_lstm", "minlstm", "minlstm_vanilla"):
        assert build_model(name)(torch.randn(2, 10, 2)).shape == (2, 2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_seq_models.py -q`
Expected: FAIL with `ImportError: cannot import name 'MinLSTMSeq'`

- [ ] **Step 3a: Implement in `src/minrnn.py`**

Add to `MinLSTMLayer` (after `forward_sequential`):

```python
    def step(self, x_t, h):
        """One recurrent update, x_t (batch, input) -> h (batch, hidden).

        This is the whole per-slot cost of streaming inference: three small
        linears on x_t and an elementwise blend, with no h-dependent matmul.
        """
        f_prime, i_prime = self.normalised_gates(x_t)
        return f_prime * h + i_prime * self.candidate(x_t)
```

Append class:

```python
class MinLSTMSeq(nn.Module):
    """Stacked minLSTM emitting a prediction at every step.

    Trained with the parallel scan over whole trajectories; deployed with
    step(), which carries one hidden vector per layer and costs O(1) per slot
    regardless of how long the user has been tracked.

    residual_norm=True wraps each layer as x + layer(LayerNorm(x)) after an
    input projection, the block structure the minLSTM paper uses for deeper
    stacks. Without it, depth > 2 stacks bare layers.
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1, log_space=True, residual_norm=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.residual_norm = residual_norm
        if residual_norm:
            self.proj = nn.Linear(input_size, hidden_size)
            self.norms = nn.ModuleList(nn.LayerNorm(hidden_size) for _ in range(num_layers))
            self.final_norm = nn.LayerNorm(hidden_size)
            sizes = [hidden_size] * (num_layers + 1)
        else:
            sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            MinLSTMLayer(sizes[i], sizes[i + 1], log_space=log_space)
            for i in range(num_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def _combine(self, n, x, h):
        """Feed layer n's output h forward, given that layer's input x."""
        if self.residual_norm:
            return x + self.drop(h)
        return self.drop(h) if n < self.num_layers - 1 else h

    def _layer_input(self, n, x):
        return self.norms[n](x) if self.residual_norm else x

    def _head(self, x):
        if self.residual_norm:
            x = self.final_norm(x)
        return self.fc(self.drop(x))

    def forward(self, x, mode="parallel"):
        if mode not in ("parallel", "sequential"):
            raise ValueError(f"unknown mode {mode!r}")
        if self.residual_norm:
            x = self.proj(x)
        for n, layer in enumerate(self.layers):
            run = layer.forward_parallel if mode == "parallel" else layer.forward_sequential
            x = self._combine(n, x, run(self._layer_input(n, x)))
        return self._head(x)

    def init_state(self, batch, device=None, dtype=torch.float32):
        return [torch.zeros(batch, self.hidden_size, device=device, dtype=dtype)
                for _ in self.layers]

    def step(self, x_t, state):
        if self.residual_norm:
            x_t = self.proj(x_t)
        new_state = []
        for n, layer in enumerate(self.layers):
            h = layer.step(self._layer_input(n, x_t), state[n])
            new_state.append(h)
            x_t = self._combine(n, x_t, h)
        return self._head(x_t), new_state
```

- [ ] **Step 3b: Implement in `src/models.py`**

Add to `PlainLSTMLayer`:

```python
    def step(self, x_t, state):
        """One LSTM cell update; state is (h, c). Needs h @ W_hh every slot."""
        h, c = state
        gates = x_t @ self.weight_ih.T + self.bias_ih + h @ self.weight_hh.T + self.bias_hh
        i, f, g, o = gates.chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, (h, c)
```

Append classes and registry entries (update the import to `from src.minrnn import MinLSTMPredictor, MinLSTMSeq`):

```python
class LSTMSeq(nn.Module):
    """nn.LSTM emitting a prediction at every step; step() carries (h, c)."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mode=None):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out))

    def init_state(self, batch, device=None, dtype=torch.float32):
        zeros = torch.zeros(self.num_layers, batch, self.hidden_size, device=device, dtype=dtype)
        return zeros, zeros.clone()

    def step(self, x_t, state):
        out, state = self.lstm(x_t.unsqueeze(1), state)
        return self.fc(self.drop(out[:, 0])), state


class PlainLSTMSeq(nn.Module):
    """Hand-written LSTM emitting every step -- the architecture-fair timing
    comparator for minLSTM_seq, as PlainLSTMPredictor is for the windowed one."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            PlainLSTMLayer(sizes[i], sizes[i + 1]) for i in range(num_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    load_state_dict_from_cudnn = PlainLSTMPredictor.load_state_dict_from_cudnn

    def forward(self, x, mode=None):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n < self.num_layers - 1:
                x = self.drop(x)
        return self.fc(self.drop(x))

    def init_state(self, batch, device=None, dtype=torch.float32):
        return [(torch.zeros(batch, self.hidden_size, device=device, dtype=dtype),
                 torch.zeros(batch, self.hidden_size, device=device, dtype=dtype))
                for _ in self.layers]

    def step(self, x_t, state):
        new_state = []
        for n, layer in enumerate(self.layers):
            x_t, s = layer.step(x_t, state[n])
            new_state.append(s)
            if n < self.num_layers - 1:
                x_t = self.drop(x_t)
        return self.fc(self.drop(x_t)), new_state
```

Add to the `MODELS` dict:

```python
    # Streaming variants: a prediction at every step, step() for O(1) inference.
    "lstm_seq": LSTMSeq,
    "plain_lstm_seq": PlainLSTMSeq,
    "minlstm_seq": MinLSTMSeq,
    "minlstm_seq_res": lambda **kw: MinLSTMSeq(residual_norm=True, **kw),
```

Note: `PlainLSTMSeq.load_state_dict_from_cudnn` reuses the windowed method, which reads `fused.lstm` / `fused.fc` and `self.layers` / `self.fc` — the same attribute names exist on `LSTMSeq`/`PlainLSTMSeq`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_seq_models.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/minrnn.py src/models.py tests/test_seq_models.py
git commit -m "Add streaming sequence models with O(1) step inference"
```

---

### Task 4: Sequence training loop

**Files:**
- Modify: `src/train.py` (append)
- Test: `tests/test_train_seq.py`

**Interfaces:**
- Consumes: seq models (Task 3), `build_sequences` dict (Task 2).
- Produces:
  - `train_seq_model(model, train, val, warmup=10, epochs=200, batch_size=32, lr=1e-3, weight_decay=0.0, device=None, patience=20, grad_clip=1.0, epoch_callback=None, mode="parallel") -> history` where `train`/`val` are `(X, Y)` arrays and `history = {"train_loss": [...], "val_loss": [...], "best_val_loss": float}`.
  - `predict_seq(model, X, device=None, mode="parallel", batch_size=256) -> np.ndarray (N, L, out)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_seq.py
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data import build_sequences
from src.models import build_model
from src.train import predict_seq, set_seed, train_seq_model
from src.trajectories import generate_corpus


def _split():
    corpus = generate_corpus(n=10, steps=60, rng=np.random.default_rng(0))
    tr = build_sequences(corpus[:8])
    va = build_sequences(corpus[8:], scales=tr["scales"])
    return (tr["X"], tr["Y"]), (va["X"], va["Y"])


def test_training_reduces_validation_loss():
    train, val = _split()
    set_seed(0)
    model = build_model("minlstm_seq", hidden_size=16, num_layers=2, dropout=0.0)
    hist = train_seq_model(model, train, val, warmup=10, epochs=15, batch_size=4,
                           lr=3e-3, device="cpu", patience=50)
    assert hist["val_loss"][-1] < hist["val_loss"][0]
    assert hist["best_val_loss"] == min(hist["val_loss"])


def test_predict_seq_shape():
    train, _ = _split()
    model = build_model("lstm_seq", hidden_size=8, num_layers=1, dropout=0.0)
    assert predict_seq(model, train[0], device="cpu").shape == train[1].shape


def test_warmup_steps_do_not_contribute_to_loss():
    """Targets inside warm-up are garbage; with warm-up masked, loss is unchanged."""
    train, val = _split()
    noisy_train = (train[0], train[1].copy())
    noisy_train[1][:, :10] = 1e6
    losses = []
    for t in (train, noisy_train):
        set_seed(1)
        model = build_model("minlstm_seq", hidden_size=8, num_layers=1, dropout=0.0)
        losses.append(train_seq_model(model, t, val, warmup=10, epochs=2, batch_size=4,
                                      device="cpu")["train_loss"])
    assert np.allclose(losses[0], losses[1])


def test_epoch_callback_called_and_can_abort():
    train, val = _split()
    seen = []

    def callback(epoch, val_loss):
        seen.append((epoch, val_loss))
        if epoch == 1:
            raise RuntimeError("pruned")

    model = build_model("minlstm_seq", hidden_size=8, num_layers=1, dropout=0.0)
    with pytest.raises(RuntimeError, match="pruned"):
        train_seq_model(model, train, val, epochs=5, batch_size=4, device="cpu",
                        epoch_callback=callback)
    assert [e for e, _ in seen] == [0, 1]


class _NaNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode=None):
        return x[..., :2] * self.w + float("nan")


def test_nan_validation_loss_stops_without_crashing():
    """Review focus 3."""
    train, val = _split()
    hist = train_seq_model(_NaNModel(), train, val, epochs=50, batch_size=4,
                           device="cpu", patience=3)
    assert hist["best_val_loss"] == float("inf")
    assert len(hist["val_loss"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_train_seq.py -q`
Expected: FAIL with `ImportError: cannot import name 'predict_seq'`

- [ ] **Step 3: Implement (append to `src/train.py`)**

```python
def _masked_mse(pred, target, warmup):
    return torch.mean((pred[:, warmup:] - target[:, warmup:]) ** 2)


def train_seq_model(model, train, val, warmup=10, epochs=200, batch_size=32, lr=1e-3,
                    weight_decay=0.0, device=None, patience=20, grad_clip=1.0,
                    epoch_callback=None, mode="parallel"):
    """Train a streaming model on whole trajectories with a loss at every step.

    Same optimiser, schedule and early-stopping rule as train_model, so the
    windowed and streaming studies differ only in formulation. Steps k < warmup
    are excluded from the loss exactly as they are from every metric.

    epoch_callback(epoch, val_loss) runs after each epoch; Optuna pruning raises
    from inside it. A non-finite validation loss counts as no improvement, so a
    diverging trial early-stops instead of being saved as best.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)
    train_loader = DataLoader(WindowDataset(*train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(*val), batch_size=batch_size)

    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(1, patience // 3))

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, stale = float("inf"), None, 0

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = _masked_mse(model(xb, mode=mode), yb, warmup)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            running += loss.item()
        history["train_loss"].append(running / len(train_loader))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                running += _masked_mse(model(xb.to(device), mode=mode),
                                       yb.to(device), warmup).item()
        val_loss = running / len(val_loader)
        history["val_loss"].append(val_loss)

        if np.isfinite(val_loss):
            scheduler.step(val_loss)
        if epoch_callback is not None:
            epoch_callback(epoch, val_loss)

        if np.isfinite(val_loss) and val_loss < best_loss - 1e-9:
            best_loss, stale = val_loss, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_loss"] = best_loss
    return history


@torch.no_grad()
def predict_seq(model, X, device=None, mode="parallel", batch_size=256):
    """Every-step predictions for whole trajectories, (N, L, out)."""
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    out = [model(X[i:i + batch_size].to(device), mode=mode).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)
```

Note: the NaN test expects exactly 3 epochs with `patience=3` (stale reaches 3 on epoch index 2).

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_train_seq.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_seq.py
git commit -m "Add whole-trajectory training loop with warm-up mask and pruning hook"
```

---

### Task 5: Compute benchmarks

**Files:**
- Create: `src/bench.py`
- Test: `tests/test_bench.py`

**Interfaces:**
- Consumes: `build_model` seq names (Task 3).
- Produces:
  - `time_train_step(name, seq_len, batch=64, hidden=64, layers=2, device="cpu", compile=False, warmup=5, iters=20, input_size=2) -> {"ms": float, "peak_mib": float, "error": str|None}`
  - `time_step_latency(name, device="cpu", hidden=64, layers=2, iters=500, input_size=2) -> float` (µs per slot)
  - `time_window_latency(name, window, device="cpu", hidden=64, layers=2, iters=500, input_size=2) -> float` (µs)
  - `macs_per_step(name, input_size=2, hidden=64, layers=2, output_size=2) -> int`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_bench.py
import math

import pytest
import torch

from src.bench import macs_per_step, time_step_latency, time_train_step, time_window_latency


def test_macs_hand_computed():
    # LSTM: 4*(2+64)*64 + 4*(64+64)*64 + 64*2
    assert macs_per_step("lstm_seq") == 16896 + 32768 + 128
    assert macs_per_step("plain_lstm_seq") == macs_per_step("lstm_seq")
    # minLSTM: 3*2*64 + 3*64*64 + 64*2
    assert macs_per_step("minlstm_seq") == 384 + 12288 + 128
    # residual: proj 2*64 + 2 * 3*64*64 + 64*2
    assert macs_per_step("minlstm_seq_res") == 128 + 24576 + 128
    with pytest.raises(ValueError):
        macs_per_step("transformer")


@pytest.mark.parametrize("name", ["lstm_seq", "plain_lstm_seq", "minlstm_seq"])
def test_bench_functions_run_on_cpu(name):
    r = time_train_step(name, seq_len=16, batch=4, hidden=8, layers=1, warmup=1, iters=2)
    assert r["ms"] > 0 and r["error"] is None and math.isnan(r["peak_mib"])
    assert time_step_latency(name, hidden=8, layers=1, iters=5) > 0
    assert time_window_latency(name, window=10, hidden=8, layers=1, iters=5) > 0


def test_compile_failure_is_recorded_not_raised(monkeypatch):
    """Review focus 4."""
    def broken(*a, **k):
        raise RuntimeError("no compiler")
    monkeypatch.setattr(torch, "compile", broken)
    r = time_train_step("minlstm_seq", seq_len=16, batch=2, hidden=8, layers=1,
                        compile=True, warmup=1, iters=1)
    assert math.isnan(r["ms"]) and "no compiler" in r["error"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_bench.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.bench'`

- [ ] **Step 3: Implement `src/bench.py`**

```python
"""Compute benchmarks for the streaming study.

Two different costs matter and they favour different designs:

  training   one forward+backward over a whole trajectory -- where minLSTM's
             parallel scan can beat a recurrence, but only past some length
  per slot   what a base station pays each time a new position arrives --
             one step() for a streaming model, a full window forward for the
             windowed design

CUDA calls are asynchronous, so every timed region is synchronised.
"""

import math
import time

import torch
import torch.nn.functional as F

from src.models import build_model


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def _model(name, input_size, hidden, layers, device):
    return build_model(name, input_size=input_size, hidden_size=hidden,
                       num_layers=layers, dropout=0.0).to(device)


def time_train_step(name, seq_len, batch=64, hidden=64, layers=2, device="cpu",
                    compile=False, warmup=5, iters=20, input_size=2):
    """Forward + every-step MSE + backward + Adam step, in milliseconds."""
    device = torch.device(device)
    model = _model(name, input_size, hidden, layers, device)
    try:
        forward = torch.compile(model) if compile else model
        opt = torch.optim.Adam(model.parameters())
        x = torch.randn(batch, seq_len, input_size, device=device)
        y = torch.randn(batch, seq_len, 2, device=device)

        def one_step():
            opt.zero_grad(set_to_none=True)
            F.mse_loss(forward(x), y).backward()
            opt.step()

        for _ in range(warmup):
            one_step()
        _sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(iters):
            one_step()
        _sync(device)
    except Exception as exc:  # compile/backends fail per platform; record, don't crash
        return {"ms": math.nan, "peak_mib": math.nan, "error": f"{type(exc).__name__}: {exc}"}
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else math.nan
    return {"ms": (time.perf_counter() - start) / iters * 1000.0, "peak_mib": peak, "error": None}


@torch.no_grad()
def time_step_latency(name, device="cpu", hidden=64, layers=2, iters=500, input_size=2):
    """Microseconds for one streaming step() at batch 1."""
    device = torch.device(device)
    model = _model(name, input_size, hidden, layers, device).eval()
    state = model.init_state(1, device=device)
    x = torch.randn(1, input_size, device=device)
    for _ in range(min(50, iters)):
        _, state = model.step(x, state)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        _, state = model.step(x, state)
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e6


@torch.no_grad()
def time_window_latency(name, window, device="cpu", hidden=64, layers=2, iters=500,
                        input_size=2):
    """Microseconds for one full forward over a (1, window) input -- the per-slot
    cost of the windowed design, which re-reads the window every slot."""
    device = torch.device(device)
    model = _model(name, input_size, hidden, layers, device).eval()
    x = torch.randn(1, window, input_size, device=device)
    for _ in range(min(50, iters)):
        model(x)
    _sync(device)
    start = time.perf_counter()
    for _ in range(iters):
        model(x)
    _sync(device)
    return (time.perf_counter() - start) / iters * 1e6


def macs_per_step(name, input_size=2, hidden=64, layers=2, output_size=2):
    """Multiply-accumulates for one streaming step (matmuls only).

    An LSTM layer multiplies both x_t and h_{t-1}: 4*(in+H)*H. A minLSTM layer
    only multiplies x_t: 3*in*H. The h-dependent matmul is also what forces the
    LSTM to be sequential.
    """
    head = hidden * output_size
    sizes = [input_size] + [hidden] * layers
    if name in ("lstm_seq", "plain_lstm_seq"):
        return sum(4 * (sizes[i] + hidden) * hidden for i in range(layers)) + head
    if name == "minlstm_seq":
        return sum(3 * sizes[i] * hidden for i in range(layers)) + head
    if name == "minlstm_seq_res":
        return input_size * hidden + layers * 3 * hidden * hidden + head
    raise ValueError(f"no MAC model for {name!r}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_bench.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/bench.py tests/test_bench.py
git commit -m "Add training-step, per-slot latency and MAC benchmarks"
```

---

### Task 6: End-to-end study script

**Files:**
- Create: `experiments/__init__.py` (empty), `experiments/run_study.py`
- Test: `tests/test_run_study.py`

**Interfaces:**
- Consumes: everything above; `generate_corpus`, `split_by_trajectory`, `split_by_regime`, `REGIMES`; `hold`, `constant_velocity`, `ls_linear`; `set_seed`, `resolve_device`.
- Produces: `main(argv=None)`; files `<out>/streaming_study.json`, `<out>/streaming_predictions.npz`, `<out>/streaming_*.png`, `<models>/<arch>_<mode>.pt`. JSON top-level keys: `meta`, `classical`, `learned`, `generalisation`, `bench`. NPZ keys: `truth`, `regimes`, `warmup`, and one `pred_<method>` per method (absolute metres, `(N_test, L, 2)`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_study.py
import json

import numpy as np

from experiments.run_study import main


def test_smoke_run_writes_complete_results(tmp_path):
    out, models = tmp_path / "results", tmp_path / "models"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(models)])

    study = json.loads((out / "streaming_study.json").read_text())
    assert set(study) >= {"meta", "classical", "learned", "generalisation", "bench"}
    assert {"hold", "constant_velocity", "ls_linear", "kalman", "imm"} <= set(study["classical"])
    for key in ("lstm_seq_direct", "minlstm_seq_direct", "minlstm_seq_hybrid"):
        entry = study["learned"][key]
        assert len(entry["test_mae"]) == 1 and np.isfinite(entry["test_mae"][0])
        assert set(entry["per_regime"]) <= {"pedestrian", "cyclist", "vehicular"}
    assert study["bench"]["train_step"]["minlstm_seq"]["seq_lens"] == [16, 64]
    assert study["bench"]["macs"]["minlstm_seq"] == 12800

    preds = np.load(out / "streaming_predictions.npz")
    assert preds["truth"].ndim == 3
    assert preds["pred_imm"].shape == preds["truth"].shape
    assert preds["pred_minlstm_seq_hybrid"].shape == preds["truth"].shape
    assert (models / "minlstm_seq_direct.pt").exists()
    assert list(out.glob("streaming_*.png"))


def test_skip_flags(tmp_path):
    out = tmp_path / "r"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(tmp_path / "m"),
          "--skip-bench", "--skip-generalisation"])
    study = json.loads((out / "streaming_study.json").read_text())
    assert study["bench"] == {} and study["generalisation"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_run_study.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'experiments'`

- [ ] **Step 3: Implement `experiments/run_study.py`**

```python
"""Streaming study: classical trackers vs LSTM vs minLSTM, plus compute.

    python experiments/run_study.py --smoke            # < 2 min on CPU, wiring check
    python experiments/run_study.py --quick            # ~15 min on a GPU, sanity run
    python experiments/run_study.py                    # full study

Results are written after every phase, so a crash keeps what finished.
"""

import argparse
import json
import pathlib
import platform
import sys
import time
from collections import defaultdict
from itertools import product

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

from src.baselines import (constant_velocity, hold, imm_stream, kalman_cv_stream,
                           ls_linear, stream_window_predictor)
from src.bench import macs_per_step, time_step_latency, time_train_step, time_window_latency
from src.data import (WARMUP, build_sequences, masked_mae_abs, masked_mae_metres,
                      sequence_truth, split_by_regime, split_by_trajectory, stream_to_sequence)
from src.models import build_model
from src.train import predict_seq, resolve_device, set_seed, train_seq_model
from src.trajectories import REGIMES, generate_corpus

SCALES = {
    "smoke": dict(n_traj=24, trials=2, seeds=1, epochs=3, patience=2,
                  seq_lens=[16, 64], bench_iters=2, latency_iters=20),
    "quick": dict(n_traj=90, trials=4, seeds=2, epochs=40, patience=8,
                  seq_lens=[16, 32, 64, 128, 256, 512], bench_iters=10, latency_iters=200),
    "full": dict(n_traj=600, trials=30, seeds=5, epochs=200, patience=20,
                 seq_lens=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                 bench_iters=20, latency_iters=1000),
}
CORPUS = dict(steps=200, sigma=0.5, dt=0.1)
ARCHS = ["lstm_seq", "minlstm_seq", "minlstm_seq_res"]
MODES = ["direct", "hybrid"]
KF_Q = [0.1, 1.0, 10.0, 100.0, 1000.0]
IMM_GRID = dict(q=[0.3, 3.0, 30.0], omega=[0.1, 0.25, 0.5], p_stay=[0.9, 0.97])
LS_WINDOWS = [3, 5, 10, 20]
BENCH_ARCHS = ["lstm_seq", "plain_lstm_seq", "minlstm_seq"]
COLOURS = {"lstm_seq": "#2a78d6", "plain_lstm_seq": "#eb6834", "minlstm_seq": "#1baf7a",
           "minlstm_seq_compiled": "#0f6e4c", "minlstm_seq_res": "#7a5cc4"}

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


# --- classical ---------------------------------------------------------------

def stream_preds(trajs, fn):
    return np.stack([stream_to_sequence(fn(t["observed"])) for t in trajs])


def truths(trajs):
    return np.stack([sequence_truth(t) for t in trajs])


def per_regime(pred_abs, trajs):
    groups = defaultdict(list)
    for p, t in zip(pred_abs, trajs):
        groups[t["regime"]].append(masked_mae_abs(p[None], sequence_truth(t)[None]))
    return {r: float(np.mean(v)) for r, v in groups.items()}


def classical_predictors(dt, sigma):
    """name -> (list of candidate (params, fn)) to tune on validation."""
    return {
        "hold": [({}, lambda o: stream_window_predictor(o, hold, 1))],
        "constant_velocity": [({}, lambda o: stream_window_predictor(o, constant_velocity, 2))],
        "ls_linear": [({"window": w}, lambda o, w=w: stream_window_predictor(o, ls_linear, w))
                      for w in LS_WINDOWS],
        "kalman": [({"q": q}, lambda o, q=q: kalman_cv_stream(o, dt=dt, sigma=sigma, q=q))
                   for q in KF_Q],
        "imm": [({"q": q, "omega": w, "p_stay": p},
                 lambda o, q=q, w=w, p=p: imm_stream(o, dt=dt, sigma=sigma, q=q, omega=w, p_stay=p))
                for q, w, p in product(IMM_GRID["q"], IMM_GRID["omega"], IMM_GRID["p_stay"])],
    }


def phase_classical(val_c, test_c):
    results, preds, fns = {}, {}, {}
    val_truth, test_truth = truths(val_c), truths(test_c)
    for name, candidates in classical_predictors(CORPUS["dt"], CORPUS["sigma"]).items():
        scored = [(masked_mae_abs(stream_preds(val_c, fn), val_truth), params, fn)
                  for params, fn in candidates]
        val_mae, params, fn = min(scored, key=lambda s: s[0])
        test_pred = stream_preds(test_c, fn)
        results[name] = dict(params=params, val_mae=val_mae,
                             test_mae=masked_mae_abs(test_pred, test_truth),
                             per_regime=per_regime(test_pred, test_c))
        preds[name], fns[name] = test_pred, fn
        log(f"classical {name:<18} val {val_mae:.3f} m  test {results[name]['test_mae']:.3f} m  {params}")
    return results, preds, fns


# --- learned -----------------------------------------------------------------

def suggest(trial):
    return dict(
        hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
        layers=trial.suggest_categorical("layers", [1, 2, 3]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.2),
        batch=trial.suggest_categorical("batch", [16, 32, 64]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
    )


def fit(arch, params, data, seed, cfg, device, callback=None):
    set_seed(seed)
    model = build_model(arch, input_size=data["train"]["X"].shape[-1],
                        hidden_size=params["hidden"], num_layers=params["layers"],
                        dropout=params["dropout"])
    train_seq_model(model, (data["train"]["X"], data["train"]["Y"]),
                    (data["val"]["X"], data["val"]["Y"]), warmup=WARMUP,
                    epochs=cfg["epochs"], batch_size=params["batch"], lr=params["lr"],
                    weight_decay=params["weight_decay"], device=device,
                    patience=cfg["patience"], epoch_callback=callback)
    return model


def evaluate(model, split, device):
    pred = predict_seq(model, split["X"], device=device)
    return masked_mae_metres(pred, split), split["anchors"] + pred * split["scales"]["y"]


def make_datasets(train_c, val_c, test_c, mode, imm_params):
    tr = build_sequences(train_c, mode, imm_params)
    return {"train": tr,
            "val": build_sequences(val_c, mode, imm_params, scales=tr["scales"]),
            "test": build_sequences(test_c, mode, imm_params, scales=tr["scales"])}


def search(arch, data, cfg, device, seed):
    def objective(trial):
        params = suggest(trial)

        def callback(epoch, val_loss):
            trial.report(val_loss if np.isfinite(val_loss) else 1e9, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = fit(arch, params, data, seed, cfg, device, callback)
        mae, _ = evaluate(model, data["val"], device)
        return mae if np.isfinite(mae) else float("inf")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective, n_trials=cfg["trials"], show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def phase_learned(datasets, test_c, cfg, device, seed, models_dir):
    results, preds = {}, {}
    for mode, arch in product(MODES, ARCHS):
        key = f"{arch}_{mode}"
        data = datasets[mode]
        params, best_val = search(arch, data, cfg, device, seed)
        val_maes, test_maes, first = [], [], None
        for s in range(cfg["seeds"]):
            model = fit(arch, params, data, seed + s, cfg, device)
            val_maes.append(evaluate(model, data["val"], device)[0])
            test_mae, test_pred = evaluate(model, data["test"], device)
            test_maes.append(test_mae)
            if first is None:
                first = (model, test_pred)
        model, test_pred = first
        torch.save(model.state_dict(), models_dir / f"{key}.pt")
        results[key] = dict(arch=arch, mode=mode, params=params, search_best_val=best_val,
                            val_mae=val_maes, test_mae=test_maes,
                            test_mean=float(np.mean(test_maes)), test_std=float(np.std(test_maes)),
                            per_regime=per_regime(test_pred, test_c),
                            n_params=sum(p.numel() for p in model.parameters()))
        preds[key] = test_pred
        log(f"learned {key:<24} test {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} m  {params}")
    return results, preds


def phase_generalisation(corpus, learned, classical_fns, imm_params, cfg, device, seed):
    rows = []
    for regime in REGIMES:
        seen, unseen = split_by_regime(corpus, held_out=regime)
        if len(unseen) < 2 or len(seen) < 10:   # val split needs >= 2 trajectories
            continue
        tr_c, va_c, _ = split_by_trajectory(seen, (0.8, 0.2, 0.0), rng=np.random.default_rng(seed))
        row = dict(regime=regime, n_unseen=len(unseen))
        for name in ("kalman", "imm"):
            row[name] = masked_mae_abs(stream_preds(unseen, classical_fns[name]), truths(unseen))
        for key, entry in learned.items():
            data = make_datasets(tr_c, va_c, unseen, entry["mode"], imm_params)
            model = fit(entry["arch"], entry["params"], data, seed, cfg, device)
            row[key] = evaluate(model, data["test"], device)[0]
        rows.append(row)
        log(f"held-out {regime}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()
                                               if isinstance(v, float)))
    return rows


# --- compute -----------------------------------------------------------------

def phase_bench(cfg, device):
    bench = {"device": str(device), "train_step": {}, "latency_us": {}, "macs": {}}
    variants = [(a, a, False) for a in BENCH_ARCHS]
    if device.type == "cuda":
        variants.append(("minlstm_seq_compiled", "minlstm_seq", True))
    for label, arch, compiled in variants:
        rows = [time_train_step(arch, T, device=device, compile=compiled,
                                iters=cfg["bench_iters"]) for T in cfg["seq_lens"]]
        bench["train_step"][label] = dict(seq_lens=cfg["seq_lens"],
                                          ms=[r["ms"] for r in rows],
                                          peak_mib=[r["peak_mib"] for r in rows],
                                          errors=[r["error"] for r in rows])
        log(f"bench {label:<22} " + " ".join(f"{m:.2f}" for m in bench["train_step"][label]["ms"]))
    for dev in {device, torch.device("cpu")}:
        for arch in BENCH_ARCHS + ["minlstm_seq_res"]:
            entry = {"step": time_step_latency(arch, device=dev, iters=cfg["latency_iters"])}
            for W in (10, 40):
                entry[f"window_{W}"] = time_window_latency(arch, W, device=dev,
                                                           iters=cfg["latency_iters"])
            bench["latency_us"][f"{arch}@{dev.type}"] = entry
    for arch in BENCH_ARCHS + ["minlstm_seq_res"]:
        bench["macs"][arch] = macs_per_step(arch)
    return bench


# --- figures -----------------------------------------------------------------

def plot_accuracy(study, path):
    rows = [(f"{k} (classical)", v["test_mae"], 0.0) for k, v in study["classical"].items()]
    rows += [(k, v["test_mean"], v["test_std"]) for k, v in study["learned"].items()]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.2))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows],
            color=["#8a8984" if "classical" in r[0] else "#1baf7a" if "minlstm" in r[0]
                   else "#2a78d6" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test MAE (m), lower is better")
    ax.set_title("Next-position prediction on held-out trajectories")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_step(bench, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, entry in bench["train_step"].items():
        ax.plot(entry["seq_lens"], entry["ms"], marker="o", label=label,
                color=COLOURS.get(label))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("training step (ms)")
    ax.set_title(f"Training step time ({bench['device']})")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_generalisation(rows, path):
    if not rows:
        return
    keys = [k for k in rows[0] if k not in ("regime", "n_unseen")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax.bar(np.arange(len(rows)) + i * width, [r[key] for r in rows], width,
               label=key, color=COLOURS.get(key.rsplit("_", 1)[0]))
    ax.set_xticks(np.arange(len(rows)) + 0.4 - width / 2)
    ax.set_xticklabels([f"held-out {r['regime']}" for r in rows])
    ax.set_ylabel("MAE (m)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- main --------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true")
    scale.add_argument("--quick", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-generalisation", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scale = "smoke" if args.smoke else "quick" if args.quick else "full"
    cfg = SCALES[scale]
    device = resolve_device(args.device)
    out, models_dir = pathlib.Path(args.out), pathlib.Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    json_path = out / "streaming_study.json"

    study = dict(meta=dict(scale=scale, cfg=cfg, corpus=CORPUS, seed=args.seed, warmup=WARMUP,
                           device=str(device), torch=torch.__version__,
                           python=platform.python_version(),
                           gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                           started=time.strftime("%Y-%m-%d %H:%M:%S")),
                 classical={}, learned={}, generalisation=[], bench={})
    log(f"scale={scale} device={device} gpu={study['meta']['gpu']}")

    rng = np.random.default_rng(args.seed)
    corpus = generate_corpus(n=cfg["n_traj"], steps=CORPUS["steps"], rng=rng,
                             sigma=CORPUS["sigma"], dt=CORPUS["dt"])
    train_c, val_c, test_c = split_by_trajectory(corpus, (0.7, 0.15, 0.15), rng=rng)
    log(f"corpus {len(train_c)}/{len(val_c)}/{len(test_c)} train/val/test trajectories")

    study["classical"], classical_preds, classical_fns = phase_classical(val_c, test_c)
    save_json(json_path, study)

    imm_params = dict(dt=CORPUS["dt"], sigma=CORPUS["sigma"], **study["classical"]["imm"]["params"])
    datasets = {mode: make_datasets(train_c, val_c, test_c, mode, imm_params) for mode in MODES}
    study["learned"], learned_preds = phase_learned(datasets, test_c, cfg, device, args.seed, models_dir)
    save_json(json_path, study)

    np.savez_compressed(out / "streaming_predictions.npz", truth=truths(test_c),
                        regimes=np.array([t["regime"] for t in test_c]), warmup=WARMUP,
                        **{f"pred_{k}": v for k, v in {**classical_preds, **learned_preds}.items()})
    plot_accuracy(study, out / "streaming_accuracy.png")

    if not args.skip_generalisation:
        study["generalisation"] = phase_generalisation(corpus, study["learned"], classical_fns,
                                                       imm_params, cfg, device, args.seed)
        save_json(json_path, study)
        plot_generalisation(study["generalisation"], out / "streaming_generalisation.png")

    if not args.skip_bench:
        study["bench"] = phase_bench(cfg, device)
        save_json(json_path, study)
        plot_train_step(study["bench"], out / "streaming_train_step.png")

    study["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(json_path, study)
    log(f"done -> {json_path}")
    return study


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_run_study.py -q` then `python -m pytest -q`
Expected: all pass; smoke run under ~2 minutes. If the smoke run is slower, report the timing (do not cut phases).

- [ ] **Step 5: Commit**

```bash
git add experiments/__init__.py experiments/run_study.py tests/test_run_study.py
git commit -m "Add end-to-end streaming study script"
```

---

### Task 7: Beam misalignment evaluation and LaTeX tables

**Files:**
- Create: `experiments/beam_eval.py`, `experiments/make_tables.py`
- Test: `tests/test_postprocess.py`

**Interfaces:**
- Consumes: `streaming_predictions.npz` / `streaming_study.json` layout from Task 6; `misalignment_beamwidths` from `src/beamforming.py`.
- Produces:
  - `beam_eval.main(argv)` → writes `<out>/beam_eval.json` = `{"bs_pos": [x, y], "bs_source": str, "n_tx": [...], "methods": {name: {"misalign_bw": [...per Nt], "outage": [...per Nt]}}}`.
  - `make_tables.main(argv)` → writes `<paper>/tables/accuracy.tex`, `compute.tex`, `generalisation.tex`, and `beam.tex` when `beam_eval.json` exists.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_postprocess.py
import json

import numpy as np
import pytest

from experiments import beam_eval, make_tables


def _fake_predictions(path):
    truth = np.stack([np.stack([np.full(15, 260.0), np.linspace(400, 420, 15)], 1)] * 2)
    np.savez(path, truth=truth, regimes=np.array(["cyclist", "vehicular"]), warmup=10,
             pred_hold=truth - [0.0, 1.5], pred_imm=truth + [0.0, 0.1])


def test_beam_eval_ranks_better_predictor_lower(tmp_path):
    _fake_predictions(tmp_path / "streaming_predictions.npz")
    beam_eval.main(["--results", str(tmp_path), "--bs-pos", "235", "350"])
    out = json.loads((tmp_path / "beam_eval.json").read_text())
    assert out["n_tx"] == [8, 16, 32, 64, 128] and out["bs_source"] == "cli"
    hold, imm = out["methods"]["hold"], out["methods"]["imm"]
    assert all(i < h for i, h in zip(imm["misalign_bw"], hold["misalign_bw"]))
    assert hold["misalign_bw"][-1] > hold["misalign_bw"][0]   # narrower beams, more beamwidths


def test_beam_eval_without_position_exits_cleanly(tmp_path, monkeypatch):
    """Review focus 5."""
    _fake_predictions(tmp_path / "streaming_predictions.npz")
    monkeypatch.setattr(beam_eval, "bs_from_deepmimo", lambda: None)
    with pytest.raises(SystemExit) as exc:
        beam_eval.main(["--results", str(tmp_path)])
    assert "--bs-pos" in str(exc.value)


def test_make_tables_writes_tex(tmp_path):
    study = {
        "classical": {"imm": {"test_mae": 0.31, "params": {}}, "kalman": {"test_mae": 0.35, "params": {}}},
        "learned": {"minlstm_seq_hybrid": {"test_mean": 0.28, "test_std": 0.01, "n_params": 13186,
                                           "arch": "minlstm_seq", "mode": "hybrid"}},
        "generalisation": [{"regime": "vehicular", "n_unseen": 5, "imm": 0.4, "minlstm_seq_hybrid": 0.38}],
        "bench": {"device": "cuda", "macs": {"minlstm_seq": 12800, "lstm_seq": 49792},
                  "latency_us": {"minlstm_seq@cuda": {"step": 90.0, "window_10": 300.0, "window_40": 310.0}},
                  "train_step": {"minlstm_seq": {"seq_lens": [16, 64], "ms": [1.0, 1.1]}}},
    }
    res = tmp_path / "results"
    res.mkdir()
    (res / "streaming_study.json").write_text(json.dumps(study))
    make_tables.main(["--results", str(res), "--paper", str(tmp_path / "paper")])
    tables = tmp_path / "paper" / "tables"
    acc = (tables / "accuracy.tex").read_text()
    assert "0.280" in acc and "IMM" in acc and r"\pm" in acc
    assert (tables / "compute.tex").exists() and (tables / "generalisation.tex").exists()
    assert not (tables / "beam.tex").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_postprocess.py -q`
Expected: FAIL with `ImportError: cannot import name 'beam_eval'`

- [ ] **Step 3a: Implement `experiments/beam_eval.py`**

```python
"""Beam misalignment of every predictor, in half-power beamwidths, Nt = 8..128.

    python experiments/beam_eval.py                       # BS position from DeepMIMO o1_60
    python experiments/beam_eval.py --bs-pos 235.5 350.2  # or given explicitly

Geometry only: a beam aimed at the predicted position, judged at the true one.
The SNR sweep through the ray-traced twin is notebook 6.
"""

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.beamforming import misalignment_beamwidths

N_TX = [8, 16, 32, 64, 128]


def bs_from_deepmimo():
    """BS (x, y) from the o1_60 scenario, or None if DeepMIMO is unavailable."""
    try:
        import deepmimo as dm
        bs0 = dm.load("o1_60")[0]
    except Exception:
        return None
    for key in ("tx_pos", "bs_pos", "txrx_pos"):
        try:
            return np.asarray(bs0[key], dtype=float).reshape(-1)[:2]
        except Exception:
            continue
    return None


def evaluate(pred, truth, warmup, bs_pos):
    p, t = pred[:, warmup:].reshape(-1, 2), truth[:, warmup:].reshape(-1, 2)
    rows = {"misalign_bw": [], "outage": []}
    for n in N_TX:
        mis = np.array([misalignment_beamwidths(a, b, bs_pos, n) for a, b in zip(p, t)])
        rows["misalign_bw"].append(float(mis.mean()))
        rows["outage"].append(float((mis > 0.5).mean()))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--bs-pos", nargs=2, type=float, default=None)
    args = p.parse_args(argv)
    results = pathlib.Path(args.results)

    if args.bs_pos is not None:
        bs_pos, source = np.array(args.bs_pos), "cli"
    else:
        bs_pos, source = bs_from_deepmimo(), "deepmimo:o1_60"
        if bs_pos is None:
            raise SystemExit("No base-station position: install DeepMIMO with the o1_60 "
                             "scenario, or pass --bs-pos X Y")

    data = np.load(results / "streaming_predictions.npz")
    truth, warmup = data["truth"], int(data["warmup"])
    methods = {k[len("pred_"):]: evaluate(data[k], truth, warmup, bs_pos)
               for k in data.files if k.startswith("pred_")}
    out = {"bs_pos": bs_pos.tolist(), "bs_source": source, "n_tx": N_TX, "methods": methods}
    (results / "beam_eval.json").write_text(json.dumps(out, indent=2))
    for name, rows in methods.items():
        print(f"{name:<26}" + "".join(f"{v:>8.3f}" for v in rows["misalign_bw"]))
    return out


if __name__ == "__main__":
    main()
```

- [ ] **Step 3b: Implement `experiments/make_tables.py`**

```python
"""JSON results -> LaTeX tables, so no number in the paper is typed by hand.

    python experiments/make_tables.py
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

LABELS = {
    "hold": "Reactive (hold)", "constant_velocity": "Constant velocity",
    "ls_linear": "LS line fit", "kalman": "Kalman (CV)", "imm": "IMM (CV+CT)",
    "lstm_seq_direct": "LSTM", "minlstm_seq_direct": "minLSTM",
    "minlstm_seq_res_direct": "minLSTM-res",
    "lstm_seq_hybrid": "IMM + LSTM", "minlstm_seq_hybrid": "IMM + minLSTM",
    "minlstm_seq_res_hybrid": "IMM + minLSTM-res",
}


def label(key):
    return LABELS.get(key, key.replace("_", r"\_"))


def table(caption, ref, columns, rows):
    head = " & ".join(columns) + r" \\"
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{ref}}}\n"
            f"\\begin{{tabular}}{{l{'r' * (len(columns) - 1)}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def accuracy(study):
    rows = [(label(k), f"{v['test_mae']:.3f}", "0") for k, v in study["classical"].items()]
    rows += [(label(k), f"{v['test_mean']:.3f} $\\pm$ {v['test_std']:.3f}", f"{v['n_params']:,}")
             for k, v in study["learned"].items()]
    return table("One-step prediction error on held-out trajectories (mean $\\pm$ std over seeds).",
                 "tab:accuracy", ["Method", "MAE (m)", "Params"], rows)


def compute(bench):
    rows = []
    for key, lat in bench.get("latency_us", {}).items():
        arch, dev = key.split("@")
        rows.append((f"{label(arch + '_direct')} ({dev})", f"{lat['step']:.1f}",
                     f"{lat['window_10']:.1f}", f"{lat['window_40']:.1f}",
                     f"{bench['macs'].get(arch, 0):,}"))
    return table("Per-slot inference cost: streaming step vs.\\ re-reading a $W$-slot window ($\\mu$s).",
                 "tab:compute", ["Model", "Step", "$W{=}10$", "$W{=}40$", "MACs/step"], rows)


def generalisation(rows_in):
    if not rows_in:
        return table("Held-out mobility regime.", "tab:generalisation", ["Method"], [])
    keys = [k for k in rows_in[0] if k not in ("regime", "n_unseen")]
    rows = [(label(k), *[f"{r[k]:.3f}" for r in rows_in]) for k in keys]
    return table("MAE (m) on a mobility regime never seen in training.", "tab:generalisation",
                 ["Method", *[r["regime"] for r in rows_in]], rows)


def beam(beam_json):
    rows = [(label(k), *[f"{v:.2f}" for v in m["misalign_bw"]])
            for k, m in beam_json["methods"].items()]
    return table("Mean beam misalignment (half-power beamwidths) vs.\\ array size.", "tab:beam",
                 ["Method", *[f"$N_t{{=}}{n}$" for n in beam_json["n_tx"]]], rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--paper", default=str(ROOT / "paper"))
    args = p.parse_args(argv)
    results, tables = pathlib.Path(args.results), pathlib.Path(args.paper) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    study = json.loads((results / "streaming_study.json").read_text())
    (tables / "accuracy.tex").write_text(accuracy(study))
    (tables / "compute.tex").write_text(compute(study.get("bench", {})))
    (tables / "generalisation.tex").write_text(generalisation(study.get("generalisation", [])))
    beam_path = results / "beam_eval.json"
    if beam_path.exists():
        (tables / "beam.tex").write_text(beam(json.loads(beam_path.read_text())))
    print(f"tables -> {tables}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_postprocess.py -q` then `python -m pytest -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add experiments/beam_eval.py experiments/make_tables.py tests/test_postprocess.py
git commit -m "Add beam misalignment evaluation and LaTeX table generation"
```

---

### Task 8 (controller, not a subagent): GCP runbook and paper skeleton

**Files:**
- Create: `docs/RUN_ON_GCP.md`, `paper/main.tex`, `paper/refs.bib`
- Modify: `README.md` (short "v3: streaming study" section), `requirements.txt` (no new deps expected)

- [ ] Write the runbook (VM creation, clone branch, install, `pytest`, `--quick`, full run in `tmux`, `beam_eval`, `make_tables`, commit results back).
- [ ] Write `paper/main.tex` (IEEEtran conference) that `\input`s `tables/*.tex`; results prose written only after real numbers exist.
- [ ] Run full test suite; commit; tell the user to push.
