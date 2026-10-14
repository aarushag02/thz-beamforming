"""Windowing, anchored normalisation and honest splits.

Two departures from 2_lstm_training.ipynb, both deliberate:

1. Inputs come from the *observed* (noisy) path while targets come from the
   *true* path. The base station never has exact coordinates, so predicting
   true position from noisy history is the real problem.
2. Splits hold out whole trajectories. The notebook split the tail 20% of each
   trajectory, but consecutive windows share W-1 of their W positions, so a
   validation window was nearly a copy of a training window and validation loss
   tracked training loss by construction.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.baselines import imm_stream


def make_windows(observed, true, window):
    """Slide a window over one trajectory.

    Anchoring on the last observed position turns "predict absolute position"
    into "predict displacement", which keeps the target around a metre
    regardless of where in a 550 m corridor the user is. The anchor is noisy,
    but it is noisy in a way the model can see and correct for, since it is
    also the last element of the input window.

    Returns X (N, window, 2), y (N, 2), anchors (N, 2).
    """
    observed = np.asarray(observed, dtype=float)[:, :2]
    true = np.asarray(true, dtype=float)[:, :2]
    n = len(observed) - window

    anchors = observed[window - 1:window - 1 + n]
    X = np.stack([observed[i:i + window] - anchors[i] for i in range(n)])
    y = true[window:window + n] - anchors
    return X, y, anchors


def fit_scale(X, percentile=99.0):
    """Per-axis scale from the training inputs, as a robust near-max.

    Fitted rather than hardcoded: the published [2.0, 18.0] came from three
    noiseless hand-drawn paths and does not transfer to a noisy corpus.
    """
    scale = np.percentile(np.abs(np.asarray(X)), percentile, axis=tuple(range(np.ndim(X) - 1)))
    return np.where(scale == 0, 1.0, scale)


def build_dataset(corpus, window, scales=None):
    """Window an entire corpus into flat arrays, fitting the scales if absent.

    Inputs and targets are scaled separately and deliberately. A W-step window
    spans tens of metres while the target is a single ~1.6 m step, so one
    shared scale squashes the target toward zero and recreates the vanishing-
    signal failure the project already solved once at W=10. The effect grows
    with W, which is exactly the direction this study is pushing.

    Returns X, y, anchors, {"x": scale_x, "y": scale_y}.
    """
    parts = [make_windows(t["observed"], t["true"], window) for t in corpus]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    anchors = np.concatenate([p[2] for p in parts])
    if scales is None:
        scales = {"x": fit_scale(X), "y": fit_scale(y)}
    return X / scales["x"], y / scales["y"], anchors, scales


def split_by_trajectory(corpus, fractions=(0.7, 0.15, 0.15), rng=None):
    """Hold out whole trajectories, never windows."""
    rng = rng or np.random.default_rng()
    order = rng.permutation(len(corpus))
    n_train = int(fractions[0] * len(corpus))
    n_val = int(fractions[1] * len(corpus))
    pick = lambda idx: [corpus[i] for i in idx]
    return (pick(order[:n_train]),
            pick(order[n_train:n_train + n_val]),
            pick(order[n_train + n_val:]))


def split_by_regime(corpus, held_out):
    """Train on some mobility bands, test on one the model never saw."""
    seen = [t for t in corpus if t["regime"] != held_out]
    unseen = [t for t in corpus if t["regime"] == held_out]
    return seen, unseen


class WindowDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(np.asarray(X), dtype=torch.float32)
        self.y = torch.tensor(np.asarray(y), dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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
