"""Seeded training loop shared by every architecture.

One loop for all models so the comparison cannot drift: same optimiser, same
schedule, same early-stopping rule, same seeds. Differences in the results
table are then attributable to the architecture rather than to the harness.
"""

import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data import WindowDataset


def resolve_device(preferred=None):
    """cuda -> mps -> cpu. The existing notebooks hardcode mps, which fails on
    the GCP T4."""
    if preferred is not None:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mae_metres(pred, truth, scale):
    """Mean Euclidean error in metres, undoing the per-axis normalisation."""
    error = (np.asarray(pred) - np.asarray(truth)) * np.asarray(scale)
    return float(np.mean(np.linalg.norm(error, axis=1)))


def train_model(model, train_data, val_data, epochs=500, batch_size=64, lr=1e-3,
                weight_decay=0.0, device=None, patience=50, grad_clip=1.0,
                mode="parallel"):
    """Train one model, returning its history and restoring the best weights.

    `mode` selects the scan path for the minimal RNNs and is ignored by the
    LSTM variants, so every architecture can share this loop.
    """
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device)

    train_loader = DataLoader(WindowDataset(*train_data), batch_size=batch_size,
                              shuffle=True)
    val_loader = DataLoader(WindowDataset(*val_data), batch_size=batch_size)

    criterion = nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=max(1, patience // 3))

    history = {"train_loss": [], "val_loss": []}
    best_loss, best_state, stale = float("inf"), None, 0

    for _ in range(epochs):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb, mode=mode), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimiser.step()
            running += loss.item()
        history["train_loss"].append(running / len(train_loader))

        model.eval()
        running = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                running += criterion(model(xb.to(device), mode=mode),
                                     yb.to(device)).item()
        val_loss = running / len(val_loader)
        history["val_loss"].append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_loss - 1e-9:
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
def predict(model, X, device=None, mode="parallel", batch_size=1024):
    device = resolve_device(device) if not isinstance(device, torch.device) else device
    model = model.to(device).eval()
    X = torch.tensor(np.asarray(X), dtype=torch.float32)
    out = [model(X[i:i + batch_size].to(device), mode=mode).cpu().numpy()
           for i in range(0, len(X), batch_size)]
    return np.concatenate(out)


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
