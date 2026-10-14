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
