import numpy as np
import torch

from src.data import build_dataset
from src.models import build_model
from src.train import mae_metres, resolve_device, set_seed, train_model
from src.trajectories import generate_corpus


def _tiny_split(window=10):
    corpus = generate_corpus(n=8, steps=80, rng=np.random.default_rng(0), sigma=0.5)
    Xtr, ytr, _, scales = build_dataset(corpus[:6], window)
    Xva, yva, _, _ = build_dataset(corpus[6:], window, scales=scales)
    return (Xtr, ytr), (Xva, yva), scales


def test_mae_metres_undoes_the_normalisation():
    scale = np.array([2.0, 4.0])
    pred = np.array([[1.0, 0.0], [0.0, 0.0]])
    truth = np.array([[0.0, 0.0], [0.0, 1.0]])

    # Errors in normalised space are (1,0) and (0,-1); in metres (2,0) and (0,-4).
    assert np.isclose(mae_metres(pred, truth, scale), 3.0)


def test_resolve_device_returns_a_usable_torch_device():
    device = resolve_device()

    assert isinstance(device, torch.device)
    torch.zeros(1, device=device)


def test_seeding_makes_two_training_runs_identical():
    train, val, _ = _tiny_split()

    losses = []
    for _ in range(2):
        set_seed(123)
        model = build_model("minlstm", hidden_size=16, num_layers=2)
        history = train_model(model, train, val, epochs=3, batch_size=32, device="cpu")
        losses.append(history["val_loss"])

    assert np.allclose(losses[0], losses[1])


def test_training_reduces_validation_loss():
    train, val, _ = _tiny_split()
    set_seed(0)
    model = build_model("minlstm", hidden_size=32, num_layers=2)

    history = train_model(model, train, val, epochs=25, batch_size=32, device="cpu")

    assert history["val_loss"][-1] < history["val_loss"][0]


def test_early_stopping_halts_before_the_epoch_budget():
    train, val, _ = _tiny_split()
    set_seed(0)
    model = build_model("minlstm", hidden_size=16, num_layers=2)

    history = train_model(model, train, val, epochs=200, batch_size=32,
                          device="cpu", patience=3)

    assert len(history["val_loss"]) < 200
