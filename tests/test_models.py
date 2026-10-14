import torch

from src.models import MODELS, LSTMPredictor, PlainLSTMPredictor, build_model


def test_cudnn_lstm_matches_the_published_parameter_count():
    """Continuity with the existing results: README reports 50,818."""
    model = LSTMPredictor(hidden_size=64, num_layers=2)

    assert sum(p.numel() for p in model.parameters()) == 50_818


def test_plain_lstm_reproduces_cudnn_lstm_given_the_same_weights():
    """The timing study compares a fused cuDNN kernel against hand-written
    PyTorch. That is only meaningful if they compute the same function."""
    torch.manual_seed(0)
    fused = LSTMPredictor(hidden_size=32, num_layers=2, dropout=0.0).eval()
    plain = PlainLSTMPredictor(hidden_size=32, num_layers=2, dropout=0.0).eval()
    plain.load_state_dict_from_cudnn(fused)

    x = torch.randn(6, 16, 2)
    with torch.no_grad():
        assert torch.allclose(fused(x), plain(x), atol=1e-5)


def test_plain_and_fused_lstm_have_identical_parameter_counts():
    fused = LSTMPredictor(hidden_size=64, num_layers=2)
    plain = PlainLSTMPredictor(hidden_size=64, num_layers=2)

    assert (sum(p.numel() for p in fused.parameters())
            == sum(p.numel() for p in plain.parameters()))


def test_registry_builds_every_architecture_with_a_common_interface():
    x = torch.randn(4, 10, 2)

    # The *_seq models emit a prediction at every step, (B, L, 2); their
    # interface is covered in test_seq_models.py.
    for name in (n for n in MODELS if not n.endswith("_seq") and not n.endswith("_seq_res")):
        model = build_model(name, hidden_size=32, num_layers=2)
        assert model(x).shape == (4, 2), f"{name} returned the wrong shape"
