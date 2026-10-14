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


# --- CUDA (skipped without a GPU; these run on the VM) -----------------------

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@needs_cuda
@pytest.mark.parametrize("name", SEQ)
def test_cuda_streaming_step_matches_full_forward(name):
    torch.manual_seed(1)
    model = build_model(name, input_size=2, hidden_size=16, num_layers=2,
                        dropout=0.0).to("cuda").eval()
    x = torch.randn(3, 40, 2, device="cuda")
    with torch.no_grad():
        state = model.init_state(3, device="cuda")
        outs = []
        for k in range(x.shape[1]):
            y, state = model.step(x[:, k], state)
            outs.append(y)
        assert torch.allclose(torch.stack(outs, dim=1), model(x), atol=1e-4)


@needs_cuda
@pytest.mark.parametrize("residual_norm", [False, True])
def test_cuda_minlstm_seq_parallel_matches_sequential(residual_norm):
    torch.manual_seed(2)
    model = MinLSTMSeq(input_size=2, hidden_size=16, num_layers=3, dropout=0.0,
                       residual_norm=residual_norm).to("cuda").eval()
    x = torch.randn(2, 198, 2, device="cuda")
    with torch.no_grad():
        assert torch.allclose(model(x, mode="parallel"), model(x, mode="sequential"), atol=1e-4)


@needs_cuda
def test_cuda_minlstm_seq_gradients_are_finite():
    torch.manual_seed(5)
    model = build_model("minlstm_seq", input_size=2, hidden_size=16, num_layers=2,
                        dropout=0.0).to("cuda")
    x = torch.randn(8, 198, 2, device="cuda")
    model(x).sum().backward()
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
