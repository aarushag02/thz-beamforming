import torch

from src.minrnn import MinLSTMLayer


def test_normalised_gates_sum_to_one():
    """minLSTM's length-independence scaling requires f' + i' == 1 elementwise."""
    torch.manual_seed(0)
    layer = MinLSTMLayer(input_size=2, hidden_size=8)
    x = torch.randn(3, 5, 2)

    f_prime, i_prime = layer.normalised_gates(x)

    assert f_prime.shape == (3, 5, 8)
    assert torch.allclose(f_prime + i_prime, torch.ones_like(f_prime), atol=1e-6)


def _ema_layer(size=4):
    """A MinLSTMLayer rigged so that h_t = 0.5*h_{t-1} + 0.5*x_t.

    Zeroing the gate weights and biases gives f = i = sigmoid(0) = 0.5, hence
    f' = i' = 0.5; an identity h~ makes the closed form h_t = 1 - 0.5^t when
    x is all ones and h_0 = 0.
    """
    layer = MinLSTMLayer(input_size=size, hidden_size=size)
    with torch.no_grad():
        for lin in (layer.linear_f, layer.linear_i):
            lin.weight.zero_()
            lin.bias.zero_()
        layer.linear_h.weight.copy_(torch.eye(size))
        layer.linear_h.bias.zero_()
    return layer


def test_sequential_recurrence_matches_closed_form():
    layer = _ema_layer(size=4)
    steps = 6
    x = torch.ones(1, steps, 4)

    h = layer.forward_sequential(x)

    expected = torch.tensor([1.0 - 0.5 ** (t + 1) for t in range(steps)])
    assert h.shape == (1, steps, 4)
    assert torch.allclose(h[0, :, 0], expected, atol=1e-6)


def test_parallel_matches_sequential():
    """The prefix scan must reproduce the unrolled recurrence exactly."""
    torch.manual_seed(1)
    layer = MinLSTMLayer(input_size=3, hidden_size=16)
    x = torch.randn(4, 32, 3)

    h_parallel = layer.forward_parallel(x)
    h_sequential = layer.forward_sequential(x)

    assert h_parallel.shape == h_sequential.shape
    assert torch.allclose(h_parallel, h_sequential, atol=1e-5)


def test_log_space_parallel_matches_log_space_sequential():
    """Appendix B's log-space path must agree with its own unrolled form.

    Note it does NOT agree with the vanilla path: log-space applies g() to the
    candidate state to keep it positive, which is a different model, not just
    different arithmetic.
    """
    torch.manual_seed(2)
    layer = MinLSTMLayer(input_size=3, hidden_size=16, log_space=True)
    x = torch.randn(4, 32, 3)

    h_parallel = layer.forward_parallel(x)
    h_sequential = layer.forward_sequential(x)

    assert torch.allclose(h_parallel, h_sequential, atol=1e-5)


def test_log_space_is_finite_and_differentiable_at_long_sequences():
    """The reason log-space exists: 4096 steps of repeated gate multiplication."""
    torch.manual_seed(3)
    layer = MinLSTMLayer(input_size=2, hidden_size=32, log_space=True)
    x = torch.randn(2, 4096, 2, requires_grad=True)

    h = layer.forward_parallel(x)
    h.sum().backward()

    assert torch.isfinite(h).all(), "forward produced non-finite hidden states"
    assert torch.isfinite(x.grad).all(), "backward produced non-finite gradients"
    for name, p in layer.named_parameters():
        assert torch.isfinite(p.grad).all(), f"non-finite gradient in {name}"


def test_vanilla_parallel_underflows_where_log_space_survives():
    """Documents the failure mode log-space is there to avoid.

    The cumulative product of 4096 gates in (0,1) underflows to zero, so the
    vanilla scan divides by zero. If this ever starts passing, the vanilla path
    has become safe for long sequences and this test should be revisited.
    """
    torch.manual_seed(3)
    layer = MinLSTMLayer(input_size=2, hidden_size=32, log_space=False)
    x = torch.randn(2, 4096, 2)

    h = layer.forward_parallel(x)

    assert not torch.isfinite(h).all()


def test_predictor_parameter_count_matches_paper_ratio():
    """minLSTM should cost ~26% of the 50,818-param LSTM it replaces.

    Per layer: 3 Linears of (d_in -> d_h). Layer 1 is 3*(2*64+64)=576,
    layer 2 is 3*(64*64+64)=12480, head Linear(64,2)=130.
    """
    from src.minrnn import MinLSTMPredictor

    model = MinLSTMPredictor(input_size=2, hidden_size=64, num_layers=2, output_size=2)

    assert sum(p.numel() for p in model.parameters()) == 13_186


def test_predictor_returns_one_vector_per_sequence():
    from src.minrnn import MinLSTMPredictor

    torch.manual_seed(4)
    model = MinLSTMPredictor(input_size=2, hidden_size=32, num_layers=2, output_size=2)
    x = torch.randn(8, 10, 2)

    assert model(x).shape == (8, 2)


def test_predictor_inference_path_matches_training_path():
    """Training uses the scan, deployment unrolls step by step. A base station
    must get the same beam either way."""
    from src.minrnn import MinLSTMPredictor

    torch.manual_seed(5)
    model = MinLSTMPredictor(input_size=2, hidden_size=32, num_layers=2,
                             output_size=2, log_space=True)
    model.eval()
    x = torch.randn(8, 10, 2)

    with torch.no_grad():
        parallel = model(x, mode="parallel")
        sequential = model(x, mode="sequential")

    assert torch.allclose(parallel, sequential, atol=1e-5)


def test_vanilla_parallel_gradients_die_at_realistic_window_lengths():
    """Why log-space is the training default, not an option.

    The vanilla scan divides by cumprod(f'), which halves roughly every step:
    ~1e-10 by T=20 and ~1e-37 by T=80, at which point fp32 gradients are NaN.
    This bites at window lengths this project actually uses, not only at the
    4096-step extreme.
    """
    torch.manual_seed(0)
    layer = MinLSTMLayer(input_size=2, hidden_size=32, log_space=False)
    x = torch.randn(4, 80, 2, requires_grad=True)

    layer.forward_parallel(x).sum().backward()

    assert not torch.isfinite(x.grad).all()


def test_log_space_gradients_survive_the_same_window_length():
    torch.manual_seed(0)
    layer = MinLSTMLayer(input_size=2, hidden_size=32, log_space=True)
    x = torch.randn(4, 80, 2, requires_grad=True)

    layer.forward_parallel(x).sum().backward()

    assert torch.isfinite(x.grad).all()


def test_predictor_defaults_to_log_space():
    """The trainable default must be the numerically sound path."""
    from src.minrnn import MinLSTMPredictor

    assert MinLSTMPredictor().log_space is True
