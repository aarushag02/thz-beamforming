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
        if compile:
            torch._dynamo.reset()  # else the recompile limit silently falls back to eager mid-sweep
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
