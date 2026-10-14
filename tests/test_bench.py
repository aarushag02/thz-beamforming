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
