"""minLSTM (Feng et al., arXiv:2410.01201), with length-independence scaling.

Implements Algorithms 3 and 4 (vanilla) and the log-space formulation of
Appendix B, in both sequential and parallel-scan modes.

The two parameterisations are not the same model. The log-space variant passes
the candidate state through g() to keep it strictly positive so the recurrence
can be carried in logs; the vanilla variant does not. Each mode's parallel path
reproduces its own sequential path, and that is what the tests assert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def g(x):
    """Positive-valued activation from Appendix B: x + 0.5 for x >= 0, else sigmoid(x).

    Continuous at zero (both branches give 0.5) and strictly positive everywhere,
    which is what lets the recurrence be carried in log space.
    """
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    """log(g(x)), computed without ever evaluating log of a non-positive number.

    relu() in the first branch keeps the unselected side finite so that
    torch.where does not propagate NaN through the backward pass.
    """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def parallel_scan_log(log_coeffs, log_values):
    """Heinsen (2023) prefix scan for h_t = a_t * h_{t-1} + b_t, carried in logs.

    log_coeffs: (batch, steps, dim)     -- log(a_1 .. a_steps)
    log_values: (batch, steps + 1, dim) -- log(b_0 .. b_steps), b_0 being h_0

    Returns h_1 .. h_steps with shape (batch, steps, dim).
    """
    a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    return torch.exp(a_star + log_h0_plus_b_star)[:, 1:]


class MinLSTMLayer(nn.Module):
    """A single minLSTM layer.

    h_t = f'_t * h_{t-1} + i'_t * h~_t, where f' and i' are the sigmoid forget
    and input gates renormalised to sum to one. All three quantities depend on
    x_t alone and never on h_{t-1} -- that is what turns the recurrence into a
    prefix scan, and it is the whole source of the parallelism.
    """

    def __init__(self, input_size, hidden_size, log_space=False):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.log_space = log_space
        self.linear_f = nn.Linear(input_size, hidden_size)
        self.linear_i = nn.Linear(input_size, hidden_size)
        self.linear_h = nn.Linear(input_size, hidden_size)

    def normalised_gates(self, x):
        """Return (f', i') with f' + i' == 1, per Algorithm 3."""
        f = torch.sigmoid(self.linear_f(x))
        i = torch.sigmoid(self.linear_i(x))
        denom = f + i
        return f / denom, i / denom

    def candidate(self, x):
        raw = self.linear_h(x)
        return g(raw) if self.log_space else raw

    def _zero_state(self, batch, x):
        return torch.zeros(batch, self.hidden_size, dtype=x.dtype, device=x.device)

    def forward_sequential(self, x, h_0=None):
        """Unrolled recurrence: inference mode, and the reference the parallel
        paths are tested against."""
        batch, steps, _ = x.shape
        f_prime, i_prime = self.normalised_gates(x)
        h_tilde = self.candidate(x)

        h = self._zero_state(batch, x) if h_0 is None else h_0
        outputs = []
        for t in range(steps):
            h = f_prime[:, t] * h + i_prime[:, t] * h_tilde[:, t]
            outputs.append(h)
        return torch.stack(outputs, dim=1)

    def forward_parallel(self, x, h_0=None):
        """Same recurrence as a prefix scan over the sequence axis."""
        if self.log_space:
            return self._forward_parallel_log(x, h_0)
        return self._forward_parallel_vanilla(x, h_0)

    def _forward_parallel_vanilla(self, x, h_0=None):
        """Unrolling gives h_t = A_t * (h_0 + sum_{s<=t} i'_s h~_s / A_s) with
        A_t = prod_{s<=t} f'_s -- a cumulative product and a cumulative sum.

        A_t underflows once the sequence is long, which is exactly why the
        log-space path exists. Use this one for short sequences and for reading.
        """
        batch, _, _ = x.shape
        f_prime, i_prime = self.normalised_gates(x)
        h_tilde = self.candidate(x)

        a = torch.cumprod(f_prime, dim=1)
        h_0 = (torch.zeros(batch, 1, self.hidden_size, dtype=x.dtype, device=x.device)
               if h_0 is None else h_0.unsqueeze(1))
        return a * (h_0 + torch.cumsum((i_prime * h_tilde) / a, dim=1))

    def _forward_parallel_log(self, x, h_0=None):
        """Appendix B: gates and candidate straight to logs, then one scan.

        diff = softplus(-f_raw) - softplus(-i_raw) collapses the gate
        normalisation into a single softplus per gate, avoiding the explicit
        division by f + i.
        """
        batch, _, _ = x.shape
        diff = F.softplus(-self.linear_f(x)) - F.softplus(-self.linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        log_h_tilde = log_g(self.linear_h(x))

        if h_0 is None:
            # True zero initial state: log(0) = -inf contributes nothing to the
            # scan, and the padded position it occupies is sliced off anyway.
            log_h_0 = torch.full((batch, 1, self.hidden_size), float('-inf'),
                                 dtype=x.dtype, device=x.device)
        else:
            log_h_0 = log_g(h_0).unsqueeze(1)

        return parallel_scan_log(log_f, torch.cat([log_h_0, log_i + log_h_tilde], dim=1))


class MinLSTMPredictor(nn.Module):
    """Stacked minLSTM with a linear head -- drop-in for the project's LSTMPredictor.

    log_space defaults to True because the vanilla scan divides by cumprod(f'),
    which is ~1e-10 by 20 steps and NaN by 80 -- within the window lengths this
    project uses. The paper trains in log space for the same reason. Pass
    log_space=False only for the ablation, and only at short windows.

    Depth matters more here than in an LSTM. A single layer's gates depend only
    on x_t, so they are time-independent; it is stacking that makes layer two's
    gates see time-dependent inputs. The paper measures 37.6% vs 85.7% on
    selective copy going from one layer to two, so num_layers=1 is expected to
    be weak and is kept available only as an ablation.
    """

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1, log_space=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.log_space = log_space
        sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            MinLSTMLayer(sizes[i], sizes[i + 1], log_space=log_space)
            for i in range(num_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mode="parallel"):
        """mode='parallel' for the training scan, 'sequential' for step-by-step
        inference. Both must give the same answer; the tests enforce it."""
        if mode not in ("parallel", "sequential"):
            raise ValueError(f"unknown mode {mode!r}")
        for n, layer in enumerate(self.layers):
            run = layer.forward_parallel if mode == "parallel" else layer.forward_sequential
            x = run(x)
            if n < self.num_layers - 1:
                x = self.drop(x)
        return self.fc(self.drop(x[:, -1, :]))
