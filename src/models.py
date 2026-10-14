"""The architectures under comparison, behind one interface.

Three LSTM-side entries rather than one, because "is minLSTM faster?" has two
different honest answers:

  lstm        nn.LSTM, dispatching to a fused cuDNN kernel -- what you deploy
  plain_lstm  the same recurrence in hand-written PyTorch -- the architecture-
              fair comparison, which is what the minLSTM paper itself used
  minlstm     the candidate

Comparing minlstm against cuDNN measures kernel engineering; comparing it
against plain_lstm measures the architecture. Report both.
"""

import torch
import torch.nn as nn

from src.minrnn import MinLSTMPredictor


class LSTMPredictor(nn.Module):
    """nn.LSTM + linear head. Matches the project's existing model exactly."""

    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, mode=None):
        out, _ = self.lstm(x)
        return self.fc(self.drop(out[:, -1, :]))


class PlainLSTMLayer(nn.Module):
    """One LSTM layer unrolled in Python, parameterised exactly like nn.LSTM.

    Same names and shapes as nn.LSTM's weight_ih_l0 / weight_hh_l0 / bias_ih_l0
    / bias_hh_l0, with gates in PyTorch's [input, forget, cell, output] order,
    so weights transfer across and the equivalence test can assert it.
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))
        bound = hidden_size ** -0.5
        for w in (self.weight_ih, self.weight_hh):
            nn.init.uniform_(w, -bound, bound)

    def forward(self, x):
        batch, steps, _ = x.shape
        h = x.new_zeros(batch, self.hidden_size)
        c = x.new_zeros(batch, self.hidden_size)

        gates_x = x @ self.weight_ih.T + self.bias_ih
        outputs = []
        for t in range(steps):
            gates = gates_x[:, t] + h @ self.weight_hh.T + self.bias_hh
            i, f, g, o = gates.chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1)


class PlainLSTMPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, output_size=2,
                 dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        sizes = [input_size] + [hidden_size] * num_layers
        self.layers = nn.ModuleList(
            PlainLSTMLayer(sizes[i], sizes[i + 1]) for i in range(num_layers)
        )
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def load_state_dict_from_cudnn(self, fused):
        """Copy weights across from an nn.LSTM-based model, for the
        equivalence test and for reusing a trained checkpoint."""
        with torch.no_grad():
            for n, layer in enumerate(self.layers):
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    getattr(layer, name).copy_(getattr(fused.lstm, f"{name}_l{n}"))
            self.fc.weight.copy_(fused.fc.weight)
            self.fc.bias.copy_(fused.fc.bias)

    def forward(self, x, mode=None):
        for n, layer in enumerate(self.layers):
            x = layer(x)
            if n < self.num_layers - 1:
                x = self.drop(x)
        return self.fc(self.drop(x[:, -1, :]))


MODELS = {
    "lstm": LSTMPredictor,
    "plain_lstm": PlainLSTMPredictor,
    # The paper's trainable method: log-space scan. The vanilla scan is kept
    # only as an ablation, because it loses its gradients past ~40 steps.
    "minlstm": MinLSTMPredictor,
    "minlstm_log": MinLSTMPredictor,
    "minlstm_vanilla": lambda **kw: MinLSTMPredictor(log_space=False, **kw),
}


def build_model(name, **kwargs):
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](**kwargs)
