"""Robustness of the predictive gain to per-element phase errors in the transmit beam.

    python experiments/phase_noise_eval.py
    python experiments/phase_noise_eval.py --sigma 0 0.1 0.3 --draws 20

Every method -- reactive, oracle and every predictor alike -- forms its beam f
exactly as in snr_eval.py (fully-digital MRT) or codebook_eval.py (analog DFT beam,
N_RF = 1). The hardware then applies it with an independent phase error on each
element,

    f_k  ->  f_k * exp(j phi_k),   phi_k ~ N(0, sigma^2),

and the SNR is measured on the TRUE next-slot channel, as before. This models
phase-shifter and calibration error, which is what an independent per-element
phase error physically is. It is not oscillator phase noise: the local oscillator
is shared by every element, a common phase rotation leaves |h . f|^2 unchanged,
and its real cost (inter-carrier interference) needs more than one subcarrier.

Within a draw every method receives the same phi, so differences between methods
reflect the methods, not the luck of the draw. For a matched beam the mean
array-gain loss is -10 log10(e^{-s^2} + (1 - e^{-s^2}) sum_k a_k^2), a_k being each
element's share of channel power; the sum is O(1/Nt), so the loss tends to
4.34 sigma^2 dB for large arrays and is slightly smaller at Nt = 64. The
"expected_loss_db" field records that large-array value as a sanity check.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.codebook_eval import dft_codebook, hybrid_precoder, precoder_snr
from experiments.snr_eval import (REFERENCE_NT, deepmimo_channels, load_twin, lookup_channels,
                                  reference_noise, summarise)
from src.twin import DigitalTwin

SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]


def mrt_precoder(h):
    """Unit-norm conj(h) per row; an all-zero channel gives an all-zero beam."""
    f = np.conj(h)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return np.divide(f, norm, out=np.zeros_like(f), where=norm > 0)


def apply_phase_error(f, phi):
    """Rotate each element of each beam by phi (same shape as f). Norm is unchanged."""
    return f * np.exp(1j * phi)


def expected_loss_db(sigma):
    """Large-array limit of the mean matched-beam loss, ~4.34 sigma^2 dB."""
    return float(10 * np.log10(np.e) * sigma ** 2)


def _average(dicts):
    """Element-wise mean of identically-shaped nested dicts of floats."""
    first = dicts[0]
    if isinstance(first, dict):
        return {k: _average([d[k] for d in dicts]) for k in first}
    return float(np.mean(dicts))


def evaluate_phase_noise(chans, valid, regimes, noise_power, precoder, sigmas, draws, seed=0):
    """sigma -> metrics (summarise() format, averaged over draws) for one precoder.

    precoder: (M, Nt) channel -> (M, Nt) unit-norm beam, applied to each method's
    looked-up channel; the result is scored on chans["oracle"].
    """
    h_true = chans["oracle"]
    beams = {name: precoder(h) for name, h in chans.items()}
    out = {}
    for sigma in sigmas:
        runs = []
        for d in range(1 if sigma == 0 else draws):
            phi = np.random.default_rng([seed, d]).normal(0.0, sigma, size=h_true.shape)
            snr_lin = {name: precoder_snr(apply_phase_error(f, phi), h_true, noise_power)
                       for name, f in beams.items()}
            runs.append(summarise(snr_lin, valid, regimes))
        metrics = _average(runs)
        metrics["expected_loss_db"] = expected_loss_db(sigma)
        out[f"{sigma:g}"] = metrics
    return out


def print_table(name, res, methods=("imm", "minlstm_seq_hybrid", "lstm_seq_hybrid")):
    print(f"\n{name}")
    head = "".join(f"{m[:18] + ' gain':>24}" for m in methods)
    print(f"{'sigma':>6}{'expected':>10}{'oracle loss':>13}{'reactive SNR':>14}{head}{'gap %':>7}")
    base = res["0"]
    for sigma, r in res.items():
        if sigma == "expected_loss_db":
            continue
        gains = "".join(f"{r[m]['gain_db']:>+17.2f} ±{r[m]['gain_db_ci95']:.2f}"
                        for m in methods if m in r)
        print(f"{sigma:>6}{r['expected_loss_db']:>10.2f}"
              f"{base['oracle']['snr_db'] - r['oracle']['snr_db']:>13.2f}"
              f"{r['reactive']['snr_db']:>14.2f}{gains}"
              f"{100 * r['minlstm_seq_hybrid']['oracle_gap_recovered']:>7.0f}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--scenarios", default=str(ROOT / "notebooks" / "deepmimo_scenarios"))
    p.add_argument("--n-tx", type=int, default=64)
    p.add_argument("--oversample", nargs="+", type=int, default=[1, 2])
    p.add_argument("--sigma", nargs="+", type=float, default=SIGMAS)
    p.add_argument("--draws", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    results = pathlib.Path(args.results)

    data = np.load(results / "streaming_predictions.npz")
    truth, warmup, regimes = data["truth"], int(data["warmup"]), data["regimes"]
    preds = {k[len("pred_"):]: data[k] for k in data.files
             if k.startswith("pred_") and not re.search(r"_s\d+$", k)}

    bs, grid, bs_pos = load_twin(args.scenarios)
    nearest = DigitalTwin(grid, channel_fn=None).nearest_indices
    noise = reference_noise(truth, warmup, nearest, deepmimo_channels(bs, REFERENCE_NT))
    chans, valid, info = lookup_channels(preds, truth, warmup, nearest,
                                         deepmimo_channels(bs, args.n_tx))

    precoders = {"mrt": mrt_precoder}
    for os_ in args.oversample:
        W = dft_codebook(args.n_tx, os_)
        precoders[f"dft{W.shape[1]}"] = lambda h, W=W: hybrid_precoder(h, W, 1)

    out = {"n_tx": args.n_tx, "sigmas": args.sigma, "draws": args.draws, "seed": args.seed,
           "noise_power": noise, "model": "iid per-element phase error on the transmit beam, "
           "same draw for every method", **info, "precoders": {}}
    for name, fn in precoders.items():
        res = evaluate_phase_noise(chans, valid, regimes, noise, fn, args.sigma, args.draws, args.seed)
        out["precoders"][name] = res
        print_table(f"Nt = {args.n_tx}, {name}", res)
    (results / "phase_noise_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n-> {results / 'phase_noise_eval.json'}")
    return out


if __name__ == "__main__":
    main()
