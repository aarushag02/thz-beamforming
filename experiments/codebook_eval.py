"""Hybrid beamforming with a DFT codebook, against fully-digital MRT.

    python experiments/codebook_eval.py                      # Nt=64, 64 and 128 beams
    python experiments/codebook_eval.py --oversample 1 --n-rf 1 2 4

A 64-element mmWave array is driven through phase shifters, not one DAC per
element, so the transmit beam is f = F_RF F_BB:

  F_RF  (Nt, N_RF)  analog precoder: N_RF columns of a DFT codebook (constant modulus)
  F_BB  (N_RF,)     digital precoder: MRT on the effective channel h^T F_RF

Each method chooses its N_RF beams -- and sets F_BB -- from the channel it looked
up (twin at the predicted position, true slot-t channel for reactive, true t+1
channel for the oracle), then is scored on the TRUE channel at t+1, as in
snr_eval.py. With one user and one receive antenna, N_RF = 1 is analog beam
selection and F_BB is a scalar; N_RF > 1 combines beams to recover multipath gain.

Beam training. A predictor does not have to trust its single best beam: it can
measure its top-k candidate beams on the live channel and keep the strongest. The
alternative is an exhaustive sweep of all K beams every update, which finds the
best beam for the slot but spends K measurements. With tau = (time to measure one
beam) / (beam-update interval), effective spectral efficiency is

    SE_eff = (1 - k * tau) * log2(1 + SNR)

tau is a deployment assumption, so it is swept. In 5G NR FR2 one SSB is 4 OFDM
symbols at 120 kHz SCS (~36 us): 3.6e-4 of the 100 ms update interval this corpus
uses, 1.8e-3 at 20 ms updates, ~1e-2 at the few-ms updates fast users need.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.snr_eval import (REFERENCE_NT, beam_snr, deepmimo_channels, load_twin,
                                  lookup_channels, reference_noise, summarise)
from src.twin import DigitalTwin

TAUS = [3.6e-4, 1.8e-3, 1e-2]


def dft_codebook(n_tx, oversample=1):
    """(n_tx, K) unit-norm steering vectors, K = oversample * n_tx, spatial
    frequencies u_k = -1 + 2k/K spanning the whole [-1, 1) visible region."""
    u = -1.0 + 2.0 * np.arange(oversample * n_tx) / (oversample * n_tx)
    return np.exp(1j * np.pi * np.outer(np.arange(n_tx), u)) / np.sqrt(n_tx)


def beam_ranking(h, W):
    """Codebook indices ordered strongest-first for each row of h (M, Nt)."""
    return np.argsort(-np.abs(h @ W), axis=1)


def hybrid_precoder(h_src, W, n_rf):
    """f = F_RF F_BB, unit norm, built entirely from h_src (M, Nt).

    F_RF takes the n_rf strongest codebook beams for h_src; F_BB is MRT on the
    effective channel h_src^T F_RF, i.e. its conjugate. Received signal is h . f,
    the same convention as src.beamforming.snr.
    """
    top = beam_ranking(h_src, W)[:, :n_rf]               # (M, n_rf)
    F_rf = W[:, top].transpose(1, 0, 2)                  # (M, Nt, n_rf)
    h_eff = np.einsum("mn,mnr->mr", h_src, F_rf)
    f = np.einsum("mnr,mr->mn", F_rf, np.conj(h_eff))
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return np.divide(f, norm, out=np.zeros_like(f), where=norm > 0)


def precoder_snr(f, h_true, noise_power):
    return np.abs(np.sum(h_true * f, axis=1)) ** 2 / noise_power


def refine_top_k(h_src, h_true, W, k):
    """Measure the k strongest beams for h_src on the live channel; keep the best.

    Returns the chosen codebook index per slot. k = 1 is "trust the prediction".
    """
    cand = beam_ranking(h_src, W)[:, :k]                         # (M, k)
    measured = np.abs(np.einsum("mn,nmk->mk", h_true, W[:, cand]))
    return cand[np.arange(len(cand)), np.argmax(measured, axis=1)]


def evaluate_codebook(chans, valid, regimes, noise_power, W, n_rf_list, top_k, taus):
    """All codebook metrics for one codebook W, from lookup_channels output."""
    h_true = chans["oracle"]
    n_beams = W.shape[1]
    best = beam_ranking(h_true, W)[:, 0]                 # the beam an oracle would pick

    hybrid = {}
    for n_rf in n_rf_list:
        snr_lin = {name: precoder_snr(hybrid_precoder(h, W, n_rf), h_true, noise_power)
                   for name, h in chans.items()}
        metrics = summarise(snr_lin, valid, regimes)
        if n_rf == 1:
            first = {name: beam_ranking(h, W)[:, 0] for name, h in chans.items()}
            mask = valid.reshape(-1)
            for name, m in metrics.items():
                m["beam_match"] = float(np.mean(first[name][mask] == best[mask]))
        hybrid[str(n_rf)] = metrics

    # Beam training: exhaustive sweep vs measuring a predictor's top-k beams.
    mask = valid.reshape(-1)

    def row(beam_idx, measured):
        s = precoder_snr(W[:, beam_idx].T, h_true, noise_power)[mask]
        se = float(np.mean(np.log2(1 + s)))
        return {"beams_measured": int(measured),
                "snr_db": float(np.mean(10 * np.log10(np.maximum(s, 1e-3)))),
                "se": se, "beam_match": float(np.mean(beam_idx[mask] == best[mask])),
                "se_eff": {f"{t:g}": max(0.0, 1 - measured * t) * se for t in taus}}

    training = {"exhaustive": row(best, n_beams)}
    for name, h in chans.items():
        if name == "oracle":
            continue
        training[name] = {str(k): row(refine_top_k(h, h_true, W, k), k) for k in top_k}
    return {"n_beams": int(n_beams), "hybrid": hybrid, "training": training}


def print_summary(n_tx, K, res, mrt):
    print(f"\nNt = {n_tx}, DFT codebook K = {K} beams")
    rfs = list(res["hybrid"])
    head = "".join(f"{'N_RF=' + r:>15}" for r in rfs) + f"{'digital MRT':>15}"
    print(f"{'SNR gain vs reactive (dB)':<26}{head}{'beam match':>12}")
    for name, m in res["hybrid"][rfs[0]].items():
        cells = "".join(f"{res['hybrid'][r][name]['gain_db']:>+15.2f}" for r in rfs)
        print(f"{name:<26}{cells}{mrt[name]['gain_db']:>+15.2f}{100 * m['beam_match']:>11.0f}%")
    taus = list(res["training"]["exhaustive"]["se_eff"])
    print(f"\n{'beam training':<26}{'beams':>6}{'SNR dB':>8}{'SE':>7}"
          + "".join(f"{'SE_eff t=' + t:>16}" for t in taus))

    def line(label, r):
        print(f"{label:<26}{r['beams_measured']:>6}{r['snr_db']:>8.2f}{r['se']:>7.2f}"
              + "".join(f"{r['se_eff'][t]:>16.2f}" for t in taus))
    line("exhaustive sweep", res["training"]["exhaustive"])
    for name in ("reactive", "imm", "minlstm_seq_hybrid", "lstm_seq_hybrid"):
        for k, r in res["training"].get(name, {}).items():
            line(f"{name} top-{k}", r)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--scenarios", default=str(ROOT / "notebooks" / "deepmimo_scenarios"))
    p.add_argument("--n-tx", type=int, default=64)
    p.add_argument("--oversample", nargs="+", type=int, default=[1, 2])
    p.add_argument("--n-rf", nargs="+", type=int, default=[1, 2, 4])
    p.add_argument("--top-k", nargs="+", type=int, default=[1, 3, 5])
    p.add_argument("--tau", nargs="+", type=float, default=TAUS)
    args = p.parse_args(argv)
    if 1 not in args.n_rf:
        args.n_rf = [1] + args.n_rf        # beam match is defined on the analog beam
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
    mrt = summarise({n: beam_snr(h, chans["oracle"], noise) for n, h in chans.items()},
                    valid, regimes)

    out = {"n_tx": args.n_tx, "bs_pos": bs_pos.tolist(), "noise_power": noise,
           "taus": args.tau, "top_k": args.top_k, "n_rf": args.n_rf, **info,
           "mrt": mrt, "codebooks": {}}
    for os_ in args.oversample:
        W = dft_codebook(args.n_tx, os_)
        res = evaluate_codebook(chans, valid, regimes, noise, W, args.n_rf, args.top_k, args.tau)
        out["codebooks"][str(W.shape[1])] = res
        print_summary(args.n_tx, W.shape[1], res, mrt)
    (results / "codebook_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n{info['n_slots']:,} slots -> {results / 'codebook_eval.json'}")
    return out


if __name__ == "__main__":
    main()
