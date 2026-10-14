"""Beam misalignment of every predictor, in half-power beamwidths, Nt = 8..128.

    python experiments/beam_eval.py                       # BS position from DeepMIMO o1_60
    python experiments/beam_eval.py --bs-pos 235.5 350.2  # or given explicitly

Geometry only: a beam aimed at the predicted position, judged at the true one.
The SNR sweep through the ray-traced twin is notebook 6.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.beamforming import misalignment_beamwidths

N_TX = [8, 16, 32, 64, 128]


def bs_from_deepmimo():
    """BS (x, y) from the o1_60 scenario, or None if DeepMIMO is unavailable."""
    try:
        import deepmimo as dm
        bs0 = dm.load("o1_60")[0]
    except Exception:
        return None
    for key in ("tx_pos", "bs_pos", "txrx_pos"):
        try:
            return np.asarray(bs0[key], dtype=float).reshape(-1)[:2]
        except Exception:
            continue
    return None


def evaluate(pred, truth, warmup, bs_pos):
    p, t = pred[:, warmup:].reshape(-1, 2), truth[:, warmup:].reshape(-1, 2)
    rows = {"misalign_bw": [], "outage": []}
    for n in N_TX:
        mis = np.array([misalignment_beamwidths(a, b, bs_pos, n) for a, b in zip(p, t)])
        rows["misalign_bw"].append(float(mis.mean()))
        rows["outage"].append(float((mis > 0.5).mean()))
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--bs-pos", nargs=2, type=float, default=None)
    args = p.parse_args(argv)
    results = pathlib.Path(args.results)

    if args.bs_pos is not None:
        bs_pos, source = np.array(args.bs_pos), "cli"
    else:
        bs_pos, source = bs_from_deepmimo(), "deepmimo:o1_60"
        if bs_pos is None:
            raise SystemExit("No base-station position: install DeepMIMO with the o1_60 "
                             "scenario, or pass --bs-pos X Y")

    data = np.load(results / "streaming_predictions.npz")
    truth, warmup = data["truth"], int(data["warmup"])
    methods = {k[len("pred_"):]: evaluate(data[k], truth, warmup, bs_pos)
               for k in data.files
               if k.startswith("pred_") and not re.search(r"_s\d+$", k)}
    out = {"bs_pos": bs_pos.tolist(), "bs_source": source, "n_tx": N_TX, "methods": methods}
    (results / "beam_eval.json").write_text(json.dumps(out, indent=2))
    for name, rows in methods.items():
        print(f"{name:<26}" + "".join(f"{v:>8.3f}" for v in rows["misalign_bw"]))
    return out


if __name__ == "__main__":
    main()
