"""SNR and spectral efficiency of every predictor through the ray-traced O1_60 twin.

    python experiments/snr_eval.py                       # Nt = 64
    python experiments/snr_eval.py --n-tx 8 16 32 64 128 # array sweep

Every method aims an MRT beam for slot t+1 and is scored on the TRUE channel at t+1:

  reactive  beam from the channel at the user's true position at slot t, i.e. the
            CSI a beam sweep at slot t would give, one slot stale when it is used
  <method>  beam from the twin's channel at that method's predicted position
  oracle    beam from the true channel at t+1 -- the ceiling no predictor can pass

Two departures from 3_beamforming.ipynb, both deliberate. The notebook scored its
reactive beam on the same slot it was built from, so it never paid for staleness;
and it resolved predictions against the trajectory's own points up to t+1, which
returned the true future channel. Here channels come from the full 0.2 m grid
(src/twin.py) and every beam is judged one slot after it was formed.

Noise is fixed across array sizes (main's rule: mean per-element channel power at
Nt=8, divided by 100), so a larger array shows its array gain rather than having it
calibrated away.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.twin import DigitalTwin

REFERENCE_NT = 8          # array size the noise power is calibrated at
REFERENCE_SNR_DB = 20.0   # per-element SNR at REFERENCE_NT, as in 3_beamforming.ipynb
FLOOR_DB = -30.0          # a beam aimed at a grid point with no paths gets this, not -inf
OUTAGE_DB = 3.0           # slot counts as an outage when it is this far below the oracle


def beam_snr(h_beam, h_true, noise_power):
    """Linear SNR of MRT weights built from h_beam, received through h_true.

    Row-wise, for (M, Nt) arrays: |h_true . conj(h_beam)|^2 / ||h_beam||^2 / N0.
    Same maths as src.beamforming.mrt_weights + snr, vectorised over slots. A beam
    built from an all-zero channel transmits nothing and scores zero.
    """
    norm2 = np.sum(np.abs(h_beam) ** 2, axis=1)
    gain = np.abs(np.sum(h_true * np.conj(h_beam), axis=1)) ** 2
    return np.divide(gain, norm2 * noise_power, out=np.zeros_like(norm2), where=norm2 > 0)


def _mean_ci(per_traj):
    """Mean over trajectories and a 95% normal interval half-width."""
    per_traj = per_traj[np.isfinite(per_traj)]
    half = 1.96 * per_traj.std(ddof=1) / np.sqrt(len(per_traj)) if len(per_traj) > 1 else 0.0
    return float(per_traj.mean()), float(half)


def lookup_channels(preds, truth, warmup, nearest, channel_fn):
    """Channel each method would beamform from, plus the true channel it meets.

    preds:      name -> (N, L, 2) predicted position of slot t+1 at sequence index k
    truth:      (N, L, 2) true position of slot t+1, so truth[:, k-1] is slot t
    nearest:    (M, 2) positions -> (M,) grid indices
    channel_fn: sorted unique grid indices (U,) -> channels (U, Nt)

    Returns ({name: (M, Nt)} with "oracle" the true t+1 channel and "reactive" the
    true slot-t channel, valid mask (N, S), info). Only k >= warmup is kept, the
    index set every accuracy metric uses.
    """
    n_traj, length, _ = truth.shape
    positions = {"oracle": truth[:, warmup:], "reactive": truth[:, warmup - 1:length - 1]}
    positions.update({name: p[:, warmup:] for name, p in preds.items()})

    idx = {name: nearest(p.reshape(-1, 2)) for name, p in positions.items()}
    unique = np.unique(np.concatenate(list(idx.values())))
    H = np.asarray(channel_fn(unique))
    chans = {name: H[np.searchsorted(unique, ix)] for name, ix in idx.items()}

    # A true position with no ray paths has no channel to beamform at; no method can
    # score there, so those slots are dropped for everyone alike.
    valid = (np.sum(np.abs(chans["oracle"]) ** 2, axis=1) > 0).reshape(n_traj, length - warmup)
    return chans, valid, {"n_slots": int(valid.size), "n_dropped": int((~valid).sum()),
                          "n_grid_points": int(len(unique))}


def summarise(snr_lin, valid, regimes):
    """Per-method SNR / SE / gain metrics from linear per-slot SNRs (M,).

    snr_lin must hold "oracle" and "reactive"; gains are relative to "reactive",
    outage to "oracle", and gain_vs_imm is added when "imm" is present.
    """
    n_traj, slots = valid.shape
    snr_db, se = {}, {}
    for name, s in snr_lin.items():
        s = np.asarray(s).reshape(n_traj, slots)
        snr_db[name] = np.where(valid, 10 * np.log10(np.maximum(s, 10 ** (FLOOR_DB / 10))), np.nan)
        se[name] = np.where(valid, np.log2(1 + s), np.nan)

    regimes = np.asarray(regimes)
    react, orc = snr_db["reactive"], snr_db["oracle"]
    oracle_gap = np.nanmean(orc) - np.nanmean(react)
    out = {}
    for name in snr_lin:
        gain = snr_db[name] - react
        gain_mean, gain_ci = _mean_ci(np.nanmean(gain, axis=1))
        m = {
            "snr_db": float(np.nanmean(snr_db[name])),
            "se": float(np.nanmean(se[name])),
            "gain_db": gain_mean, "gain_db_ci95": gain_ci,
            "se_gain": float(np.nanmean(se[name] - se["reactive"])),
            "oracle_gap_recovered": float((np.nanmean(snr_db[name]) - np.nanmean(react)) / oracle_gap)
                                    if oracle_gap > 0 else float("nan"),
            "outage": float(np.nanmean(np.where(valid, orc - snr_db[name] > OUTAGE_DB, np.nan))),
            "gain_db_by_regime": {str(r): float(np.nanmean(gain[regimes == r]))
                                  for r in np.unique(regimes)},
        }
        if "imm" in snr_db and name not in ("imm", "oracle", "reactive"):
            m["gain_vs_imm_db"], m["gain_vs_imm_db_ci95"] = _mean_ci(
                np.nanmean(snr_db[name] - snr_db["imm"], axis=1))
        out[name] = m
    return out


def evaluate(preds, truth, warmup, regimes, nearest, channel_fn, noise_power):
    """Fully-digital MRT: each method's beam is conj(h) of the channel it looked up,
    scored on the true next-slot channel. Arguments as in lookup_channels."""
    chans, valid, info = lookup_channels(preds, truth, warmup, nearest, channel_fn)
    snr_lin = {name: beam_snr(h, chans["oracle"], noise_power) for name, h in chans.items()}
    return summarise(snr_lin, valid, regimes), info


def load_twin(scenarios_dir):
    """O1_60 TX set 10 / tx_idx 0 over RX set 0 -- bs0 in 1_data_generation.ipynb.

    Loading one TX/RX pair needs ~1 GB; the full scenario needs tens of GB.
    """
    import deepmimo as dm
    dm.config.set("scenarios_folder", str(scenarios_dir))
    bs = dm.load("o1_60", tx_sets={10: [0]}, rx_sets={0: "all"})
    return bs, np.asarray(bs["rx_pos"])[:, :2], np.asarray(bs["tx_pos"]).reshape(-1)[:2]


def deepmimo_channels(bs, n_tx):
    """Grid indices -> (U, n_tx) narrowband channels from an n_tx-element ULA."""
    import deepmimo as dm

    def fn(indices):
        params = dm.ChannelParameters()
        params.bs_antenna.shape = np.array([n_tx, 1])
        H = np.asarray(bs.trim(idxs=list(map(int, indices))).compute_channels(params))
        if H.shape[2] != n_tx:   # a silent fallback to 8 elements would fake the sweep
            raise RuntimeError(f"asked DeepMIMO for {n_tx} TX antennas, got {H.shape[2]}")
        return H[:, 0, :, 0]
    return fn


def reference_noise(truth, warmup, nearest, channel_fn):
    """Main's calibration: mean per-element power at the true positions / 100."""
    unique = np.unique(nearest(truth[:, warmup:].reshape(-1, 2)))
    return float(np.mean(np.abs(channel_fn(unique)) ** 2) / 10 ** (REFERENCE_SNR_DB / 10))


def print_table(n_tx, metrics):
    print(f"\nNt = {n_tx}")
    print(f"{'method':<24}{'SNR dB':>8}{'gain dB':>14}{'SE':>7}{'dSE':>7}{'gap %':>7}{'outage':>8}")
    for name, m in metrics.items():
        print(f"{name:<24}{m['snr_db']:>8.2f}{m['gain_db']:>+8.2f} ±{m['gain_db_ci95']:.2f}"
              f"{m['se']:>7.2f}{m['se_gain']:>+7.2f}{100 * m['oracle_gap_recovered']:>7.0f}"
              f"{100 * m['outage']:>7.1f}%")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--scenarios", default=str(ROOT / "notebooks" / "deepmimo_scenarios"))
    p.add_argument("--n-tx", nargs="+", type=int, default=[64])
    args = p.parse_args(argv)
    results = pathlib.Path(args.results)

    data = np.load(results / "streaming_predictions.npz")
    truth, warmup, regimes = data["truth"], int(data["warmup"]), data["regimes"]
    preds = {k[len("pred_"):]: data[k] for k in data.files
             if k.startswith("pred_") and not re.search(r"_s\d+$", k)}

    bs, grid, bs_pos = load_twin(args.scenarios)
    nearest = DigitalTwin(grid, channel_fn=None).nearest_indices
    noise = reference_noise(truth, warmup, nearest, deepmimo_channels(bs, REFERENCE_NT))

    out = {"bs_pos": bs_pos.tolist(), "noise_power": noise,
           "noise_rule": f"mean per-element |h|^2 at Nt={REFERENCE_NT} / {10 ** (REFERENCE_SNR_DB / 10):g}",
           "n_tx": args.n_tx, "warmup": warmup, "n_trajectories": int(len(truth)), "results": {}}
    for n_tx in args.n_tx:
        metrics, info = evaluate(preds, truth, warmup, regimes, nearest,
                                 deepmimo_channels(bs, n_tx), noise)
        out["results"][str(n_tx)] = metrics
        out.update(info)
        print_table(n_tx, metrics)
    (results / "snr_eval.json").write_text(json.dumps(out, indent=2))
    print(f"\n{info['n_slots']:,} slots scored ({info['n_dropped']} dropped: no paths at the true "
          f"position), {info['n_grid_points']:,} grid points -> {results / 'snr_eval.json'}")
    return out


if __name__ == "__main__":
    main()
