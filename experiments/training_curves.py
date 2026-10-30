"""Learning curves in metres for the four reported models, every seed, plus a reproduction check.

    python experiments/training_curves.py              # full: 4 models x 5 seeds, ~30 min on a laptop GPU
    python experiments/training_curves.py --smoke      # wiring check
    python experiments/training_curves.py --plot-only  # redraw from results/training_curves.json

Also draws the tuning history (best validation error per Optuna trial) from the search re-run in
results/search_curves.json (experiments/search_curves.py).

Retrains LSTM, minLSTM, IMM + LSTM and IMM + minLSTM with the hyperparameters, seeds, corpus
and split of results/streaming_study.json, and after every epoch records the train and
validation position error in metres (the metric the paper reports). Losses are normalised
differently in direct and hybrid mode, so metres are the only scale on which all four
models and the tuned IMM can share one axis. The final test MAE of each seed is compared
with the study's, so this run doubles as a check of the published accuracy numbers.
"""

import argparse
import json
import logging
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Helvetica absent -> Arial

from experiments.make_poster_figures import ACCENT, DARK, INK, MID, save
from experiments.run_study import CORPUS, SCALES, evaluate, log, make_model, save_json
from experiments.search_curves import build_datasets
from src.data import WARMUP
from src.train import resolve_device, set_seed, train_seq_model

KEYS = ["lstm_seq_direct", "minlstm_seq_direct", "lstm_seq_hybrid", "minlstm_seq_hybrid"]
NAMES = {"lstm_seq_direct": "LSTM", "minlstm_seq_direct": "minLSTM",
         "lstm_seq_hybrid": "IMM + LSTM", "minlstm_seq_hybrid": "IMM + minLSTM"}
# proposed in the accent, its LSTM counterpart in grey; direct networks dashed
STYLE = {"lstm_seq_direct": dict(color=MID, ls="--"), "minlstm_seq_direct": dict(color="#7fa3d1", ls="--"),
         "lstm_seq_hybrid": dict(color=MID, ls="-"), "minlstm_seq_hybrid": dict(color=ACCENT, ls="-")}


def run_seed(arch, params, data, cfg, device, seed):
    """Train one model, recording train/val MAE (m) and wall time after every epoch."""
    set_seed(seed)
    model = make_model(arch, params, data["train"]["X"].shape[-1])
    rec = dict(train_mae=[], val_mae=[], seconds=[])
    t0 = time.time()

    def callback(epoch, val_loss):
        rec["seconds"].append(time.time() - t0)
        rec["val_mae"].append(evaluate(model, data["val"], device)[0])
        rec["train_mae"].append(evaluate(model, data["train"], device)[0])

    train_seq_model(model, (data["train"]["X"], data["train"]["Y"]), (data["val"]["X"], data["val"]["Y"]),
                    warmup=WARMUP, epochs=cfg["epochs"], batch_size=params["batch"], lr=params["lr"],
                    weight_decay=params["weight_decay"], device=device, patience=cfg["patience"],
                    epoch_callback=callback)
    rec["final_val_mae"] = evaluate(model, data["val"], device)[0]   # best epoch, restored
    rec["test_mae"] = evaluate(model, data["test"], device)[0]
    rec["best_epoch"] = int(np.argmin(rec["val_mae"]))
    return rec


def stacked(runs, field):
    """Seeds early-stop at different epochs: pad with NaN so a mean covers only seeds still training."""
    n = max(len(r[field]) for r in runs)
    return np.array([r[field] + [np.nan] * (n - len(r[field])) for r in runs])


def draw_curves(ax, result, keys, lw_scale=1.0):
    for key in keys:
        val = stacked(result["runs"][key], "val_mae")
        epochs = np.arange(1, val.shape[1] + 1)
        ax.fill_between(epochs, np.nanmin(val, 0), np.nanmax(val, 0), color=STYLE[key]["color"],
                        alpha=0.18, lw=0)
        ax.plot(epochs, np.nanmean(val, 0), lw=lw_scale * (3.2 if key == "minlstm_seq_hybrid" else 2.4),
                label=NAMES[key], **STYLE[key])


def plot(result, out):
    imm_val = result["meta"]["imm_val_mae"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    draw_curves(ax, result, KEYS)
    ax.axhline(imm_val, color=INK, lw=2, ls=":", zorder=1, label=f"Tuned IMM alone ({imm_val:.2f} m)")
    ax.set_ylim(0.24, 0.78)
    ax.set_xlim(0, None)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Validation position error (m)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", ncol=3, handlelength=2.4, columnspacing=1.2)

    # the two IMM-anchored models differ by about a centimetre: zoom in on them
    hybrids = ["lstm_seq_hybrid", "minlstm_seq_hybrid"]
    ins = ax.inset_axes([0.42, 0.30, 0.56, 0.40])
    draw_curves(ins, result, hybrids, lw_scale=0.8)
    ins.axhline(imm_val, color=INK, lw=1.6, ls=":", zorder=1)
    ins.set_xlim(0, 120)
    ins.set_ylim(0.258, 0.288)
    ins.set_yticks([0.26, 0.27, 0.28])
    ins.tick_params(labelsize=11.5)
    ins.grid(axis="x", visible=False)
    ins.set_title("Zoom: IMM-anchored models", fontsize=12.5, loc="left", pad=4)
    for side in ("top", "right"):
        ins.spines[side].set_visible(True)
    ins.patch.set_facecolor("white")
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "fig_training_curves")


def plot_search(search, imm_val, out):
    """Best validation error so far against Optuna trial, from results/search_curves.json."""
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ends = {}
    for key in KEYS:
        trials = search["searches"][key]["trials"]
        n = np.arange(1, len(trials) + 1)
        vals = [t["value"] if t["state"] == "COMPLETE" else np.nan for t in trials]
        best = np.fmin.accumulate(np.nan_to_num(vals, nan=np.inf))
        ax.scatter(n, vals, s=36, color=STYLE[key]["color"], alpha=0.45, lw=0, zorder=2)
        ax.step(n, best, where="post", lw=3.2 if key == "minlstm_seq_hybrid" else 2.4,
                label=NAMES[key], zorder=3, **STYLE[key])
        ends[key] = best[-1]
    ax.axhline(imm_val, color=INK, lw=2, ls=":", zorder=1, label=f"Tuned IMM alone ({imm_val:.2f} m)")
    # end-of-search labels, nudged apart where the best values nearly coincide
    y_prev = -np.inf
    for key in sorted(ends, key=ends.get):
        y = max(ends[key], y_prev + 0.014)
        ax.text(len(trials) + 0.4, y, f"{ends[key]:.3f}", va="center", fontsize=12.5,
                color=STYLE[key]["color"] if key != "lstm_seq_hybrid" else DARK,
                fontweight="bold" if key == "minlstm_seq_hybrid" else "normal")
        y_prev = y
    ax.set_xlim(0.5, len(trials) + 2.6)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_ylim(0.25, 0.55)
    ax.set_xlabel("Optuna trial (dots: completed trials; pruned trials are not shown)")
    ax.set_ylabel("Best validation position error (m)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, handlelength=2.6)
    out.mkdir(parents=True, exist_ok=True)
    save(fig, out, "fig_search_history")


def summarise(result):
    study = json.loads((ROOT / "results" / "streaming_study.json").read_text())["learned"]
    rows = {}
    for key in KEYS:
        runs = result["runs"][key]
        test = [r["test_mae"] for r in runs]
        rows[key] = dict(test_mean=float(np.mean(test)), test_std=float(np.std(test)),
                         study_test_mean=study[key]["test_mean"],
                         best_epoch=[r["best_epoch"] for r in runs],
                         epochs_run=[len(r["val_mae"]) for r in runs],
                         seconds_per_epoch=float(np.mean([r["seconds"][-1] / len(r["seconds"]) for r in runs])))
        log(f"{key:<20} test {rows[key]['test_mean']:.3f} +- {rows[key]['test_std']:.3f} m "
            f"(study {rows[key]['study_test_mean']:.3f})  best epochs {rows[key]['best_epoch']}")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    out = pathlib.Path(args.out)
    json_path = out / ("training_curves_smoke.json" if args.smoke else "training_curves.json")
    if args.plot_only:
        result = json.loads(json_path.read_text())
        plot(result, out / "poster")
        plot_search(json.loads((out / "search_curves.json").read_text()), result["meta"]["imm_val_mae"],
                    out / "poster")
        return

    study = json.loads((ROOT / "results" / "streaming_study.json").read_text())
    scale = "smoke" if args.smoke else "full"
    cfg = SCALES[scale]
    seed = study["meta"]["seed"]
    device = resolve_device(args.device)
    imm_params = dict(dt=CORPUS["dt"], sigma=CORPUS["sigma"], **study["classical"]["imm"]["params"])
    datasets = build_datasets(cfg, seed, imm_params)
    result = dict(meta=dict(scale=scale, cfg=cfg, seed=seed, device=str(device),
                            imm_val_mae=study["classical"]["imm"]["val_mae"],
                            started=time.strftime("%Y-%m-%d %H:%M:%S")), runs={})
    for key in KEYS:
        entry = study["learned"][key]
        result["runs"][key] = []
        for s in range(cfg["seeds"]):
            rec = run_seed(entry["arch"], entry["params"], datasets[entry["mode"]], cfg, device, seed + s)
            result["runs"][key].append(rec)
            log(f"{key:<20} seed {s}  test {rec['test_mae']:.3f} m  best epoch {rec['best_epoch']}"
                f"/{len(rec['val_mae'])}  {rec['seconds'][-1]:.0f}s")
            save_json(json_path, result)
    result["summary"] = summarise(result)
    result["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(json_path, result)
    plot(result, out / "poster")
    if (out / "search_curves.json").exists():
        plot_search(json.loads((out / "search_curves.json").read_text()), result["meta"]["imm_val_mae"],
                    out / "poster")


if __name__ == "__main__":
    main()
