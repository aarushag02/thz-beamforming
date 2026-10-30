"""Hyperparameter-search and training-curve figures for the learned models.

    python experiments/search_curves.py --smoke       # < 2 min on CPU, wiring check
    python experiments/search_curves.py               # full, ~1 h on a T4
    python experiments/search_curves.py --plot-only   # redraw from results/search_curves.json

run_study.py keeps only each search's winner. This re-runs the same Optuna search (same
corpus, sampler seed, pruner and search space) and keeps every trial with its per-epoch
validation loss, then retrains each configuration reported in streaming_study.json once
(seed 0) and keeps its train/validation loss per epoch. Hybrid mode uses the IMM
parameters stored in streaming_study.json instead of re-tuning the IMM. GPU
non-determinism means re-run trials can differ slightly from the original search.
"""

import argparse
import json
import pathlib
import sys
import time
from itertools import product

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, NullFormatter

from experiments.run_study import (ARCHS, COLOURS, CORPUS, MODES, SCALES, SEARCH_SEED_OFFSET,
                                   evaluate, fit, log, make_datasets, make_model, save_json,
                                   suggest)
from src.data import WARMUP, split_by_trajectory
from src.train import resolve_device, set_seed, train_seq_model
from src.trajectories import generate_corpus

NAMES = {"lstm_seq": "LSTM", "minlstm_seq": "minLSTM", "minlstm_seq_res": "minLSTM-res"}


def label(key):
    arch, mode = key.rsplit("_", 1)
    return f"IMM + {NAMES[arch]}" if mode == "hybrid" else NAMES[arch]


def run_search(arch, data, cfg, device, seed):
    """run_study.search, but returning every trial instead of only the winner."""
    def objective(trial):
        params = suggest(trial)

        def callback(epoch, val_loss):
            trial.report(val_loss if np.isfinite(val_loss) else 1e9, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = fit(arch, params, data, seed + SEARCH_SEED_OFFSET, cfg, device, callback)
        mae, _ = evaluate(model, data["val"], device)
        return mae if np.isfinite(mae) else float("inf")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=10))
    study.optimize(objective, n_trials=cfg["trials"], show_progress_bar=False)
    trials = [dict(number=t.number, state=t.state.name, value=t.value, params=t.params,
                   val_loss=[t.intermediate_values[e] for e in sorted(t.intermediate_values)])
              for t in study.trials]
    return dict(trials=trials, best_params=dict(study.best_params), best_value=float(study.best_value))


def run_curve(arch, params, data, cfg, device, seed):
    """Train one configuration exactly as run_study.fit does and return its loss history."""
    set_seed(seed)
    model = make_model(arch, params, data["train"]["X"].shape[-1])
    history = train_seq_model(model, (data["train"]["X"], data["train"]["Y"]),
                              (data["val"]["X"], data["val"]["Y"]), warmup=WARMUP,
                              epochs=cfg["epochs"], batch_size=params["batch"], lr=params["lr"],
                              weight_decay=params["weight_decay"], device=device,
                              patience=cfg["patience"])
    return dict(params=params, train_loss=history["train_loss"], val_loss=history["val_loss"],
                best_epoch=int(np.argmin(history["val_loss"])))


# --- plots -------------------------------------------------------------------

def grid_axes(n_rows=len(MODES), n_cols=len(ARCHS), **kw):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.3 * n_rows), squeeze=False, **kw)
    return fig, {(mode, arch): axes[i, j] for i, mode in enumerate(MODES) for j, arch in enumerate(ARCHS)}


def plot_history(searches, path):
    """Validation MAE of each completed trial, best-so-far line, pruned trials along the top."""
    fig, axes = grid_axes(sharey="row")
    for (mode, arch), ax in axes.items():
        key = f"{arch}_{mode}"
        if key not in searches:
            ax.set_visible(False)
            continue
        trials = searches[key]["trials"]
        done = [t for t in trials if t["state"] == "COMPLETE" and np.isfinite(t["value"])]
        pruned = [t["number"] for t in trials if t["state"] == "PRUNED"]
        xs, ys = [t["number"] for t in done], [t["value"] for t in done]
        ax.scatter(xs, ys, s=22, color=COLOURS[arch], zorder=3, label="completed trial")
        if done:
            best = np.minimum.accumulate(ys)
            ax.step(xs, best, where="post", color="black", lw=1.2, label="best so far")
        if pruned:
            ax.scatter(pruned, [1.0] * len(pruned), marker="x", s=20, color="grey",
                       transform=ax.get_xaxis_transform(), clip_on=False, label="pruned trial")
        ax.set_title(f"{label(key)}  ({len(done)} completed, {len(pruned)} pruned)", fontsize=10)
        ax.set_xlabel("trial")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=0.3)
    for mode in MODES:
        axes[(mode, ARCHS[0])].set_ylabel("validation MAE (m)")
    handles, labels = next(ax for ax in axes.values() if ax.get_visible()).get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def imm_line(ax, imm_loss):
    """Hybrid targets are the IMM's residual, so a zero correction scores exactly this."""
    ax.axhline(imm_loss, color="black", ls=":", lw=1.4, label="IMM alone (zero correction)")


def plot_trial_curves(searches, path, imm_loss=None):
    """Per-epoch validation loss of every trial; pruned trials grey, the winner bold."""
    fig, axes = grid_axes()
    for (mode, arch), ax in axes.items():
        key = f"{arch}_{mode}"
        if key not in searches:
            ax.set_visible(False)
            continue
        trials = searches[key]["trials"]
        done = [t for t in trials if t["state"] == "COMPLETE" and t["value"] is not None]
        best = min(done, key=lambda t: t["value"])["number"] if done else None
        for t in trials:
            loss = np.asarray(t["val_loss"], dtype=float)
            loss[loss >= 1e9] = np.nan
            if t["number"] == best:
                continue
            ax.plot(loss, lw=0.8, color=COLOURS[arch] if t["state"] == "COMPLETE" else "#b0b0b0",
                    alpha=0.8 if t["state"] == "COMPLETE" else 0.6)
        if best is not None:
            ax.plot(trials[best]["val_loss"], lw=2.2, color="black", label=f"best (trial {best})")
        if mode == "hybrid" and imm_loss is not None:
            imm_line(ax, imm_loss)
        if best is not None:
            ax.legend(fontsize=8, loc="upper right")
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.3g"))
        ax.yaxis.set_minor_formatter(FormatStrFormatter("%.3g"))
        ax.set_title(label(key), fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3, which="both")
    for mode in MODES:
        axes[(mode, ARCHS[0])].set_ylabel("validation loss (MSE, normalised)")
    fig.suptitle("Every search trial; grey = pruned by the median rule", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr(searches, path):
    """Validation MAE against learning rate for completed trials, marker size = hidden width."""
    fig, axes = grid_axes(sharey="row")
    sizes = {32: 18, 64: 40, 128: 80}
    for (mode, arch), ax in axes.items():
        key = f"{arch}_{mode}"
        if key not in searches:
            ax.set_visible(False)
            continue
        done = [t for t in searches[key]["trials"] if t["state"] == "COMPLETE" and np.isfinite(t["value"])]
        for layers, marker in zip([1, 2, 3], ["o", "s", "^"]):
            pts = [t for t in done if t["params"]["layers"] == layers]
            ax.scatter([t["params"]["lr"] for t in pts], [t["value"] for t in pts],
                       s=[sizes[t["params"]["hidden"]] for t in pts], marker=marker,
                       color=COLOURS[arch], alpha=0.75, edgecolor="black", linewidth=0.4,
                       label=f"{layers} layer{'s' * (layers > 1)}")
        ax.set_xscale("log")
        ax.set_xlim(2.5e-4, 1.2e-2)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(label(key), fontsize=10)
        ax.set_xlabel("learning rate")
        ax.grid(alpha=0.3, which="both")
    for mode in MODES:
        axes[(mode, ARCHS[0])].set_ylabel("validation MAE (m)")
    handles = [plt.Line2D([], [], ls="", marker=m, color="grey", label=f"{n} layer{'s' * (n > 1)}")
               for n, m in zip([1, 2, 3], ["o", "s", "^"])]
    handles += [plt.scatter([], [], s=s, color="white", edgecolor="black", label=f"hidden {h}")
                for h, s in sizes.items()]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training(curves, path, imm_loss=None):
    """Train (dashed) and validation (solid) loss per epoch of each reported configuration.

    The panels do not share a y-axis: direct targets are displacements and hybrid targets
    are IMM residuals, each normalised by its own scale, so their losses are not comparable.
    """
    fig, axes = plt.subplots(1, len(MODES), figsize=(11, 4))
    for ax, mode in zip(axes, MODES):
        for arch in ARCHS:
            key = f"{arch}_{mode}"
            if key not in curves:
                continue
            c = curves[key]
            ax.plot(c["train_loss"], ls="--", lw=1.2, color=COLOURS[arch])
            ax.plot(c["val_loss"], lw=1.8, color=COLOURS[arch], label=label(key))
            ax.scatter([c["best_epoch"]], [c["val_loss"][c["best_epoch"]]], color=COLOURS[arch],
                       edgecolor="black", zorder=3, s=30)
        if mode == "direct":
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(FormatStrFormatter("%.2g"))
        elif imm_loss is not None:
            imm_line(ax, imm_loss)
            # zoom on the band around the IMM; the first-epoch train loss of a randomly
            # initialised correction can sit far above it and would flatten the panel
            vals = [v for a in ARCHS if f"{a}_{mode}" in curves
                    for v in curves[f"{a}_{mode}"]["train_loss"][1:] + curves[f"{a}_{mode}"]["val_loss"]]
            top = 1.15 * imm_loss
            ax.set_ylim(0.95 * min(vals), top)
            if any(v > top for a in ARCHS if f"{a}_{mode}" in curves
                   for v in curves[f"{a}_{mode}"]["train_loss"] + curves[f"{a}_{mode}"]["val_loss"]):
                ax.text(0.01, 0.98, "epoch-0 train loss off scale", transform=ax.transAxes,
                        ha="left", va="top", fontsize=8, color="grey")
        ax.set_xlabel("epoch")
        ax.set_title("direct" if mode == "direct" else "hybrid (correction to IMM)", fontsize=11)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
        ax.set_ylabel("loss (MSE, normalised)")
    fig.suptitle("Training curves of the selected configurations "
                 "(solid = validation, dashed = train, dot = best epoch)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all(result, out):
    imm_loss = result["meta"].get("imm_alone_val_loss")
    plot_history(result["searches"], out / "search_history.png")
    plot_trial_curves(result["searches"], out / "search_trial_curves.png", imm_loss)
    plot_lr(result["searches"], out / "search_lr.png")
    plot_training(result["curves"], out / "training_curves_selected.png", imm_loss)
    log(f"figures -> {out}/search_*.png, training_curves_selected.png")


# --- main --------------------------------------------------------------------

def build_datasets(cfg, seed, imm_params):
    rng = np.random.default_rng(seed)
    corpus = generate_corpus(n=cfg["n_traj"], steps=CORPUS["steps"], rng=rng,
                             sigma=CORPUS["sigma"], dt=CORPUS["dt"])
    train_c, val_c, test_c = split_by_trajectory(corpus, (0.7, 0.15, 0.15), rng=rng)
    return {mode: make_datasets(train_c, val_c, test_c, mode, imm_params) for mode in MODES}


def imm_alone_loss(data):
    """Validation loss of a hybrid model that outputs zero, i.e. of the IMM by itself."""
    return float(np.mean(data["val"]["Y"][:, WARMUP:] ** 2))


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / ("search_curves_smoke.json" if args.smoke else "search_curves.json")
    if args.plot_only:
        result = json.loads(json_path.read_text())
        if "imm_alone_val_loss" not in result["meta"]:   # files written before it was recorded
            meta = result["meta"]
            data = build_datasets(meta["cfg"], meta["seed"], meta["imm_params"])["hybrid"]
            meta["imm_alone_val_loss"] = imm_alone_loss(data)
            save_json(json_path, result)
        plot_all(result, out)
        return

    scale = "smoke" if args.smoke else "full"
    cfg = SCALES[scale]
    device = resolve_device(args.device)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = json.loads((ROOT / "results" / "streaming_study.json").read_text())
    imm_params = dict(dt=CORPUS["dt"], sigma=CORPUS["sigma"], **study["classical"]["imm"]["params"])
    datasets = build_datasets(cfg, args.seed, imm_params)
    log(f"scale={scale} device={device} corpus {len(datasets['direct']['train']['X'])}/"
        f"{len(datasets['direct']['val']['X'])}/{len(datasets['direct']['test']['X'])}")

    result = dict(meta=dict(scale=scale, cfg=cfg, seed=args.seed, device=str(device),
                            imm_params=imm_params, imm_alone_val_loss=imm_alone_loss(datasets["hybrid"]),
                            started=time.strftime("%Y-%m-%d %H:%M:%S")),
                  searches={}, curves={})
    for mode, arch in product(MODES, ARCHS):
        key = f"{arch}_{mode}"
        t0 = time.time()
        result["searches"][key] = run_search(arch, datasets[mode], cfg, device, args.seed)
        params = study["learned"].get(key, {}).get("params", result["searches"][key]["best_params"])
        result["curves"][key] = run_curve(arch, params, datasets[mode], cfg, device, args.seed)
        save_json(json_path, result)
        s = result["searches"][key]
        n_pruned = sum(t["state"] == "PRUNED" for t in s["trials"])
        log(f"{key:<24} best val {s['best_value']:.3f} m  ({n_pruned}/{len(s['trials'])} pruned)  "
            f"{time.time() - t0:.0f}s")
    result["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(json_path, result)
    plot_all(result, out)


if __name__ == "__main__":
    main()
