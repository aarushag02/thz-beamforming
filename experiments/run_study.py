"""Streaming study: classical trackers vs LSTM vs minLSTM, plus compute.

    python experiments/run_study.py --smoke            # < 2 min on CPU, wiring check
    python experiments/run_study.py --quick            # ~15 min on a GPU, sanity run
    python experiments/run_study.py                    # full study

Results are written after every phase, so a crash keeps what finished.
"""

import argparse
import json
import pathlib
import platform
import sys
import time
from collections import defaultdict
from itertools import product

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch

from src.baselines import (constant_velocity, hold, imm_stream, kalman_cv_stream,
                           ls_linear, stream_window_predictor)
from src.bench import macs_per_step, time_step_latency, time_train_step, time_window_latency
from src.data import (WARMUP, build_sequences, masked_mae_abs, masked_mae_metres,
                      sequence_truth, split_by_regime, split_by_trajectory, stream_to_sequence)
from src.models import build_model
from src.train import predict_seq, resolve_device, set_seed, train_seq_model
from src.trajectories import REGIMES, generate_corpus

SCALES = {
    "smoke": dict(n_traj=24, trials=2, seeds=1, epochs=3, patience=2,
                  seq_lens=[16, 64], bench_iters=2, latency_iters=20),
    "quick": dict(n_traj=90, trials=4, seeds=2, epochs=40, patience=8,
                  seq_lens=[16, 32, 64, 128, 256, 512], bench_iters=10, latency_iters=200),
    "full": dict(n_traj=600, trials=30, seeds=5, epochs=200, patience=20,
                 seq_lens=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
                 bench_iters=20, latency_iters=1000),
}
CORPUS = dict(steps=200, sigma=0.5, dt=0.1)
ARCHS = ["lstm_seq", "minlstm_seq", "minlstm_seq_res"]
MODES = ["direct", "hybrid"]
KF_Q = [0.1, 1.0, 10.0, 100.0, 1000.0]
IMM_GRID = dict(q=[0.3, 3.0, 30.0], omega=[0.1, 0.25, 0.5], p_stay=[0.9, 0.97])
LS_WINDOWS = [3, 5, 10, 20]
BENCH_ARCHS = ["lstm_seq", "plain_lstm_seq", "minlstm_seq"]
COLOURS = {"lstm_seq": "#2a78d6", "plain_lstm_seq": "#eb6834", "minlstm_seq": "#1baf7a",
           "minlstm_seq_compiled": "#0f6e4c", "minlstm_seq_res": "#7a5cc4"}

_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def save_json(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float))
    tmp.replace(path)


# --- classical ---------------------------------------------------------------

def stream_preds(trajs, fn):
    return np.stack([stream_to_sequence(fn(t["observed"])) for t in trajs])


def truths(trajs):
    return np.stack([sequence_truth(t) for t in trajs])


def per_regime(pred_abs, trajs):
    groups = defaultdict(list)
    for p, t in zip(pred_abs, trajs):
        groups[t["regime"]].append(masked_mae_abs(p[None], sequence_truth(t)[None]))
    return {r: float(np.mean(v)) for r, v in groups.items()}


def classical_predictors(dt, sigma):
    """name -> (list of candidate (params, fn)) to tune on validation."""
    return {
        "hold": [({}, lambda o: stream_window_predictor(o, hold, 1))],
        "constant_velocity": [({}, lambda o: stream_window_predictor(o, constant_velocity, 2))],
        "ls_linear": [({"window": w}, lambda o, w=w: stream_window_predictor(o, ls_linear, w))
                      for w in LS_WINDOWS],
        "kalman": [({"q": q}, lambda o, q=q: kalman_cv_stream(o, dt=dt, sigma=sigma, q=q))
                   for q in KF_Q],
        "imm": [({"q": q, "omega": w, "p_stay": p},
                 lambda o, q=q, w=w, p=p: imm_stream(o, dt=dt, sigma=sigma, q=q, omega=w, p_stay=p))
                for q, w, p in product(IMM_GRID["q"], IMM_GRID["omega"], IMM_GRID["p_stay"])],
    }


def phase_classical(val_c, test_c):
    results, preds, fns = {}, {}, {}
    val_truth, test_truth = truths(val_c), truths(test_c)
    for name, candidates in classical_predictors(CORPUS["dt"], CORPUS["sigma"]).items():
        scored = [(masked_mae_abs(stream_preds(val_c, fn), val_truth), params, fn)
                  for params, fn in candidates]
        val_mae, params, fn = min(scored, key=lambda s: s[0])
        test_pred = stream_preds(test_c, fn)
        results[name] = dict(params=params, val_mae=val_mae,
                             test_mae=masked_mae_abs(test_pred, test_truth),
                             per_regime=per_regime(test_pred, test_c))
        preds[name], fns[name] = test_pred, fn
        log(f"classical {name:<18} val {val_mae:.3f} m  test {results[name]['test_mae']:.3f} m  {params}")
    return results, preds, fns


# --- learned -----------------------------------------------------------------

def suggest(trial):
    return dict(
        hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
        layers=trial.suggest_categorical("layers", [1, 2, 3]),
        lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
        dropout=trial.suggest_float("dropout", 0.0, 0.2),
        batch=trial.suggest_categorical("batch", [16, 32, 64]),
        weight_decay=trial.suggest_categorical("weight_decay", [0.0, 1e-5, 1e-4]),
    )


def fit(arch, params, data, seed, cfg, device, callback=None):
    set_seed(seed)
    model = build_model(arch, input_size=data["train"]["X"].shape[-1],
                        hidden_size=params["hidden"], num_layers=params["layers"],
                        dropout=params["dropout"])
    train_seq_model(model, (data["train"]["X"], data["train"]["Y"]),
                    (data["val"]["X"], data["val"]["Y"]), warmup=WARMUP,
                    epochs=cfg["epochs"], batch_size=params["batch"], lr=params["lr"],
                    weight_decay=params["weight_decay"], device=device,
                    patience=cfg["patience"], epoch_callback=callback)
    return model


def evaluate(model, split, device):
    pred = predict_seq(model, split["X"], device=device)
    return masked_mae_metres(pred, split), split["anchors"] + pred * split["scales"]["y"]


def make_datasets(train_c, val_c, test_c, mode, imm_params):
    tr = build_sequences(train_c, mode, imm_params)
    return {"train": tr,
            "val": build_sequences(val_c, mode, imm_params, scales=tr["scales"]),
            "test": build_sequences(test_c, mode, imm_params, scales=tr["scales"])}


def search(arch, data, cfg, device, seed):
    def objective(trial):
        params = suggest(trial)

        def callback(epoch, val_loss):
            trial.report(val_loss if np.isfinite(val_loss) else 1e9, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        model = fit(arch, params, data, seed, cfg, device, callback)
        mae, _ = evaluate(model, data["val"], device)
        return mae if np.isfinite(mae) else float("inf")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=3))
    study.optimize(objective, n_trials=cfg["trials"], show_progress_bar=False)
    return dict(study.best_params), float(study.best_value)


def phase_learned(datasets, test_c, cfg, device, seed, models_dir):
    results, preds = {}, {}
    for mode, arch in product(MODES, ARCHS):
        key = f"{arch}_{mode}"
        data = datasets[mode]
        params, best_val = search(arch, data, cfg, device, seed)
        val_maes, test_maes, first = [], [], None
        for s in range(cfg["seeds"]):
            model = fit(arch, params, data, seed + s, cfg, device)
            val_maes.append(evaluate(model, data["val"], device)[0])
            test_mae, test_pred = evaluate(model, data["test"], device)
            test_maes.append(test_mae)
            if first is None:
                first = (model, test_pred)
        model, test_pred = first
        torch.save(model.state_dict(), models_dir / f"{key}.pt")
        results[key] = dict(arch=arch, mode=mode, params=params, search_best_val=best_val,
                            val_mae=val_maes, test_mae=test_maes,
                            test_mean=float(np.mean(test_maes)), test_std=float(np.std(test_maes)),
                            per_regime=per_regime(test_pred, test_c),
                            n_params=sum(p.numel() for p in model.parameters()))
        preds[key] = test_pred
        log(f"learned {key:<24} test {np.mean(test_maes):.3f} ± {np.std(test_maes):.3f} m  {params}")
    return results, preds


def phase_generalisation(corpus, learned, classical_fns, imm_params, cfg, device, seed):
    rows = []
    for regime in REGIMES:
        seen, unseen = split_by_regime(corpus, held_out=regime)
        if len(unseen) < 2 or len(seen) < 10:   # val split needs >= 2 trajectories
            continue
        tr_c, va_c, _ = split_by_trajectory(seen, (0.8, 0.2, 0.0), rng=np.random.default_rng(seed))
        row = dict(regime=regime, n_unseen=len(unseen))
        for name in ("kalman", "imm"):
            row[name] = masked_mae_abs(stream_preds(unseen, classical_fns[name]), truths(unseen))
        for key, entry in learned.items():
            data = make_datasets(tr_c, va_c, unseen, entry["mode"], imm_params)
            model = fit(entry["arch"], entry["params"], data, seed, cfg, device)
            row[key] = evaluate(model, data["test"], device)[0]
        rows.append(row)
        log(f"held-out {regime}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items()
                                               if isinstance(v, float)))
    return rows


# --- compute -----------------------------------------------------------------

def phase_bench(cfg, device):
    bench = {"device": str(device), "train_step": {}, "latency_us": {}, "macs": {}}
    variants = [(a, a, False) for a in BENCH_ARCHS]
    if device.type == "cuda":
        variants.append(("minlstm_seq_compiled", "minlstm_seq", True))
    for label, arch, compiled in variants:
        rows = [time_train_step(arch, T, device=device, compile=compiled,
                                iters=cfg["bench_iters"]) for T in cfg["seq_lens"]]
        bench["train_step"][label] = dict(seq_lens=cfg["seq_lens"],
                                          ms=[r["ms"] for r in rows],
                                          peak_mib=[r["peak_mib"] for r in rows],
                                          errors=[r["error"] for r in rows])
        log(f"bench {label:<22} " + " ".join(f"{m:.2f}" for m in bench["train_step"][label]["ms"]))
    for dev in {device, torch.device("cpu")}:
        for arch in BENCH_ARCHS + ["minlstm_seq_res"]:
            entry = {"step": time_step_latency(arch, device=dev, iters=cfg["latency_iters"])}
            for W in (10, 40):
                entry[f"window_{W}"] = time_window_latency(arch, W, device=dev,
                                                           iters=cfg["latency_iters"])
            bench["latency_us"][f"{arch}@{dev.type}"] = entry
    for arch in BENCH_ARCHS + ["minlstm_seq_res"]:
        bench["macs"][arch] = macs_per_step(arch)
    return bench


# --- figures -----------------------------------------------------------------

def plot_accuracy(study, path):
    rows = [(f"{k} (classical)", v["test_mae"], 0.0) for k, v in study["classical"].items()]
    rows += [(k, v["test_mean"], v["test_std"]) for k, v in study["learned"].items()]
    rows.sort(key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.2))
    ax.barh([r[0] for r in rows], [r[1] for r in rows], xerr=[r[2] for r in rows],
            color=["#8a8984" if "classical" in r[0] else "#1baf7a" if "minlstm" in r[0]
                   else "#2a78d6" for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("test MAE (m), lower is better")
    ax.set_title("Next-position prediction on held-out trajectories")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_step(bench, path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for label, entry in bench["train_step"].items():
        ax.plot(entry["seq_lens"], entry["ms"], marker="o", label=label,
                color=COLOURS.get(label))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("training step (ms)")
    ax.set_title(f"Training step time ({bench['device']})")
    ax.legend(frameon=False)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_generalisation(rows, path):
    if not rows:
        return
    keys = [k for k in rows[0] if k not in ("regime", "n_unseen")]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    width = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax.bar(np.arange(len(rows)) + i * width, [r[key] for r in rows], width,
               label=key, color=COLOURS.get(key.rsplit("_", 1)[0]))
    ax.set_xticks(np.arange(len(rows)) + 0.4 - width / 2)
    ax.set_xticklabels([f"held-out {r['regime']}" for r in rows])
    ax.set_ylabel("MAE (m)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --- main --------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true")
    scale.add_argument("--quick", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--out", default=str(ROOT / "results"))
    p.add_argument("--models", default=str(ROOT / "models"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-bench", action="store_true")
    p.add_argument("--skip-generalisation", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    scale = "smoke" if args.smoke else "quick" if args.quick else "full"
    cfg = SCALES[scale]
    device = resolve_device(args.device)
    out, models_dir = pathlib.Path(args.out), pathlib.Path(args.models)
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    json_path = out / "streaming_study.json"

    study = dict(meta=dict(scale=scale, cfg=cfg, corpus=CORPUS, seed=args.seed, warmup=WARMUP,
                           device=str(device), torch=torch.__version__,
                           python=platform.python_version(),
                           gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
                           started=time.strftime("%Y-%m-%d %H:%M:%S")),
                 classical={}, learned={}, generalisation=[], bench={})
    log(f"scale={scale} device={device} gpu={study['meta']['gpu']}")

    rng = np.random.default_rng(args.seed)
    corpus = generate_corpus(n=cfg["n_traj"], steps=CORPUS["steps"], rng=rng,
                             sigma=CORPUS["sigma"], dt=CORPUS["dt"])
    train_c, val_c, test_c = split_by_trajectory(corpus, (0.7, 0.15, 0.15), rng=rng)
    log(f"corpus {len(train_c)}/{len(val_c)}/{len(test_c)} train/val/test trajectories")

    study["classical"], classical_preds, classical_fns = phase_classical(val_c, test_c)
    save_json(json_path, study)

    imm_params = dict(dt=CORPUS["dt"], sigma=CORPUS["sigma"], **study["classical"]["imm"]["params"])
    datasets = {mode: make_datasets(train_c, val_c, test_c, mode, imm_params) for mode in MODES}
    study["learned"], learned_preds = phase_learned(datasets, test_c, cfg, device, args.seed, models_dir)
    save_json(json_path, study)

    np.savez_compressed(out / "streaming_predictions.npz", truth=truths(test_c),
                        regimes=np.array([t["regime"] for t in test_c]), warmup=WARMUP,
                        **{f"pred_{k}": v for k, v in {**classical_preds, **learned_preds}.items()})
    plot_accuracy(study, out / "streaming_accuracy.png")

    if not args.skip_generalisation:
        study["generalisation"] = phase_generalisation(corpus, study["learned"], classical_fns,
                                                       imm_params, cfg, device, args.seed)
        save_json(json_path, study)
        plot_generalisation(study["generalisation"], out / "streaming_generalisation.png")

    if not args.skip_bench:
        study["bench"] = phase_bench(cfg, device)
        save_json(json_path, study)
        plot_train_step(study["bench"], out / "streaming_train_step.png")

    study["meta"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save_json(json_path, study)
    log(f"done -> {json_path}")
    return study


if __name__ == "__main__":
    main()
