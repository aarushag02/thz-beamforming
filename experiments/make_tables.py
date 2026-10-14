"""JSON results -> LaTeX tables, so no number in the paper is typed by hand.

    python experiments/make_tables.py
"""

import argparse
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

LABELS = {
    "hold": "Reactive (hold)", "constant_velocity": "Constant velocity",
    "ls_linear": "LS line fit", "kalman": "Kalman (CV)", "imm": "IMM (CV+CT)",
    "lstm_seq_direct": "LSTM", "minlstm_seq_direct": "minLSTM",
    "minlstm_seq_res_direct": "minLSTM-res",
    "plain_lstm_seq_direct": "LSTM (plain PyTorch)",
    "lstm_seq_hybrid": "IMM + LSTM", "minlstm_seq_hybrid": "IMM + minLSTM",
    "minlstm_seq_res_hybrid": "IMM + minLSTM-res",
}


def label(key):
    return LABELS.get(key, key.replace("_", r"\_"))


def table(caption, ref, columns, rows):
    head = " & ".join(columns) + r" \\"
    if not rows:   # an empty tabular body is a LaTeX error ("Misplaced \noalign")
        rows = [(f"\\multicolumn{{{len(columns)}}}{{c}}{{---}}",)]
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\\label{{{ref}}}\n"
            f"\\begin{{tabular}}{{l{'r' * (len(columns) - 1)}}}\n\\toprule\n{head}\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n")


def accuracy(study):
    rows = [(label(k), f"{v['test_mae']:.3f}", "0") for k, v in study["classical"].items()]
    rows += [(label(k), f"{v['test_mean']:.3f} $\\pm$ {v['test_std']:.3f}", f"{v['n_params']:,}")
             for k, v in study["learned"].items()]
    return table("One-step prediction error on held-out trajectories (mean $\\pm$ std over seeds).",
                 "tab:accuracy", ["Method", "MAE (m)", "Params"], rows)


def compute(bench):
    rows = []
    for key, lat in bench.get("latency_us", {}).items():
        arch, dev = key.split("@")
        rows.append((f"{label(arch + '_direct')} ({dev})", f"{lat['step']:.1f}",
                     f"{lat['window_10']:.1f}", f"{lat['window_40']:.1f}",
                     f"{bench['macs'].get(arch, 0):,}"))
    return table("Per-slot inference cost: streaming step vs.\\ re-reading a $W$-slot window ($\\mu$s).",
                 "tab:compute", ["Model", "Step", "$W{=}10$", "$W{=}40$", "MACs/step"], rows)


def generalisation(rows_in):
    if not rows_in:
        return table("MAE (m) on a mobility regime never seen in training.", "tab:generalisation",
                     ["Method"], [])
    keys = [k for k in rows_in[0] if k not in ("regime", "n_unseen")]
    rows = [(label(k), *[f"{r[k]:.3f}" for r in rows_in]) for k in keys]
    return table("MAE (m) on a mobility regime never seen in training.", "tab:generalisation",
                 ["Method", *[r["regime"] for r in rows_in]], rows)


def beam(beam_json):
    rows = [(label(k), *[f"{v:.2f}" for v in m["misalign_bw"]])
            for k, m in beam_json["methods"].items()]
    return table("Mean beam misalignment (half-power beamwidths) vs.\\ array size.", "tab:beam",
                 ["Method", *[f"$N_t{{=}}{n}$" for n in beam_json["n_tx"]]], rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument("--paper", default=str(ROOT / "paper"))
    args = p.parse_args(argv)
    results, tables = pathlib.Path(args.results), pathlib.Path(args.paper) / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    study = json.loads((results / "streaming_study.json").read_text())
    (tables / "accuracy.tex").write_text(accuracy(study))
    (tables / "compute.tex").write_text(compute(study.get("bench", {})))
    (tables / "generalisation.tex").write_text(generalisation(study.get("generalisation", [])))
    beam_path = results / "beam_eval.json"
    if beam_path.exists():
        (tables / "beam.tex").write_text(beam(json.loads(beam_path.read_text())))
    print(f"tables -> {tables}")


if __name__ == "__main__":
    main()
