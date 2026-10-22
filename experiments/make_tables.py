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


def snr_table(snr_json, n_tx="64"):
    res = snr_json["results"].get(n_tx) or snr_json["results"][str(snr_json["n_tx"][-1])]
    n_tx = next(k for k, v in snr_json["results"].items() if v is res)
    names = {"oracle": "Oracle (true CSI at $t{+}1$)", "reactive": "Reactive (true CSI at $t$)"}
    rows = [(names.get(k) or label(k), f"{m['snr_db']:.2f}",
             f"{m['gain_db']:+.2f} $\\pm$ {m['gain_db_ci95']:.2f}", f"{m['se']:.2f}",
             f"{m['se_gain']:+.2f}", f"{100 * m['oracle_gap_recovered']:.0f}",
             f"{100 * m['outage']:.1f}") for k, m in res.items()]
    return table(f"MRT beamforming through the O1\\_60 twin, $N_t{{=}}{n_tx}$: SNR, gain over the "
                 "reactive beam (95\\% CI over trajectories), spectral efficiency, share of the "
                 "reactive-to-oracle gap recovered, and outage (SNR $>$3 dB below oracle).",
                 "tab:snr", ["Method", "SNR (dB)", "Gain (dB)", "SE", "$\\Delta$SE",
                             "Gap (\\%)", "Outage (\\%)"], rows)


TRAINING_METHODS = ("reactive", "imm", "minlstm_seq_hybrid", "lstm_seq_hybrid")
REF_NAMES = {"oracle": "Oracle", "reactive": "Reactive (CSI at $t$)"}


def hybrid_table(cb_json, n_beams):
    res = cb_json["codebooks"][n_beams]["hybrid"]
    rfs = list(res)
    rows = [(REF_NAMES.get(k) or label(k),
             *[f"{res[r][k]['gain_db']:+.2f}" for r in rfs],
             f"{cb_json['mrt'][k]['gain_db']:+.2f}", f"{100 * m['beam_match']:.0f}")
            for k, m in res[rfs[0]].items()]
    return table(f"SNR gain (dB) over the reactive beam with hybrid precoding from a {n_beams}-beam "
                 f"DFT codebook at $N_t{{=}}{cb_json['n_tx']}$, against fully-digital MRT; beam match is "
                 "how often the analog beam ($N_{\\mathrm{RF}}{=}1$) equals the oracle's.",
                 f"tab:hybrid{n_beams}",
                 ["Method", *[f"$N_{{\\mathrm{{RF}}}}{{=}}{r}$" for r in rfs], "MRT", "Match (\\%)"], rows)


def training_table(cb_json, n_beams):
    tr = cb_json["codebooks"][n_beams]["training"]
    taus = list(tr["exhaustive"]["se_eff"])

    def cells(r):
        return (str(r["beams_measured"]), f"{r['snr_db']:.2f}", f"{r['se']:.2f}",
                *[f"{r['se_eff'][t]:.2f}" for t in taus])
    rows = [("Exhaustive sweep", *cells(tr["exhaustive"]))]
    rows += [(f"{REF_NAMES.get(k) or label(k)}, top-{n}", *cells(r))
             for k in TRAINING_METHODS if k in tr for n, r in tr[k].items()]
    return table(f"Beam training with a {n_beams}-beam DFT codebook: measure each method's top-$k$ "
                 "beams on the live channel, or sweep all of them. $\\tau$ is the time to measure one "
                 "beam as a fraction of the update interval; effective SE is $(1-k\\tau)\\log_2(1+\\mathrm{SNR})$.",
                 f"tab:training{n_beams}",
                 ["Method", "Beams", "SNR (dB)", "SE", *[f"SE$_{{\\mathrm{{eff}}}}$ $\\tau{{=}}{t}$" for t in taus]],
                 rows)


def phase_noise_table(pn_json, method="minlstm_seq_hybrid"):
    dfts = sorted((k for k in pn_json["precoders"] if k.startswith("dft")), key=lambda k: int(k[3:]))
    shown = ["mrt"] + dfts[-1:]
    names = {"mrt": "MRT", **{k: f"DFT-{k[3:]}" for k in dfts}}
    rows = []
    for sigma in pn_json["precoders"]["mrt"]:
        cells = [f"{sigma}", f"{pn_json['precoders']['mrt'][sigma]['expected_loss_db']:.2f}"]
        for p in shown:
            m = pn_json["precoders"][p][sigma][method]
            cells += [f"{m['snr_db']:.2f}", f"{m['gain_db']:+.2f}", f"{100 * m['oracle_gap_recovered']:.0f}"]
        rows.append(tuple(cells))
    cols = ["$\\sigma_\\phi$ (rad)", "Exp.\\ loss (dB)"]
    for p in shown:
        cols += [f"{names[p]} SNR", f"{names[p]} gain", f"{names[p]} gap (\\%)"]
    return table(f"{label(method)} under i.i.d.\\ per-element phase error $\\phi\\sim\\mathcal{{N}}(0,\\sigma_\\phi^2)$ "
                 f"on the transmit beam of every method (same draw for all), $N_t{{=}}{pn_json['n_tx']}$, "
                 f"mean over {pn_json['draws']} draws. Gain is over the equally impaired reactive beam; "
                 "expected loss is the large-array limit $10\\log_{10}(e)\\,\\sigma_\\phi^2$.",
                 "tab:phasenoise", cols, rows)


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
    snr_path = results / "snr_eval.json"
    if snr_path.exists():
        (tables / "snr.tex").write_text(snr_table(json.loads(snr_path.read_text())))
    cb_path = results / "codebook_eval.json"
    if cb_path.exists():
        cb = json.loads(cb_path.read_text())
        for n_beams in cb["codebooks"]:
            (tables / f"hybrid_{n_beams}.tex").write_text(hybrid_table(cb, n_beams))
            (tables / f"training_{n_beams}.tex").write_text(training_table(cb, n_beams))
    pn_path = results / "phase_noise_eval.json"
    if pn_path.exists():
        (tables / "phase_noise.tex").write_text(phase_noise_table(json.loads(pn_path.read_text())))
    print(f"tables -> {tables}")


if __name__ == "__main__":
    main()
