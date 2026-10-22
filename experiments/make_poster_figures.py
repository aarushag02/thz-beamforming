"""Poster figures: the proposed system against reactive beamforming, from committed JSON results.

    python experiments/make_poster_figures.py        # -> results/poster/fig*.{png,pdf}

Proposed system: IMM + minLSTM position prediction feeding hybrid beamforming with a
128-beam DFT codebook and one RF chain, Nt = 64. Every number is read from
results/streaming_study.json, codebook_eval.json and phase_noise_eval.json; nothing is typed in. The
proposed system is drawn in the accent blue, references in greys, with line style
as a second cue so no comparison depends on colour alone.
"""

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]

ACCENT = "#1c5cab"   # IMM + minLSTM, the proposed system
INK = "#1d1d1b"      # oracle, titles
DARK = "#52514e"
MID = "#9a9993"      # reactive
LIGHT = "#dcdbd5"    # full sweep / MRT
GRID = "#ecebe7"

PROPOSED = "minlstm_seq_hybrid"
CODEBOOK = "128"
LABEL = "Predictive (IMM + minLSTM)"

plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 15, "axes.titlesize": 18, "axes.titleweight": "bold",
    "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "legend.fontsize": 13.5, "legend.frameon": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#8f8e88", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": DARK, "ytick.color": DARK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.dpi": 110,
})


def title(ax, head, sub):
    ax.set_title(head, loc="left", pad=30)
    ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=13, color=DARK, va="bottom")


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}")
    plt.close(fig)


def fig_generalisation(study, out):
    """Position error on a mobility regime never seen in training, by method."""
    hatch_fill = "#cfcec8"
    methods = [("lstm_seq_direct", "LSTM", hatch_fill, "//"), ("minlstm_seq_direct", "minLSTM", hatch_fill, ".."),
               ("lstm_seq_hybrid", "IMM + LSTM", MID, None), (PROPOSED, "IMM + minLSTM", ACCENT, None)]
    detail = {d["regime"]: d for d in study["generalisation_detail"]}
    rows = study["generalisation"]
    regimes = [r["regime"] for r in rows]
    x, width = np.arange(len(regimes)), 0.19

    fig, ax = plt.subplots(figsize=(9, 5.4))
    for i, (key, name, colour, hatch) in enumerate(methods):
        vals = [r[key] for r in rows]
        err = [detail[g]["learned"].get(key, {}).get("std", 0.0) for g in regimes]
        pos = x + (i - (len(methods) - 1) / 2) * width
        ax.bar(pos, vals, width - 0.02, color=colour, hatch=hatch,
               edgecolor=DARK if hatch else "white", linewidth=0, label=name, zorder=2)
        if any(err):
            ax.errorbar(pos, vals, yerr=err, fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        if key == PROPOSED:
            for p, v in zip(pos, vals):
                ax.text(p, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5, color=ACCENT)
        if key == "lstm_seq_direct":
            for p, v, e in zip(pos, vals, err):
                if v > 0.6:
                    ax.text(p, v + e + 0.02, f"{v:.2f} m", ha="center", va="bottom", fontsize=12.5, color=DARK)
    ax.set_xticks(x, [f"{g.capitalize()}\n(held out)" for g in regimes])
    ax.set_ylabel("Position error on unseen regime (m)")
    ax.set_ylim(0, 1.2)
    ax.grid(axis="x", visible=False)
    ax.legend(ncol=2, loc="upper left", bbox_to_anchor=(0, 0.98))
    ax.set_title("Prediction error on unseen motion", loc="left", pad=12)
    save(fig, out, "fig1_generalisation")


def fig_gain(cb, out):
    """SNR gain over reactive, overall and by mobility, with the oracle ceiling."""
    res = cb["codebooks"][CODEBOOK]["hybrid"]["1"]
    groups = ["all", "pedestrian", "cyclist", "vehicular"]
    pick = lambda m, g: res[m]["gain_db"] if g == "all" else res[m]["gain_db_by_regime"][g]
    pred = [pick(PROPOSED, g) for g in groups]
    orc = [pick("oracle", g) for g in groups]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(groups))
    ax.axhline(0, color=MID, lw=2.2, ls="--", zorder=1)
    ax.text(-0.4, -0.06, "Reactive = 0 dB", ha="left",
            va="top", fontsize=12.5, color=DARK)
    ax.bar(x, pred, 0.5, color=ACCENT, label=LABEL, zorder=2)
    ax.scatter(x, orc, marker="_", s=1400, linewidths=3.5, color=INK, zorder=3,
               label="Oracle (perfect future channel)")
    for i, (p, o) in enumerate(zip(pred, orc)):
        if p >= 0.5:   # inside the bar top, clear of the oracle marker above
            ax.text(i, p - 0.08, f"{p:+.2f} dB", ha="center", va="top", fontsize=14,
                    color="white", fontweight="bold")
        else:
            ax.text(i, p - 0.07, f"{p:+.2f} dB", ha="center", va="top", fontsize=14,
                    color=ACCENT, fontweight="bold")
        ax.text(i + 0.3, o, f"{o:.2f}", ha="left", va="center", fontsize=12, color=INK)
    ax.set_xticks(x, ["All users", "Pedestrian", "Cyclist", "Vehicular"])
    ax.set_ylabel("SNR gain over reactive (dB)")
    ax.set_ylim(-0.5, 3.9)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    title(ax, "Predictive beamforming beats reactive, most at speed",
          f"Hybrid beamforming, {CODEBOOK}-beam DFT codebook, 1 RF chain, 64 antennas")
    save(fig, out, "fig2_reactive_vs_predictive")


def fig_mrt_vs_hybrid(cb, out):
    """Digital MRT vs hybrid DFT for the proposed predictor: gain and share of ceiling."""
    mrt = cb["mrt"][PROPOSED]
    hyb = cb["codebooks"][CODEBOOK]["hybrid"]["1"][PROPOSED]
    names = ["Digital MRT\n(64 RF chains)", f"Hybrid DFT-{CODEBOOK}\n(1 RF chain)"]
    colours = [LIGHT, ACCENT]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), gridspec_kw={"wspace": 0.35})
    for ax, vals, fmt, ylab, top in (
            (ax1, [mrt["gain_db"], hyb["gain_db"]], "+{:.2f} dB", "SNR gain over reactive (dB)", 1.2),
            (ax2, [100 * mrt["oracle_gap_recovered"], 100 * hyb["oracle_gap_recovered"]], "{:.0f}%",
             "Achievable gain recovered (%)", 100)):
        bars = ax.bar([0, 1], vals, 0.55, color=colours, edgecolor=[MID, ACCENT], linewidth=1.5)
        for b, v, c in zip(bars, vals, [DARK, ACCENT]):
            ax.text(b.get_x() + b.get_width() / 2, v + top * 0.02, fmt.format(v), ha="center",
                    va="bottom", fontsize=15, color=c, fontweight="bold")
        ax.set_xticks([0, 1], names)
        ax.set_ylim(0, top)
        ax.set_ylabel(ylab)
        ax.grid(axis="x", visible=False)
    ax1.set_title("Hybrid DFT recovers more gain than fully-digital MRT", loc="left", pad=30)
    ax1.text(0, 1.02, f"IMM + minLSTM prediction, 64 antennas", transform=ax1.transAxes,
             fontsize=13, color=DARK, va="bottom")
    save(fig, out, "fig3_mrt_vs_hybrid")


def fig_phase_noise(pn, out):
    """Absolute SNR vs per-element phase error for oracle, predictive and reactive."""
    res = pn["precoders"][f"dft{CODEBOOK}"]
    sig = [float(s) for s in res]
    snr = lambda m: [res[s][m]["snr_db"] for s in res]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.plot(sig, snr("oracle"), color=INK, lw=2.2, ls=":", marker="o", ms=7, label="Oracle")
    ax.plot(sig, snr(PROPOSED), color=ACCENT, lw=3.2, marker="o", ms=9, label=LABEL)
    ax.plot(sig, snr("reactive"), color=MID, lw=2.6, ls="--", marker="s", ms=8, label="Reactive")
    gap = [p - r for p, r in zip(snr(PROPOSED), snr("reactive"))]
    for i in (0, len(sig) - 1):
        ax.annotate("", xy=(sig[i], snr(PROPOSED)[i]), xytext=(sig[i], snr("reactive")[i]),
                    arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.6))
        ax.text(sig[i] + (0.012 if i == 0 else -0.012), (snr(PROPOSED)[i] + snr("reactive")[i]) / 2,
                f"+{gap[i]:.2f} dB", ha="left" if i == 0 else "right", va="center",
                fontsize=13, color=ACCENT, fontweight="bold")
    ax.set_xlabel(r"Per-element phase error $\sigma_\phi$ (rad)")
    ax.set_ylabel("Mean SNR (dB)")
    ax.set_xlim(-0.03, sig[-1] + 0.03)
    ax.set_ylim(29.0, 33.0)
    ax.legend(loc="upper right")
    title(ax, "The Predictive Gain Survives Hardware Phase Error",
          r"i.i.d. per-element phase error $\phi \sim \mathcal{N}(0, \sigma_\phi^2)$, all methods; "
          f"{pn['draws']} Monte Carlo trials")
    save(fig, out, "fig4_phase_noise")


def fig_spectral_efficiency(cb, out):
    """Effective SE after beam-search time, at three beam-update rates."""
    tr = cb["codebooks"][CODEBOOK]["training"]
    # tau = one SSB (~36 us) / update interval
    rates = [("0.00036", "Every 100 ms"), ("0.0018", "Every 20 ms"), ("0.01", "Every 3.6 ms")]
    series = [(tr["exhaustive"], f"Full beam sweep ({tr['exhaustive']['beams_measured']} beams)", LIGHT, MID),
              (tr["reactive"]["1"], "Reactive (previous beam)", MID, MID),
              (tr[PROPOSED]["3"], f"{LABEL}, 3 beams", ACCENT, ACCENT)]
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x, w = np.arange(len(rates)), 0.26
    for j, (r, lab, fill, edge) in enumerate(series):
        vals = [r["se_eff"][t] for t, _ in rates]
        pos = x + (j - 1) * w
        ax.bar(pos, vals, w - 0.03, color=fill, edgecolor=edge, linewidth=1.2, label=lab, zorder=2)
        for p, v in zip(pos, vals):
            txt = f"{v:.1f}" if v > 0 else "0\n(no time\nleft)"
            ax.text(p, v + 0.15, txt, ha="center", va="bottom", fontsize=12.5,
                    color=ACCENT if j == 2 else DARK, fontweight="bold" if j == 2 else "normal")
    ax.set_xticks(x, [f"Beam update\n{lab.lower()}" for _, lab in rates])
    ax.set_ylabel("Effective spectral efficiency (b/s/Hz)")
    ax.set_ylim(0, 16.5)
    ax.set_yticks(range(0, 13, 2))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    title(ax, "Prediction keeps throughput as beams update faster",
          "Spectral efficiency after subtracting beam-search time (5G NR SSB timing)")
    save(fig, out, "fig5_spectral_efficiency")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default=str(ROOT / "results"))
    args = p.parse_args(argv)
    results = pathlib.Path(args.results)
    out = results / "poster"
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("fig*"):
        old.unlink()

    study = json.loads((results / "streaming_study.json").read_text())
    cb = json.loads((results / "codebook_eval.json").read_text())
    pn = json.loads((results / "phase_noise_eval.json").read_text())
    fig_generalisation(study, out)
    fig_gain(cb, out)
    fig_mrt_vs_hybrid(cb, out)
    fig_phase_noise(pn, out)
    fig_spectral_efficiency(cb, out)
    print(f"figures -> {out}")


if __name__ == "__main__":
    main()
