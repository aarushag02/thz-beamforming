import json

import numpy as np
import pytest

from experiments import beam_eval, make_tables


def _fake_predictions(path):
    truth = np.stack([np.stack([np.full(15, 260.0), np.linspace(400, 420, 15)], 1)] * 2)
    np.savez(path, truth=truth, regimes=np.array(["cyclist", "vehicular"]), warmup=10,
             pred_hold=truth - [0.0, 1.5], pred_imm=truth + [0.0, 0.1])


def test_beam_eval_ranks_better_predictor_lower(tmp_path):
    _fake_predictions(tmp_path / "streaming_predictions.npz")
    beam_eval.main(["--results", str(tmp_path), "--bs-pos", "235", "350"])
    out = json.loads((tmp_path / "beam_eval.json").read_text())
    assert out["n_tx"] == [8, 16, 32, 64, 128] and out["bs_source"] == "cli"
    hold, imm = out["methods"]["hold"], out["methods"]["imm"]
    assert all(i < h for i, h in zip(imm["misalign_bw"], hold["misalign_bw"]))
    assert hold["misalign_bw"][-1] > hold["misalign_bw"][0]   # narrower beams, more beamwidths


def test_beam_eval_without_position_exits_cleanly(tmp_path, monkeypatch):
    """Review focus 5."""
    _fake_predictions(tmp_path / "streaming_predictions.npz")
    monkeypatch.setattr(beam_eval, "bs_from_deepmimo", lambda: None)
    with pytest.raises(SystemExit) as exc:
        beam_eval.main(["--results", str(tmp_path)])
    assert "--bs-pos" in str(exc.value)


def test_make_tables_writes_tex(tmp_path):
    study = {
        "classical": {"imm": {"test_mae": 0.31, "params": {}}, "kalman": {"test_mae": 0.35, "params": {}}},
        "learned": {"minlstm_seq_hybrid": {"test_mean": 0.28, "test_std": 0.01, "n_params": 13186,
                                           "arch": "minlstm_seq", "mode": "hybrid"}},
        "generalisation": [{"regime": "vehicular", "n_unseen": 5, "imm": 0.4, "minlstm_seq_hybrid": 0.38}],
        "bench": {"device": "cuda", "macs": {"minlstm_seq": 12800, "lstm_seq": 49792},
                  "latency_us": {"minlstm_seq@cuda": {"step": 90.0, "window_10": 300.0, "window_40": 310.0}},
                  "train_step": {"minlstm_seq": {"seq_lens": [16, 64], "ms": [1.0, 1.1]}}},
    }
    res = tmp_path / "results"
    res.mkdir()
    (res / "streaming_study.json").write_text(json.dumps(study))
    make_tables.main(["--results", str(res), "--paper", str(tmp_path / "paper")])
    tables = tmp_path / "paper" / "tables"
    acc = (tables / "accuracy.tex").read_text()
    assert "0.280" in acc and "IMM" in acc and r"\pm" in acc
    assert (tables / "compute.tex").exists() and (tables / "generalisation.tex").exists()
    assert not (tables / "beam.tex").exists()
