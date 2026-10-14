# tests/test_run_study.py
import json

import numpy as np
import pytest

import experiments.run_study as run_study
from experiments.run_study import main
from src.data import masked_mae_abs


@pytest.fixture(autouse=True)
def small_imm_grid(monkeypatch):
    """The real IMM grid is 120 candidates (~15 s per tuning pass even on 3 trajectories, and the
    study tunes it 4x). Wiring tests only need the grid mechanism, so use 8 candidates."""
    monkeypatch.setattr(run_study, "IMM_GRID",
                        dict(q=[0.3, 3.0], omega=[0.25, 0.8], p_stay=[0.9, 0.99]))


def test_smoke_run_writes_complete_results(tmp_path):
    out, models = tmp_path / "results", tmp_path / "models"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(models)])

    study = json.loads((out / "streaming_study.json").read_text())
    assert set(study) >= {"meta", "classical", "learned", "generalisation", "bench"}
    assert "generalisation_detail" in study
    assert {"hold", "constant_velocity", "ls_linear", "kalman", "imm"} <= set(study["classical"])
    for key in ("lstm_seq_direct", "minlstm_seq_direct", "minlstm_seq_hybrid"):
        entry = study["learned"][key]
        assert len(entry["test_mae"]) == 1 and np.isfinite(entry["test_mae"][0])
        assert set(entry["per_regime"]) <= {"pedestrian", "cyclist", "vehicular"}
        # streaming (init_state + step) and parallel forward compute the same function
        assert abs(entry["stream_mae"] - entry["test_mae"][0]) < 1e-4
        assert entry["stream_max_abs_diff"] < 1e-3
        assert len(entry["per_regime_seeds"]) == len(entry["test_mae"]) == len(entry["train_seconds"])
        assert set(entry["per_regime_std"]) == set(entry["per_regime"])
        assert entry["search_seconds"] >= 0
    assert study["meta"]["cpu_threads"] >= 1 and study["meta"]["classical_total_seconds"] > 0
    assert study["meta"]["generalisation_total_seconds"] > 0
    assert study["bench"]["train_step"]["minlstm_seq"]["seq_lens"] == [16, 64]
    assert study["bench"]["macs"]["minlstm_seq"] == 12800
    for row, detail in zip(study["generalisation"], study["generalisation_detail"]):
        assert row["regime"] == detail["regime"]
        assert set(detail["classical"]) == {"kalman", "imm"}
        assert set(detail["learned"]) == set(study["learned"])

    preds = np.load(out / "streaming_predictions.npz")
    assert preds["truth"].ndim == 3
    assert preds["pred_imm"].shape == preds["truth"].shape
    assert preds["pred_minlstm_seq_hybrid"].shape == preds["truth"].shape
    assert (models / "minlstm_seq_direct.pt").exists()
    assert (models / "minlstm_seq_direct_s0.pt").exists()
    assert np.array_equal(preds["pred_minlstm_seq_direct_s0"], preds["pred_minlstm_seq_direct"])
    assert list(out.glob("streaming_*.png"))

    # the saved predictions reproduce the reported numbers
    for name, entry in study["classical"].items():
        got = masked_mae_abs(preds[f"pred_{name}"], preds["truth"], int(preds["warmup"]))
        assert got == pytest.approx(entry["test_mae"], rel=1e-5)
    for key, entry in study["learned"].items():
        got = masked_mae_abs(preds[f"pred_{key}"], preds["truth"], int(preds["warmup"]))
        assert got == pytest.approx(entry["test_mae"][0], rel=1e-5)


def test_skip_flags(tmp_path):
    out = tmp_path / "r"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(tmp_path / "m"),
          "--skip-bench", "--skip-generalisation"])
    study = json.loads((out / "streaming_study.json").read_text())
    assert study["bench"] == {} and study["generalisation"] == []


def test_resume_reuses_finished_configs(tmp_path):
    out, models = tmp_path / "r", tmp_path / "m"
    argv = ["--smoke", "--device", "cpu", "--out", str(out), "--models", str(models),
            "--skip-bench", "--skip-generalisation"]
    main(argv)
    json_path = out / "streaming_study.json"
    study = json.loads(json_path.read_text())
    study["learned"]["minlstm_seq_direct"]["marker"] = 1
    json_path.write_text(json.dumps(study))
    ckpt = models / "minlstm_seq_direct.pt"
    mtime = ckpt.stat().st_mtime_ns

    main(argv + ["--resume"])
    study = json.loads(json_path.read_text())
    assert study["learned"]["minlstm_seq_direct"]["marker"] == 1   # not retrained
    assert ckpt.stat().st_mtime_ns == mtime                         # checkpoint untouched
    preds = np.load(out / "streaming_predictions.npz")
    assert preds["pred_minlstm_seq_direct"].shape == preds["truth"].shape
    got = masked_mae_abs(preds["pred_minlstm_seq_direct"], preds["truth"], int(preds["warmup"]))
    assert got == pytest.approx(study["learned"]["minlstm_seq_direct"]["test_mae"][0], rel=1e-4)
    # per-seed predictions are regenerated from the per-seed checkpoints
    assert preds["pred_minlstm_seq_direct_s0"].shape == preds["truth"].shape
    np.testing.assert_allclose(preds["pred_minlstm_seq_direct_s0"],
                               preds["pred_minlstm_seq_direct"], atol=1e-5)


def test_resume_without_seed_checkpoints_falls_back_to_seed_zero(tmp_path):
    out, models = tmp_path / "r", tmp_path / "m"
    argv = ["--smoke", "--device", "cpu", "--out", str(out), "--models", str(models),
            "--skip-bench", "--skip-generalisation"]
    main(argv)
    for f in models.glob("*_s[0-9].pt"):
        f.unlink()
    main(argv + ["--resume"])
    preds = np.load(out / "streaming_predictions.npz")
    assert preds["pred_lstm_seq_hybrid_s0"].shape == preds["truth"].shape
