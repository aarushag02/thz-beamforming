# tests/test_run_study.py
import json

import numpy as np

from experiments.run_study import main


def test_smoke_run_writes_complete_results(tmp_path):
    out, models = tmp_path / "results", tmp_path / "models"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(models)])

    study = json.loads((out / "streaming_study.json").read_text())
    assert set(study) >= {"meta", "classical", "learned", "generalisation", "bench"}
    assert {"hold", "constant_velocity", "ls_linear", "kalman", "imm"} <= set(study["classical"])
    for key in ("lstm_seq_direct", "minlstm_seq_direct", "minlstm_seq_hybrid"):
        entry = study["learned"][key]
        assert len(entry["test_mae"]) == 1 and np.isfinite(entry["test_mae"][0])
        assert set(entry["per_regime"]) <= {"pedestrian", "cyclist", "vehicular"}
    assert study["bench"]["train_step"]["minlstm_seq"]["seq_lens"] == [16, 64]
    assert study["bench"]["macs"]["minlstm_seq"] == 12800

    preds = np.load(out / "streaming_predictions.npz")
    assert preds["truth"].ndim == 3
    assert preds["pred_imm"].shape == preds["truth"].shape
    assert preds["pred_minlstm_seq_hybrid"].shape == preds["truth"].shape
    assert (models / "minlstm_seq_direct.pt").exists()
    assert list(out.glob("streaming_*.png"))


def test_skip_flags(tmp_path):
    out = tmp_path / "r"
    main(["--smoke", "--device", "cpu", "--out", str(out), "--models", str(tmp_path / "m"),
          "--skip-bench", "--skip-generalisation"])
    study = json.loads((out / "streaming_study.json").read_text())
    assert study["bench"] == {} and study["generalisation"] == []
