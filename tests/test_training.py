import json

import numpy as np
import pytest

from ai_trader.config import load_config
from ai_trader.features import build_features
from ai_trader.training import load_run, train


def test_actual_cpu_training_artifact_roundtrip(tmp_path, config):
    from conftest import candles

    frame = candles(7000)
    config.data.directory = str(tmp_path / "data")
    config.output = str(tmp_path / "artifacts")
    config.data.start, config.data.end = "2023-01-01", None
    config.model.rounds, config.model.early_stopping_rounds = 15, 5
    config.model.trials = 1
    config.model.min_split_rows = 100
    config.model.min_calibration_trades = 1
    path = tmp_path / "data" / "processed"
    path.mkdir(parents=True)
    frame.to_parquet(path / "BTCUSDT-5m.parquet", index=False)
    run = train(config)
    model, metadata, loaded = load_run(run)
    report = json.loads((run / "training_report.json").read_text())
    assert report["training_completed"] and report["device"] == "cpu"
    assert report["chosen_trees"] > 0
    assert report["trials"][0]["parameters"]["device_type"] == "cpu"
    assert "Untouched test period" in (run / "report.html").read_text()
    features = build_features(frame, 300).dropna()[metadata["features"]]
    assert np.isfinite(model.predict(features, num_threads=1)).all()
    assert loaded.to_dict() == config.to_dict()
    assert metadata["test_start"] == report["split_boundaries"][-1]
    (run / "model.txt").write_text("corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_run(run)


@pytest.mark.parametrize(
    "payload",
    [
        {"model": {"threads": -1}},
        {"model": {"trials": 7}},
        {"model": {"train_fraction": 0.99}},
        {"risk": {"fee_bps": -1}},
        {"risk": {"stop_loss": 0}},
        {"data": {"interval": "2m"}},
        {"data": {"symbols": ["BTCUSDT", "BTCUSDT"]}},
        {"model": {"typo": 5}},
    ],
)
def test_invalid_config_fails_early(tmp_path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload))
    with pytest.raises((ValueError, TypeError)):
        load_config(path)
