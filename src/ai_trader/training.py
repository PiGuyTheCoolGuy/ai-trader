"""Multicore LightGBM with chronological, target-purged four-way evaluation."""

import json
import logging
import platform
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .backtest import TRADE_COLUMNS, portfolio
from .config import Config, load_config
from .data import load_history
from .features import FEATURE_VERSION, build_features, labeled, purged_split, split_times
from .utils import atomic_bytes, sha256, write_json

log = logging.getLogger(__name__)
PARAMETERS = [
    {"num_leaves": 15, "min_data_in_leaf": 500, "lambda_l2": 10.0, "learning_rate": 0.03},
    {"num_leaves": 31, "min_data_in_leaf": 300, "lambda_l2": 20.0, "learning_rate": 0.03},
    {"num_leaves": 63, "min_data_in_leaf": 500, "lambda_l2": 40.0, "learning_rate": 0.02},
    {"num_leaves": 15, "min_data_in_leaf": 1000, "lambda_l2": 50.0, "learning_rate": 0.02},
    {"num_leaves": 31, "min_data_in_leaf": 1000, "lambda_l2": 50.0, "learning_rate": 0.02},
    {"num_leaves": 63, "min_data_in_leaf": 1000, "lambda_l2": 100.0, "learning_rate": 0.02},
]


def prediction_stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"rows": 0}
    return {
        "rows": len(values),
        "mean_bps": float(values.mean()),
        "std_bps": float(values.std()),
        "quantiles_bps": {
            str(q): float(np.quantile(values, q)) for q in [0, 0.1, 0.5, 0.9, 0.99, 1]
        },
    }


def regression_metrics(target: np.ndarray, predictions: np.ndarray, constant: float) -> dict:
    corr = (
        float(np.corrcoef(target, predictions)[0, 1])
        if np.std(target) > 0 and np.std(predictions) > 0
        else None
    )
    return {
        "rmse_bps": float(np.sqrt(np.mean((predictions - target) ** 2))),
        "mae_bps": float(np.mean(np.abs(predictions - target))),
        "mean_baseline_rmse_bps": float(np.sqrt(np.mean((constant - target) ** 2))),
        "zero_baseline_rmse_bps": float(np.sqrt(np.mean(target**2))),
        "correlation": corr,
        "direction_accuracy_pct": float(np.mean(np.sign(predictions) == np.sign(target)) * 100),
    }


def choose_threshold(candidates: list[dict], min_trades: int) -> tuple[float | None, str]:
    best, best_score = None, 0.0  # Explicit no-trade/cash alternative.
    for candidate in candidates:
        m = candidate["metrics"]
        eligible = m["closed_trades"] >= min_trades and m["return_pct"] > 0 and not m["risk_halted"]
        score = m["return_pct"] - 0.5 * m["max_drawdown_pct"]
        candidate["selection_score"] = score
        candidate["eligible"] = eligible
        if eligible and score > best_score:
            best, best_score = candidate["threshold_bps"], score
    reason = (
        "Calibration selected a profitable, sufficiently active threshold after costs."
        if best is not None
        else "Cash selected: no threshold met the minimum trade count, positive net return, "
        "risk limits and positive return-minus-half-drawdown score on calibration data."
    )
    return best, reason


def score_frames(
    model: lgb.Booster,
    raw: dict[str, pd.DataFrame],
    columns: list[str],
    cfg: Config,
    begin: pd.Timestamp,
    end: pd.Timestamp | None = None,
):
    frames, scores = {}, {}
    for symbol, history in raw.items():
        features = build_features(history, cfg.seconds)
        valid = features[columns].notna().all(axis=1)
        prediction = np.full(len(history), np.nan)
        if valid.any():
            prediction[valid] = model.predict(features.loc[valid, columns], num_threads=cfg.threads)
        mask = history.timestamp >= begin
        if end is not None:
            mask &= history.timestamp < end
        frames[symbol] = history.loc[mask].reset_index(drop=True)
        scores[symbol] = prediction[mask]
        if frames[symbol].empty:
            raise ValueError(f"No evaluation candles for {symbol}")
    return frames, scores


def summary(result: dict) -> dict:
    return {k: result[k] for k in ["metrics", "by_symbol", "diagnostics", "buy_hold"]}


def train(cfg: Config) -> Path:
    started = time.perf_counter()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]
    run_dir = Path(cfg.output) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "config.json", cfg.to_dict())
    raw, parts, data_info = {}, [], {}
    for symbol in cfg.data.symbols:
        history = load_history(cfg, symbol)
        raw[symbol] = history
        part, columns = labeled(history, cfg.seconds, cfg.model.horizon_bars)
        part = part.assign(symbol=symbol)
        parts.append(part)
        data_info[symbol] = {
            "candles": len(history),
            "usable_labels": len(part),
            "first": history.timestamp.iloc[0].isoformat(),
            "last": history.timestamp.iloc[-1].isoformat(),
            "parquet_sha256": sha256(
                Path(cfg.data.directory) / "processed" / f"{symbol}-{cfg.data.interval}.parquet"
            ),
        }
        log.info(
            "Features %s: %s candles, %s labeled rows",
            symbol,
            f"{len(history):,}",
            f"{len(part):,}",
        )
    data = pd.concat(parts, ignore_index=True).sort_values(["timestamp", "symbol"])
    del parts
    boundaries = split_times(
        data.timestamp,
        [cfg.model.train_fraction, cfg.model.tune_fraction, cfg.model.calibration_fraction],
    )
    splits = purged_split(data, boundaries)
    split_info = {}
    for name, frame in splits.items():
        counts = frame.groupby("symbol").size().to_dict()
        if any(counts.get(s, 0) < cfg.model.min_split_rows for s in cfg.data.symbols):
            raise ValueError(
                f"{name} needs at least {cfg.model.min_split_rows} rows per symbol: {counts}"
            )
        split_info[name] = {
            "rows": len(frame),
            "by_symbol": counts,
            "first": frame.timestamp.min().isoformat(),
            "last": frame.timestamp.max().isoformat(),
            "last_label_end": frame.label_end.max().isoformat(),
        }
        log.info(
            "%s: %s rows, %s to %s",
            name.upper(),
            f"{len(frame):,}",
            split_info[name]["first"],
            split_info[name]["last"],
        )
    fit, tune = splits["train"], splits["tune"]
    constant = float(fit.target_bps.mean())
    log.info(
        "CPU ONLY: %d threads, %d features, %d trials, up to %d boosting rounds each",
        cfg.threads,
        len(columns),
        cfg.model.trials,
        cfg.model.rounds,
    )
    best_model, best_rmse, trials = None, float("inf"), []
    for i, extra in enumerate(PARAMETERS[: cfg.model.trials], 1):
        trial_start = time.perf_counter()
        params = {
            "objective": "regression",
            "metric": "rmse",
            "device_type": "cpu",
            "num_threads": cfg.threads,
            "seed": cfg.model.seed,
            "deterministic": True,
            "force_col_wise": True,
            "verbosity": -1,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            **extra,
        }
        log.info("Trial %d/%d: %s", i, cfg.model.trials, extra)
        training = lgb.Dataset(fit[columns], label=fit.target_bps, feature_name=columns)
        validation = lgb.Dataset(tune[columns], label=tune.target_bps, reference=training)
        history = {}
        model = lgb.train(
            params,
            training,
            num_boost_round=cfg.model.rounds,
            valid_sets=[validation],
            valid_names=["tune"],
            callbacks=[
                lgb.early_stopping(cfg.model.early_stopping_rounds),
                lgb.log_evaluation(50),
                lgb.record_evaluation(history),
            ],
        )
        prediction = model.predict(tune[columns], num_threads=cfg.threads)
        result = regression_metrics(tune.target_bps.to_numpy(), prediction, constant)
        result.update(
            {
                "trial": i,
                "parameters": params,
                "best_iteration": model.best_iteration,
                "trees": model.num_trees(),
                "elapsed_seconds": time.perf_counter() - trial_start,
                "learning_curve_rmse": history["tune"]["rmse"],
            }
        )
        trials.append(result)
        log.info(
            "Trial %d completed: %d trees, RMSE %.3f bps (mean baseline %.3f), %.1fs",
            i,
            model.num_trees(),
            result["rmse_bps"],
            result["mean_baseline_rmse_bps"],
            result["elapsed_seconds"],
        )
        if result["rmse_bps"] < best_rmse:
            best_model, best_rmse = model, result["rmse_bps"]
    model = best_model
    if model is None:
        raise ValueError("No model was trained")
    model_path = run_dir / "model.txt"
    atomic_bytes(model_path, model.model_to_string().encode())
    cal_frames, cal_scores = score_frames(model, raw, columns, cfg, boundaries[1], boundaries[2])
    all_scores = np.concatenate(list(cal_scores.values()))
    finite = all_scores[np.isfinite(all_scores)]
    if not len(finite):
        raise ValueError("No finite calibration predictions")
    cost_floor = 2 * (cfg.risk.fee_bps + cfg.risk.slippage_bps) + cfg.model.min_edge_bps
    thresholds = sorted(
        {
            cost_floor,
            *[
                max(cost_floor, float(np.quantile(finite, q)))
                for q in [0.5, 0.65, 0.75, 0.85, 0.9, 0.95, 0.975, 0.99]
            ],
        }
    )
    candidates = []
    for i, threshold in enumerate(thresholds, 1):
        log.info(
            "Calibration %d/%d: gross predicted return >= %.3f bps", i, len(thresholds), threshold
        )
        result = portfolio(cal_frames, cal_scores, cfg, threshold)
        candidates.append({"threshold_bps": threshold, **summary(result)})
    threshold, reason = choose_threshold(candidates, cfg.model.min_calibration_trades)
    log.info("%s Threshold: %s", reason, threshold)
    test_frames, test_scores = score_frames(model, raw, columns, cfg, boundaries[2])
    test_result = portfolio(test_frames, test_scores, cfg, threshold)
    test = splits["test"]
    test_predictions = model.predict(test[columns], num_threads=cfg.threads)
    report = {
        "run_id": run_id,
        "training_completed": True,
        "device": "cpu",
        "threads": cfg.threads,
        "feature_count": len(columns),
        "data": data_info,
        "split_boundaries": [t.isoformat() for t in boundaries],
        "splits": split_info,
        "purged_rows": len(data) - sum(len(x) for x in splits.values()),
        "trials": trials,
        "chosen_trees": model.num_trees(),
        "threshold_bps": threshold,
        "selection_reason": reason,
        "cost_floor_bps": cost_floor,
        "calibration_predictions": prediction_stats(all_scores),
        "calibration_candidates": candidates,
        "test_predictions": prediction_stats(np.concatenate(list(test_scores.values()))),
        "test_regression": regression_metrics(
            test.target_bps.to_numpy(), test_predictions, constant
        ),
        "test": summary(test_result),
        "elapsed_seconds": time.perf_counter() - started,
        "feature_importance_gain": dict(zip(columns, model.feature_importance("gain").tolist())),
        "versions": {
            "python": platform.python_version(),
            "lightgbm": lgb.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata = {
        "schema_version": 1,
        "feature_version": FEATURE_VERSION,
        "run_id": run_id,
        "features": columns,
        "threshold_bps": threshold,
        "selection_reason": reason,
        "model_sha256": sha256(model_path),
        "config": cfg.to_dict(),
        "test_start": boundaries[2].isoformat(),
    }
    write_json(run_dir / "metadata.json", metadata)
    write_json(run_dir / "training_report.json", report)
    pd.DataFrame(test_result["trades"], columns=TRADE_COLUMNS).to_csv(
        run_dir / "test_trades.csv", index=False
    )
    pd.DataFrame(
        {"equity": test_result["equity"], "buy_hold": test_result["buy_hold_equity"]}
    ).to_csv(run_dir / "test_equity.csv", index_label="timestamp")
    from .report import render_report

    render_report(run_dir, report, test_result)
    write_json(Path(cfg.output) / "latest.json", {"run_dir": str(run_dir.resolve())})
    log.info(
        "Training complete: %s; %d trees; %.1fs; test return %.2f%%, %d trades",
        run_dir,
        model.num_trees(),
        report["elapsed_seconds"],
        test_result["metrics"]["return_pct"],
        test_result["metrics"]["closed_trades"],
    )
    return run_dir


def load_run(run_dir: str | Path) -> tuple[lgb.Booster, dict, Config]:
    path = Path(run_dir)
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata["schema_version"] != 1 or metadata["feature_version"] != FEATURE_VERSION:
        raise ValueError("Incompatible model/feature version; retrain")
    if sha256(path / "model.txt") != metadata["model_sha256"]:
        raise ValueError("Saved model checksum mismatch")
    cfg = load_config(path / "config.json")
    if cfg.to_dict() != metadata["config"]:
        raise ValueError("Saved config changed; use a new run instead")
    model = lgb.Booster(model_file=str(path / "model.txt"))
    if model.feature_name() != metadata["features"]:
        raise ValueError("Saved feature schema mismatch")
    return model, metadata, cfg
