"""Strict, JSON-based configuration; all prices/returns use explicit units."""

import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

INTERVALS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


@dataclass
class DataConfig:
    symbols: list[str] = field(default_factory=lambda: ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    interval: str = "5m"
    start: str = "2021-01-01"
    end: str | None = None  # Exclusive UTC date; None = start of today UTC.
    directory: str = "data"
    workers: int = 4
    retries: int = 4
    timeout_seconds: int = 30
    refresh: bool = False


@dataclass
class ModelConfig:
    threads: int = 0  # 0 => os.cpu_count(); no GPU option.
    horizon_bars: int = 12
    rounds: int = 2000
    early_stopping_rounds: int = 100
    trials: int = 3
    seed: int = 42
    train_fraction: float = 0.55
    tune_fraction: float = 0.15
    calibration_fraction: float = 0.15
    min_split_rows: int = 500
    min_calibration_trades: int = 30
    min_edge_bps: float = 5.0


@dataclass
class RiskConfig:
    initial_equity: float = 10000.0  # Total, divided equally among symbols.
    fee_bps: float = 10.0  # PER SIDE; illustrative, configurable.
    slippage_bps: float = 3.0  # PER SIDE, including estimated spread in backtest.
    position_fraction: float = 0.25  # Of one symbol's allocation.
    risk_per_trade: float = 0.005
    stop_loss: float = 0.015
    take_profit: float = 0.03
    daily_loss_limit: float = 0.03
    max_drawdown: float = 0.15
    min_notional: float = 10.0
    flatten_utc_day: bool = True


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    output: str = "artifacts"

    @property
    def seconds(self) -> int:
        return INTERVALS[self.data.interval]

    @property
    def threads(self) -> int:
        return self.model.threads or (os.cpu_count() or 1)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "Config":
        d, m, r = self.data, self.model, self.risk
        if d.interval not in INTERVALS:
            raise ValueError(f"interval must be one of {list(INTERVALS)}")
        if not d.symbols or len(set(d.symbols)) != len(d.symbols):
            raise ValueError("symbols must be a nonempty list without duplicates")
        if any(not re.fullmatch(r"[A-Z0-9]{5,25}", s) or not s.endswith("USDT") for s in d.symbols):
            raise ValueError("Use uppercase USDT spot symbols, e.g. BTCUSDT")
        start = date.fromisoformat(d.start)
        if d.end is not None and date.fromisoformat(d.end) <= start:
            raise ValueError("end must be after start (end is exclusive)")
        positive_ints = [
            d.workers,
            d.retries,
            d.timeout_seconds,
            m.horizon_bars,
            m.rounds,
            m.early_stopping_rounds,
            m.trials,
            m.min_split_rows,
            m.min_calibration_trades,
        ]
        if any(type(v) is not int or v < 1 for v in positive_ints):
            raise ValueError("Counts, rounds, workers, and timeouts must be positive integers")
        if type(m.threads) is not int or m.threads < 0 or m.trials > 6:
            raise ValueError("threads must be >= 0 and trials must be <= 6")
        fractions = [m.train_fraction, m.tune_fraction, m.calibration_fraction]
        if any(not 0 < f < 1 for f in fractions) or sum(fractions) >= 0.95:
            raise ValueError("Split fractions must be positive and leave at least 5% for test")
        for name in [
            "position_fraction",
            "risk_per_trade",
            "stop_loss",
            "take_profit",
            "daily_loss_limit",
            "max_drawdown",
        ]:
            if not 0 < getattr(r, name) < 1:
                raise ValueError(f"risk.{name} must be between 0 and 1")
        for name in ["fee_bps", "slippage_bps", "initial_equity", "min_notional"]:
            value = getattr(r, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"risk.{name} must be finite and nonnegative")
        if (
            r.fee_bps >= 1000
            or r.slippage_bps >= 1000
            or r.initial_equity <= 0
            or r.min_notional <= 0
        ):
            raise ValueError("Invalid costs, initial equity, or minimum notional")
        if not math.isfinite(m.min_edge_bps) or m.min_edge_bps < 0:
            raise ValueError("min_edge_bps must be finite and nonnegative")
        if type(r.flatten_utc_day) is not bool or type(d.refresh) is not bool:
            raise ValueError("flatten_utc_day and refresh must be JSON booleans")
        return self


def load_config(path: str | Path | None = None) -> Config:
    raw = json.loads(Path(path).read_text()) if path else {}
    extra = set(raw) - {"data", "model", "risk", "output"}
    if extra:
        raise ValueError(f"Unknown configuration keys: {sorted(extra)}")
    return Config(
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        risk=RiskConfig(**raw.get("risk", {})),
        output=raw.get("output", "artifacts"),
    ).validate()
