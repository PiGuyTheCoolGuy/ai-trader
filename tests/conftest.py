import numpy as np
import pandas as pd
import pytest

from ai_trader.config import Config


def candles(n=1500, start="2023-01-01", seed=5, seconds=300):
    rng = np.random.default_rng(seed)
    changes = rng.normal(0.00003, 0.002, n)
    close = 100 * np.exp(np.cumsum(changes))
    op = np.r_[close[0], close[:-1]]
    volume = rng.uniform(100, 200, n)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range(start, periods=n, freq=f"{seconds}s", tz="UTC"),
            "open": op,
            "high": np.maximum(op, close) * 1.002,
            "low": np.minimum(op, close) * 0.998,
            "close": close,
            "volume": volume,
            "quote_volume": volume * close,
            "trades": rng.integers(50, 100, n),
            "taker_volume": volume * 0.5,
        }
    )


@pytest.fixture
def history():
    return candles()


@pytest.fixture
def config():
    cfg = Config()
    cfg.data.symbols = ["BTCUSDT"]
    cfg.risk.fee_bps = 0
    cfg.risk.slippage_bps = 0
    cfg.risk.position_fraction = 0.5
    cfg.risk.risk_per_trade = 0.5
    cfg.risk.max_drawdown = 0.9
    cfg.risk.daily_loss_limit = 0.9
    cfg.risk.flatten_utc_day = False
    cfg.model.threads = 1
    return cfg
