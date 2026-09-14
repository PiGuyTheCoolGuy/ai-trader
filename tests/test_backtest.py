import numpy as np
import pandas as pd

from ai_trader.backtest import metrics, portfolio, simulate
from ai_trader.broker import Account
from ai_trader.training import choose_threshold


def bars(prices):
    prices = np.array(prices, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=len(prices), freq="5min", tz="UTC"),
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": 100,
        }
    )


def test_signal_cannot_fill_same_candle(config):
    frame = bars([100, 110, 110])
    result = simulate(frame, np.array([100, 0, 0]), config, 10, 1000)
    trade = result["trades"][0]
    assert trade["entry_price"] == 110
    assert trade["entry_time"] == frame.timestamp.iloc[1].timestamp()


def test_stop_wins_when_bar_touches_both(config):
    frame = bars([100, 100, 100])
    frame.loc[1, ["high", "low"]] = [120, 80]
    result = simulate(frame, np.array([100, 0, 0]), config, 10, 1000)
    assert result["trades"][0]["reason"] == "stop_loss"
    assert result["trades"][0]["exit_price"] == 98.5


def test_stop_gap_fills_at_open_not_optimistic_stop(config):
    frame = bars([100, 100, 90])
    result = simulate(frame, np.array([100, 0, 0]), config, 10, 1000)
    assert result["trades"][0]["exit_price"] == 90


def test_fees_slippage_and_cash_reconcile(config):
    frame = bars([100, 100, 100])
    config.risk.fee_bps, config.risk.slippage_bps = 10, 5
    result = simulate(frame, np.array([100, 0, 0]), config, 10, 1000)
    m = metrics(result["equity"], result["trades"], 1000)
    assert m["return_pct"] < 0 and m["fees_paid"] > 0
    assert np.isclose(m["final_equity"], 1000 + m["realized_pnl"])


def test_horizon_exit_is_at_correct_open(config):
    config.model.horizon_bars = 2
    frame = bars([100] * 6)
    result = simulate(frame, np.array([100, 0, 0, 0, 0, 0]), config, 10, 1000)
    t = result["trades"][0]
    assert t["reason"] == "horizon"
    assert t["exit_time"] - t["entry_time"] == 600


def test_gap_invalidates_pending_entry(config):
    frame = bars([100] * 4).drop(index=1).reset_index(drop=True)
    result = simulate(frame, np.array([100, 0, 0]), config, 10, 1000)
    assert not result["trades"]


def test_cash_is_a_valid_strategy_and_allocation_is_total(config):
    frame = bars([100] * 6)
    result = portfolio({"A": frame, "B": frame}, {"A": np.ones(6), "B": np.ones(6)}, config, None)
    assert result["metrics"]["final_equity"] == config.risk.initial_equity
    assert result["metrics"]["closed_trades"] == 0


def test_unprofitable_threshold_is_not_forced():
    candidates = [
        {
            "threshold_bps": 10,
            "metrics": {
                "closed_trades": 100,
                "return_pct": -1,
                "max_drawdown_pct": 2,
                "risk_halted": False,
            },
        }
    ]
    threshold, reason = choose_threshold(candidates, 30)
    assert threshold is None and "Cash selected" in reason
    candidates[0]["metrics"]["return_pct"] = 10
    assert choose_threshold(candidates, 30)[0] == 10


def test_daily_halt_resets_but_drawdown_halt_persists(config):
    account = Account.create(1000)
    config.risk.daily_loss_limit, config.risk.max_drawdown = 0.02, 0.10
    account.new_day(1704067200, 100)
    account.cash = 970
    assert account.check_risk(100, config.risk) == "daily_loss_limit"
    account.new_day(1704153600, 100)
    assert not account.day_halted
    account.cash = 890
    assert account.check_risk(100, config.risk) == "drawdown_limit"
    account.new_day(1704240000, 100)
    assert account.halted
