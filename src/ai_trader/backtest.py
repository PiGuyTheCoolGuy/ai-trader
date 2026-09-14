"""Next-open bar simulation; unknown intrabar order is resolved stop-first."""

import numpy as np
import pandas as pd

from .broker import Account
from .config import Config

TRADE_COLUMNS = [
    "symbol",
    "entry_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "quantity",
    "cost",
    "entry_fee",
    "exit_fee",
    "fees",
    "pnl",
    "reason",
    "signal_bps",
    "stop",
    "take",
    "due",
]


def simulate(
    frame: pd.DataFrame, scores: np.ndarray, cfg: Config, threshold: float | None, capital: float
) -> dict:
    if len(frame) != len(scores) or frame.empty:
        raise ValueError("Need nonempty candles with exactly one score per row")
    risk = cfg.risk
    account = Account.create(capital)
    slip = risk.slippage_bps / 10000
    times = frame.timestamp.astype("int64").to_numpy() // 10**9
    values = frame[["open", "high", "low", "close", "volume"]].to_numpy()
    trades, curve = [], []
    diagnostics = {
        "eligible_signals": 0,
        "entries": 0,
        "risk_blocked": 0,
        "size_blocked": 0,
        "gap_bars": 0,
    }
    for i, (timestamp, bar) in enumerate(zip(times, values)):
        op, high, low, close, volume = bar
        day_changed = account.new_day(float(timestamp), op)
        gap = i > 0 and timestamp - times[i - 1] != cfg.seconds
        diagnostics["gap_bars"] += int(gap)
        risk_reason = account.check_risk(op, risk)
        exited = False
        if account.position:
            p = account.position
            reason = (
                risk_reason
                or ("data_gap" if gap else None)
                or ("day_end" if day_changed and risk.flatten_utc_day else None)
                or ("stop_loss" if op <= p["stop"] else None)
                or ("take_profit" if op >= p["take"] else None)
                or ("horizon" if timestamp >= p["due"] else None)
            )
            if reason:
                trades.append(account.exit(op * (1 - slip), float(timestamp), risk, reason))
                account.check_risk(op, risk)
                exited = True
        # Previous CLOSED candle signal only. Never enter on a missing/gapped signal.
        score = scores[i - 1] if i and not gap else np.nan
        eligible = threshold is not None and np.isfinite(score) and score >= threshold
        if eligible:
            diagnostics["eligible_signals"] += 1
        if eligible and not exited and not account.position and volume > 0:
            if account.halted or account.day_halted:
                diagnostics["risk_blocked"] += 1
            else:
                event = account.enter(
                    op * (1 + slip),
                    float(timestamp),
                    float(timestamp + cfg.model.horizon_bars * cfg.seconds),
                    risk,
                    float(score),
                )
                diagnostics["entries"] += int(event is not None)
                diagnostics["size_blocked"] += int(event is None)
        if account.position:
            p = account.position
            # Applies on entry bar too; low/high contain the entire bar after open.
            if low <= p["stop"]:
                trades.append(
                    account.exit(
                        min(op, p["stop"]) * (1 - slip),
                        float(timestamp + cfg.seconds),
                        risk,
                        "stop_loss",
                    )
                )
            elif high >= p["take"]:
                trades.append(
                    account.exit(
                        p["take"] * (1 - slip), float(timestamp + cfg.seconds), risk, "take_profit"
                    )
                )
        reason = account.check_risk(close, risk)
        # Equity circuit breakers are sampled at bar open/close, not tick-perfect.
        if reason and account.position:
            trades.append(
                account.exit(close * (1 - slip), float(timestamp + cfg.seconds), risk, reason)
            )
        curve.append(account.equity(close))
    if account.position:
        trades.append(
            account.exit(
                values[-1][3] * (1 - slip), float(times[-1] + cfg.seconds), risk, "end_of_data"
            )
        )
        curve[-1] = account.cash
    equity = pd.Series(curve, index=frame.timestamp, name="equity")
    return {
        "equity": equity,
        "trades": trades,
        "diagnostics": diagnostics,
        "risk_halted": account.halted,
    }


def metrics(equity: pd.Series, trades: list[dict], initial: float, halted: bool = False) -> dict:
    values = np.r_[initial, equity.to_numpy()]
    drawdown = values / np.maximum.accumulate(values) - 1
    pnl = np.array([t["pnl"] for t in trades])
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    daily = equity.resample("1D").last().dropna()
    daily_return = daily.pct_change().dropna()
    sharpe = (
        float(np.sqrt(365) * daily_return.mean() / daily_return.std())
        if len(daily_return) >= 2 and daily_return.std() > 0
        else None
    )
    return {
        "initial_equity": initial,
        "final_equity": float(values[-1]),
        "return_pct": float((values[-1] / initial - 1) * 100),
        "max_drawdown_pct": float(-drawdown.min() * 100),
        "closed_trades": len(trades),
        "fills": 2 * len(trades),
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "profit_factor": float(gains / losses) if losses > 0 else None,
        "fees_paid": float(sum(t["fees"] for t in trades)),
        "realized_pnl": float(pnl.sum()),
        "risk_halted": bool(halted),
        "daily_sharpe_365d": sharpe,
    }


def portfolio(
    frames: dict[str, pd.DataFrame],
    scores: dict[str, np.ndarray],
    cfg: Config,
    threshold: float | None,
) -> dict:
    capital = cfg.risk.initial_equity / len(frames)
    curves, baseline_curves, trades, by_symbol, diagnostics = {}, {}, [], {}, {}
    baseline_trades = []
    halted = False
    for symbol, frame in frames.items():
        result = simulate(frame, scores[symbol], cfg, threshold, capital)
        curves[symbol] = result["equity"]
        trades.extend([{**t, "symbol": symbol} for t in result["trades"]])
        by_symbol[symbol] = metrics(
            result["equity"], result["trades"], capital, result["risk_halted"]
        )
        diagnostics[symbol] = result["diagnostics"]
        halted |= result["risk_halted"]
        fee, slip = cfg.risk.fee_bps / 10000, cfg.risk.slippage_bps / 10000
        quantity = capital / (float(frame.open.iloc[0]) * (1 + slip) * (1 + fee))
        baseline = frame.close * quantity
        baseline.iloc[-1] *= (1 - slip) * (1 - fee)
        baseline_trades.append(
            {
                "pnl": float(baseline.iloc[-1] - capital),
                "fees": capital * fee / (1 + fee)
                + quantity * float(frame.close.iloc[-1]) * (1 - slip) * fee,
            }
        )
        baseline_curves[symbol] = pd.Series(baseline.to_numpy(), index=frame.timestamp)
    equity = pd.DataFrame(curves).sort_index().ffill().fillna(capital).sum(axis=1)
    baseline = pd.DataFrame(baseline_curves).sort_index().ffill().fillna(capital).sum(axis=1)
    return {
        "metrics": metrics(equity, trades, cfg.risk.initial_equity, halted),
        "by_symbol": by_symbol,
        "trades": sorted(trades, key=lambda t: t["entry_time"]),
        "equity": equity,
        "buy_hold_equity": baseline,
        "diagnostics": diagnostics,
        "buy_hold": metrics(baseline, baseline_trades, cfg.risk.initial_equity),
    }
