"""Shared long-only paper accounting, sizing and persisted risk state."""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .config import RiskConfig


@dataclass
class Account:
    cash: float
    peak: float
    day_start: float
    day: str = ""
    day_halted: bool = False
    halted: bool = False
    position: dict | None = None
    last_signal: str | None = None

    @classmethod
    def create(cls, capital: float) -> "Account":
        return cls(cash=capital, peak=capital, day_start=capital)

    def to_dict(self) -> dict:
        return asdict(self)

    def equity(self, price: float) -> float:
        return self.cash + (self.position["quantity"] * price if self.position else 0)

    def new_day(self, timestamp: float, price: float) -> bool:
        day = datetime.fromtimestamp(timestamp, timezone.utc).date().isoformat()
        changed = bool(self.day and self.day != day)
        if self.day != day:
            self.day = day
            self.day_start = self.equity(price)
            self.day_halted = False
        return changed

    def check_risk(self, price: float, risk: RiskConfig) -> str | None:
        equity = self.equity(price)
        self.peak = max(self.peak, equity)
        if equity <= self.peak * (1 - risk.max_drawdown):
            self.halted = True
        if equity <= self.day_start * (1 - risk.daily_loss_limit):
            self.day_halted = True
        return "drawdown_limit" if self.halted else "daily_loss_limit" if self.day_halted else None

    def enter(
        self, price: float, timestamp: float, due: float, risk: RiskConfig, signal: float
    ) -> dict | None:
        if self.position or self.halted or self.day_halted:
            return None
        fee = risk.fee_bps / 10000
        # Sizing includes estimated round-trip costs in the stop-loss risk budget.
        stop_risk = risk.stop_loss + 2 * (risk.fee_bps + risk.slippage_bps) / 10000
        budget = min(
            self.cash * risk.position_fraction,
            self.cash * risk.risk_per_trade / stop_risk,
            self.cash / (1 + fee),
        )
        if budget < risk.min_notional:
            return None
        quantity = budget / price
        entry_fee = budget * fee
        self.cash -= budget + entry_fee
        self.position = {
            "entry_time": timestamp,
            "entry_price": price,
            "quantity": quantity,
            "cost": budget + entry_fee,
            "entry_fee": entry_fee,
            "stop": price * (1 - risk.stop_loss),
            "take": price * (1 + risk.take_profit),
            "due": due,
            "signal_bps": float(signal),
        }
        return {
            "type": "buy",
            "timestamp": timestamp,
            "price": price,
            "quantity": quantity,
            "fee": entry_fee,
            "signal_bps": float(signal),
        }

    def exit(self, price: float, timestamp: float, risk: RiskConfig, reason: str) -> dict:
        if not self.position:
            raise ValueError("Cannot sell without an open position")
        p = self.position
        proceeds = p["quantity"] * price
        fee = proceeds * risk.fee_bps / 10000
        self.cash += proceeds - fee
        trade = {
            **p,
            "exit_time": timestamp,
            "exit_price": price,
            "exit_fee": fee,
            "fees": p["entry_fee"] + fee,
            "pnl": proceeds - fee - p["cost"],
            "reason": reason,
        }
        self.position = None
        return trade
