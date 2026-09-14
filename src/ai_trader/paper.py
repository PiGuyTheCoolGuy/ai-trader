"""Forward-only paper execution at observed bid/ask; atomic SQLite event/state saves."""

import hashlib
import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from filelock import FileLock

from .broker import Account
from .config import Config
from .data import API, book_quote, get, live_candles
from .features import build_features
from .training import load_run

log = logging.getLogger(__name__)


class PaperStore:
    def __init__(self, path: Path, fingerprint: str, cfg: Config):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, payload TEXT NOT NULL)"
        )
        row = self.db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        if row:
            self.state = json.loads(row[0])
            if self.state["fingerprint"] != fingerprint:
                self.db.close()
                raise ValueError(
                    "Paper state belongs to a different model/config. Choose a new --state file."
                )
        else:
            capital = cfg.risk.initial_equity / len(cfg.data.symbols)
            self.state = {
                "fingerprint": fingerprint,
                "initial_equity": cfg.risk.initial_equity,
                "accounts": {s: Account.create(capital).to_dict() for s in cfg.data.symbols},
                "quotes": {},
                "updated_at": None,
            }
            self.save([])

    def save(self, events: list[dict]):
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO state VALUES (1, ?)",
                (json.dumps(self.state, allow_nan=False),),
            )
            self.db.executemany(
                "INSERT INTO events(payload) VALUES (?)",
                [(json.dumps(e, allow_nan=False),) for e in events],
            )

    def close(self):
        self.db.close()


def act(
    account: Account,
    bid: float,
    ask: float,
    now: float,
    cfg: Config,
    score: float | None,
    threshold: float | None,
    fresh_signal: bool,
) -> list[dict]:
    risk, events = cfg.risk, []
    changed = account.new_day(now, bid)
    reason = account.check_risk(bid, risk)
    if account.position:
        p = account.position
        reason = (
            reason
            or ("day_end" if changed and risk.flatten_utc_day else None)
            or ("stop_loss" if bid <= p["stop"] else None)
            or ("take_profit" if bid >= p["take"] else None)
            or ("horizon" if now >= p["due"] else None)
        )
        if reason:
            trade = account.exit(bid * (1 - risk.slippage_bps / 10000), now, risk, reason)
            events.append({"type": "sell", **trade})
            account.check_risk(bid, risk)
    if (
        not events
        and fresh_signal
        and score is not None
        and np.isfinite(score)
        and threshold is not None
        and score >= threshold
    ):
        event = account.enter(
            ask * (1 + risk.slippage_bps / 10000),
            now,
            now + cfg.model.horizon_bars * cfg.seconds,
            risk,
            score,
        )
        if event:
            events.append(event)
    return events


def run_paper(
    run_dir: Path,
    state_path: Path,
    poll_seconds: float = 10,
    once: bool = False,
    max_signal_age: float = 60,
) -> None:
    if poll_seconds < 1 or not np.isfinite(poll_seconds) or not 0 < max_signal_age <= 300:
        raise ValueError("poll-seconds must be >= 1 and max-signal-age must be in (0, 300]")
    model, metadata, cfg = load_run(run_dir)
    fingerprint = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        "FORWARD PAPER ONLY. No exchange account or API key. Threshold=%s bps",
        metadata["threshold_bps"],
    )
    # Lock is held for the entire process. Two workers cannot double-trade one state.
    with FileLock(str(state_path) + ".lock", timeout=0):
        store = PaperStore(state_path, fingerprint, cfg)
        primed = set()
        try:
            while True:
                cycle = time.monotonic()
                failures = 0
                for symbol in cfg.data.symbols:
                    try:
                        server_ms = int(get(API + "/api/v3/time").json()["serverTime"])
                        observed = time.monotonic()
                        latest, score = None, None
                        signal_status = "unavailable"
                        try:
                            candles = live_candles(symbol, cfg, server_ms)
                            if candles.empty:
                                raise ValueError(f"No closed candles for {symbol}")
                            latest = candles.timestamp.iloc[-1]
                            features = build_features(candles, cfg.seconds)
                            row = features.iloc[[-1]][metadata["features"]]
                            if row.notna().all(axis=1).iloc[0]:
                                value = float(model.predict(row, num_threads=cfg.threads)[0])
                                score = value if np.isfinite(value) else None
                            signal_status = "ready" if score is not None else "warming_up"
                        except (ValueError, RuntimeError, OSError) as exc:
                            log.warning(
                                "%s signals unavailable: %s; checking quote-based exits",
                                symbol,
                                exc,
                            )
                        bid, ask = book_quote(symbol)
                        now = server_ms / 1000 + time.monotonic() - observed
                        age = (
                            now - (latest.timestamp() + cfg.seconds) if latest is not None else None
                        )
                        if age is not None and (age < 0 or age > cfg.seconds):
                            signal_status = "stale"
                            log.warning(
                                "%s old signal (%.1fs); managing positions at fresh quotes",
                                symbol,
                                age,
                            )
                        account = Account(**store.state["accounts"][symbol])
                        new_signal = (
                            latest is not None and account.last_signal != latest.isoformat()
                        )
                        fresh = (
                            symbol in primed
                            and new_signal
                            and signal_status == "ready"
                            and age is not None
                            and 0 <= age <= max_signal_age
                        )
                        events = act(
                            account, bid, ask, now, cfg, score, metadata["threshold_bps"], fresh
                        )
                        # First observation after EVERY restart only primes the next signal.
                        primed.add(symbol)
                        if latest is not None:
                            account.last_signal = latest.isoformat()
                        store.state["accounts"][symbol] = account.to_dict()
                        store.state["quotes"][symbol] = {
                            "bid": bid,
                            "ask": ask,
                            "observed_at": now,
                            "signal_bps": score,
                            "signal_time": latest.isoformat() if latest is not None else None,
                            "signal_status": signal_status,
                            "equity": account.equity(bid),
                            "signal_age_seconds": age,
                        }
                        store.state["updated_at"] = datetime.fromtimestamp(
                            now, timezone.utc
                        ).isoformat()
                        store.save([{**e, "symbol": symbol} for e in events])
                        log.info(
                            "%s equity=%.2f signal=%s bps position=%s events=%d",
                            symbol,
                            account.equity(bid),
                            f"{score:.3f}" if score is not None else "warming up",
                            bool(account.position),
                            len(events),
                        )
                    except (ValueError, RuntimeError, OSError) as exc:
                        failures += 1
                        log.error("%s: %s; waiting for fresh data", symbol, exc)
                        if once:
                            raise
                if once:
                    return
                delay = max(1, poll_seconds - (time.monotonic() - cycle))
                time.sleep(max(delay, 30 if failures else 0))
        finally:
            store.close()


def status(path: Path) -> dict:
    if not path.exists():
        raise ValueError(f"No paper session at {path}")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT payload FROM state WHERE id=1").fetchone()
        if not row:
            raise ValueError("Paper session has no saved state")
        state = json.loads(row[0])
        recent = db.execute("SELECT payload FROM events ORDER BY id DESC LIMIT 10").fetchall()
        events = db.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    return {**state, "event_count": events, "recent_events": [json.loads(r[0]) for r in recent]}
