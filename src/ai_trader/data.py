"""Public spot candles: verified monthly/daily archives and GET-only live data."""

import hashlib
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import Config
from .utils import atomic_bytes, sha256, write_json

log = logging.getLogger(__name__)
ARCHIVE = "https://data.binance.vision/data/spot"
API = "https://data-api.binance.vision"
COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_volume",
    "taker_quote_volume",
    "ignore",
]
KEEP = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_volume",
]


class MissingArchive(Exception):
    pass


def get(url: str, *, retries: int = 4, timeout: int = 30, params=None) -> requests.Response:
    for attempt in range(retries):
        try:
            response = requests.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "ai-trader/0.1 (public research)"},
            )
            if response.status_code == 404:
                raise MissingArchive(url)
            if response.status_code in (403, 451, 418):
                raise RuntimeError(
                    f"Provider denied access ({response.status_code}) to {url}. "
                    "Use an accessible source; this program does not bypass restrictions."
                )
            if response.status_code == 429 or response.status_code >= 500:
                if attempt == retries - 1:
                    response.raise_for_status()
                try:
                    delay = float(response.headers.get("Retry-After", 2**attempt))
                except ValueError:
                    delay = 2**attempt
                # Never retry sooner than the provider's requested delay.
                log.warning("HTTP %s; retrying in %.1fs", response.status_code, delay)
                time.sleep(max(0, delay))
                continue
            response.raise_for_status()
            return response
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("HTTP retries exhausted")


def normalize(raw: pd.DataFrame, seconds: int) -> pd.DataFrame:
    if raw.empty:
        raise ValueError("Empty candle data")
    raw = raw.copy()
    # Some archive variants contain a header; only discard an actual header row.
    raw = raw.loc[~raw["timestamp"].astype(str).isin(["open_time", "Open time", "timestamp"])]
    for col in KEEP:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
    stamps = raw["timestamp"].astype("int64")
    # Spot switched milliseconds -> microseconds on 2025-01-01.
    millis = stamps.where(stamps < 100_000_000_000_000, stamps // 1000)
    raw["timestamp"] = pd.to_datetime(millis, unit="ms", utc=True)
    raw = raw[KEEP].sort_values("timestamp").reset_index(drop=True)
    if raw["timestamp"].duplicated().any():
        raise ValueError("Duplicate candle timestamps")
    nums = raw[KEEP[1:]].to_numpy(dtype=float)
    if not np.isfinite(nums).all():
        raise ValueError("Nonfinite candle values")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Nonpositive OHLC price")
    if (raw[["volume", "quote_volume", "trades", "taker_volume"]] < 0).any().any():
        raise ValueError("Negative volume/trades")
    if (raw.high < raw[["open", "close", "low"]].max(axis=1)).any() or (
        raw.low > raw[["open", "close", "high"]].min(axis=1)
    ).any():
        raise ValueError("Inconsistent OHLC values")
    if ((millis % (seconds * 1000)) != 0).any():
        raise ValueError("Candle timestamps do not match the configured interval")
    return raw


def parse_zip(content: bytes, seconds: int) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        names = [n for n in archive.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError("Expected exactly one CSV per archive")
        with archive.open(names[0]) as f:
            raw = pd.read_csv(f, names=COLUMNS, dtype=str)
    return normalize(raw, seconds)


def periods(start: date, end: date):
    cursor = start.replace(day=1)
    while cursor < end:
        following = (cursor.replace(day=28) + timedelta(days=4)).replace(day=1)
        yield cursor, following
        cursor = following


def archive_file(cfg: Config, symbol: str, kind: str, stamp: str) -> tuple[pd.DataFrame, dict]:
    relative = (
        f"{kind}/klines/{symbol}/{cfg.data.interval}/{symbol}-{cfg.data.interval}-{stamp}.zip"
    )
    url = f"{ARCHIVE}/{relative}"
    path = Path(cfg.data.directory) / "raw" / relative
    checksum_path = path.with_suffix(".zip.CHECKSUM")
    cached = path.exists() and checksum_path.exists() and not cfg.data.refresh
    if cached:
        digest = checksum_path.read_text().split()[0]
        cached = sha256(path) == digest
    if not cached:
        options = {"retries": cfg.data.retries, "timeout": cfg.data.timeout_seconds}
        content = get(url, **options).content
        checksum = get(url + ".CHECKSUM", **options).text
        digest = checksum.split()[0]
        if len(digest) != 64 or hashlib.sha256(content).hexdigest() != digest:
            raise ValueError(f"SHA256 mismatch: {url}")
        atomic_bytes(path, content)
        atomic_bytes(checksum_path, checksum.encode())
    frame = parse_zip(path.read_bytes(), cfg.seconds)
    return frame, {"url": url, "sha256": digest, "rows": len(frame), "cached": cached}


def download_month(
    cfg: Config, symbol: str, begin: date, following: date, start: date, end: date
) -> tuple[list[pd.DataFrame], list[dict]]:
    today = datetime.now(timezone.utc).date()
    if following <= today:
        try:
            frame, info = archive_file(cfg, symbol, "monthly", begin.strftime("%Y-%m"))
            return [frame], [info]
        except MissingArchive:
            log.info("Monthly archive absent for %s %s; checking daily files", symbol, begin)
    frames, records = [], []
    day = max(begin, start)
    while day < min(following, end):
        try:
            frame, info = archive_file(cfg, symbol, "daily", day.isoformat())
            frames.append(frame)
            records.append(info)
        except MissingArchive:
            records.append({"symbol": symbol, "missing_day": day.isoformat()})
            log.warning("Missing archive: %s %s", symbol, day)
        day += timedelta(days=1)
    return frames, records


def coverage(frame: pd.DataFrame, cfg: Config, start: date, end: date) -> dict:
    expected = int((end - start).total_seconds() / cfg.seconds)
    diffs = frame.timestamp.diff().dt.total_seconds()
    return {
        "rows": len(frame),
        "expected_rows": expected,
        "coverage_pct": 100 * len(frame) / expected,
        "first": frame.timestamp.iloc[0].isoformat(),
        "last": frame.timestamp.iloc[-1].isoformat(),
        "gap_count": int((diffs > cfg.seconds).sum()),
        "missing_candles": expected - len(frame),
    }


def download(cfg: Config) -> dict:
    start = date.fromisoformat(cfg.data.start)
    today = datetime.now(timezone.utc).date()
    end = date.fromisoformat(cfg.data.end) if cfg.data.end else today
    if not start < end <= today:
        raise ValueError("Historical range must be start < end <= today UTC (end exclusive)")
    tasks = [(s, b, f) for s in cfg.data.symbols for b, f in periods(start, end)]
    results = {s: [] for s in cfg.data.symbols}
    manifest = {
        "source": ARCHIVE,
        "interval": cfg.data.interval,
        "start": str(start),
        "end_exclusive": str(end),
        "archives": [],
        "symbols": {},
    }
    log.info("Downloading %d symbol-months using %d workers", len(tasks), cfg.data.workers)
    with ThreadPoolExecutor(max_workers=cfg.data.workers) as pool:
        futures = {
            pool.submit(download_month, cfg, s, b, f, start, end): (s, b) for s, b, f in tasks
        }
        for n, future in enumerate(as_completed(futures), 1):
            s, begin = futures[future]
            frames, records = future.result()  # Any corrupt/network failure fails the run.
            results[s].extend(frames)
            manifest["archives"].extend(records)
            log.info(
                "Download %d/%d: %s %s (%s candles)",
                n,
                len(tasks),
                s,
                begin,
                f"{sum(len(f) for f in frames):,}",
            )
    for symbol, frames in results.items():
        if not frames:
            raise ValueError(f"No historical data for {symbol}")
        frame = pd.concat(frames, ignore_index=True).sort_values("timestamp")
        frame = frame[
            (frame.timestamp >= pd.Timestamp(start, tz="UTC"))
            & (frame.timestamp < pd.Timestamp(end, tz="UTC"))
        ].reset_index(drop=True)
        if frame.empty or frame.timestamp.duplicated().any():
            raise ValueError(f"Empty/duplicate merged history: {symbol}")
        info = coverage(frame, cfg, start, end)
        if info["coverage_pct"] < 95:
            raise ValueError(
                f"{symbol}: only {info['coverage_pct']:.1f}% coverage. "
                "Choose dates after listing or repair missing archives."
            )
        if info["missing_candles"]:
            log.warning(
                "%s: %d missing candles; features/labels will reset at gaps",
                symbol,
                info["missing_candles"],
            )
        path = Path(cfg.data.directory) / "processed" / f"{symbol}-{cfg.data.interval}.parquet"
        atomic_bytes(path, frame.to_parquet(index=False))
        info["sha256"] = sha256(path)
        manifest["symbols"][symbol] = info
    write_json(Path(cfg.data.directory) / "manifest.json", manifest)
    return manifest


def load_history(cfg: Config, symbol: str) -> pd.DataFrame:
    path = Path(cfg.data.directory) / "processed" / f"{symbol}-{cfg.data.interval}.parquet"
    if not path.exists():
        raise ValueError(f"Missing {path}; run ai-trader download first")
    frame = pd.read_parquet(path)
    frame = frame[frame.timestamp >= pd.Timestamp(cfg.data.start, tz="UTC")]
    if cfg.data.end:
        frame = frame[frame.timestamp < pd.Timestamp(cfg.data.end, tz="UTC")]
    if (
        frame.empty
        or frame.timestamp.duplicated().any()
        or not frame.timestamp.is_monotonic_increasing
    ):
        raise ValueError(f"Invalid history in {path}; download again")
    return frame.reset_index(drop=True)


def live_candles(symbol: str, cfg: Config, now_ms: int) -> pd.DataFrame:
    payload = get(
        API + "/api/v3/klines",
        params={"symbol": symbol, "interval": cfg.data.interval, "limit": 1000},
    ).json()
    frame = normalize(pd.DataFrame(payload, columns=COLUMNS), cfg.seconds)
    return frame[
        frame.timestamp + pd.Timedelta(seconds=cfg.seconds)
        <= pd.to_datetime(now_ms, unit="ms", utc=True)
    ].reset_index(drop=True)


def book_quote(symbol: str) -> tuple[float, float]:
    payload = get(API + "/api/v3/ticker/bookTicker", params={"symbol": symbol}).json()
    bid, ask = float(payload["bidPrice"]), float(payload["askPrice"])
    if not np.isfinite([bid, ask]).all() or bid <= 0 or ask < bid:
        raise ValueError("Invalid bid/ask quote")
    return bid, ask
