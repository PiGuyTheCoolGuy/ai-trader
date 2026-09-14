import hashlib
import io
import zipfile

import pandas as pd
import pytest

from ai_trader.data import (
    COLUMNS,
    MissingArchive,
    archive_file,
    download,
    get,
    normalize,
    parse_zip,
)


def archive_bytes(stamp=1704067200000):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as f:
        f.writestr("candle.csv", f"{stamp},100,102,99,101,10,{stamp + 299999},1010,5,6,606,0\n")
    return stream.getvalue()


def test_millisecond_and_microsecond_archives():
    ms = 1735689600000
    left = parse_zip(archive_bytes(ms), 300)
    right = parse_zip(archive_bytes(ms * 1000), 300)
    pd.testing.assert_frame_equal(left, right)
    assert str(left.timestamp.iloc[0]) == "2025-01-01 00:00:00+00:00"


def test_reject_duplicate_and_bad_ohlc():
    row = [1704067200000, 100, 102, 99, 101, 10, 0, 1010, 5, 6, 606, 0]
    with pytest.raises(ValueError, match="Duplicate"):
        normalize(pd.DataFrame([row, row], columns=COLUMNS), 300)
    row[2] = 98
    with pytest.raises(ValueError, match="Inconsistent"):
        normalize(pd.DataFrame([row], columns=COLUMNS), 300)


def test_corruption_is_not_cached_as_valid(tmp_path, monkeypatch, config):
    config.data.directory = str(tmp_path)
    content = archive_bytes()
    digest = hashlib.sha256(content).hexdigest()
    calls = []

    class Response:
        def __init__(self, url):
            self.content = content
            self.text = digest + "  candle.zip"

    def fake_get(url, **kwargs):
        calls.append(url)
        return Response(url)

    monkeypatch.setattr("ai_trader.data.get", fake_get)
    _, info = archive_file(config, "BTCUSDT", "monthly", "2024-01")
    assert not info["cached"]
    _, info = archive_file(config, "BTCUSDT", "monthly", "2024-01")
    assert info["cached"] and len(calls) == 2
    path = next(tmp_path.rglob("*.zip"))
    path.write_bytes(b"corrupted")
    _, info = archive_file(config, "BTCUSDT", "monthly", "2024-01")
    assert not info["cached"] and len(calls) == 4


def test_reject_checksum_mismatch(tmp_path, monkeypatch, config):
    config.data.directory = str(tmp_path)

    class Response:
        content = archive_bytes()
        text = "0" * 64

    monkeypatch.setattr("ai_trader.data.get", lambda *a, **k: Response())
    with pytest.raises(ValueError, match="SHA256"):
        archive_file(config, "BTCUSDT", "monthly", "2024-01")
    assert not list(tmp_path.rglob("*.zip"))


def test_monthly_fallback_daily(tmp_path, monkeypatch, config):
    from datetime import date

    from conftest import candles

    from ai_trader.data import download_month

    calls = []

    def fake_archive(cfg, symbol, kind, stamp):
        calls.append((kind, stamp))
        if kind == "monthly":
            raise MissingArchive(stamp)
        return candles(288, start=stamp), {"url": stamp}

    monkeypatch.setattr("ai_trader.data.archive_file", fake_archive)
    frames, _ = download_month(
        config, "BTCUSDT", date(2024, 1, 1), date(2024, 2, 1), date(2024, 1, 2), date(2024, 1, 4)
    )
    assert len(frames) == 2
    assert calls == [("monthly", "2024-01"), ("daily", "2024-01-02"), ("daily", "2024-01-03")]


def test_rate_limit_respects_retry_after(monkeypatch):
    from requests import Response

    responses = []
    for code in (429, 200):
        r = Response()
        r.status_code = code
        r.headers["Retry-After"] = "3"
        responses.append(r)
    sleeps = []
    monkeypatch.setattr("ai_trader.data.requests.get", lambda *a, **k: responses.pop(0))
    monkeypatch.setattr("ai_trader.data.time.sleep", sleeps.append)
    assert get("https://example.com").status_code == 200
    assert sleeps == [3.0]


def test_download_manifest_and_date_clipping(tmp_path, monkeypatch, config):
    from conftest import candles

    config.data.start, config.data.end = "2024-01-02", "2024-01-04"
    config.data.directory = str(tmp_path)
    monkeypatch.setattr(
        "ai_trader.data.download_month",
        lambda *a: ([candles(31 * 288, start="2024-01-01")], [{"url": "mock", "sha256": "mock"}]),
    )
    manifest = download(config)
    assert manifest["symbols"]["BTCUSDT"]["rows"] == 576
    assert manifest["symbols"]["BTCUSDT"]["coverage_pct"] == 100
