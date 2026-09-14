import pytest

from ai_trader.broker import Account
from ai_trader.paper import PaperStore, act, status


def test_no_startup_or_stale_signal_trade(config):
    account = Account.create(1000)
    assert act(account, 99, 100, 1704067200, config, 100, 10, False) == []
    assert account.position is None


def test_live_entry_uses_ask_and_exit_uses_observed_bid(config):
    account = Account.create(1000)
    events = act(account, 99, 100, 1704067200, config, 100, 10, True)
    assert events[0]["price"] == 100
    events = act(account, 90, 91, 1704067210, config, None, 10, False)
    assert events[0]["exit_price"] == 90
    assert events[0]["reason"] == "stop_loss"


def test_sqlite_atomic_resume_and_model_identity(tmp_path, config):
    path = tmp_path / "paper.sqlite"
    store = PaperStore(path, "model-a", config)
    account = Account(**store.state["accounts"]["BTCUSDT"])
    events = act(account, 99, 100, 1704067200, config, 100, 10, True)
    store.state["accounts"]["BTCUSDT"] = account.to_dict()
    store.save(events)
    store.close()
    store = PaperStore(path, "model-a", config)
    resumed = Account(**store.state["accounts"]["BTCUSDT"])
    assert resumed.position["entry_price"] == 100
    assert act(resumed, 99, 100, 1704067210, config, 100, 10, True) == []
    store.close()
    assert status(path)["event_count"] == 1
    with pytest.raises(ValueError, match="different model"):
        PaperStore(path, "model-b", config)


def test_no_network_needed_to_read_paper_status(tmp_path, config, monkeypatch):
    path = tmp_path / "paper.sqlite"
    store = PaperStore(path, "model", config)
    store.close()
    monkeypatch.setattr(
        "ai_trader.paper.get", lambda *a, **k: pytest.fail("Unexpected network access")
    )
    assert status(path)["initial_equity"] == 10000


@pytest.mark.parametrize("unavailable", [False, True])
def test_missing_or_old_signals_still_manage_positions(
    tmp_path, config, monkeypatch, history, unavailable
):
    import hashlib
    import json

    from ai_trader.features import build_features
    from ai_trader.paper import run_paper

    class Model:
        def predict(self, *a, **k):
            return [100]

    meta = {"features": list(build_features(history, 300).columns), "threshold_bps": 10}
    fingerprint = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    path = tmp_path / "paper.sqlite"
    store = PaperStore(path, fingerprint, config)
    account = Account.create(1000)
    now = history.timestamp.iloc[-1].timestamp() + 2 * config.seconds
    account.enter(100, now - 600, now + 600, config.risk, 100)
    store.state["accounts"]["BTCUSDT"] = account.to_dict()
    store.save([])
    store.close()

    class Response:
        def json(self):
            return {"serverTime": int(now * 1000 + 1000)}

    def fake_candles(*args):
        if unavailable:
            raise ValueError("Candles unavailable")
        return history

    monkeypatch.setattr("ai_trader.paper.load_run", lambda *a: (Model(), meta, config))
    monkeypatch.setattr("ai_trader.paper.get", lambda *a: Response())
    monkeypatch.setattr("ai_trader.paper.live_candles", fake_candles)
    monkeypatch.setattr("ai_trader.paper.book_quote", lambda *a: (90, 91))
    run_paper(tmp_path, path, once=True)
    state = status(path)
    assert state["accounts"]["BTCUSDT"]["position"] is None
    assert state["recent_events"][0]["reason"] == "stop_loss"
    assert state["quotes"]["BTCUSDT"]["signal_status"] == (
        "unavailable" if unavailable else "stale"
    )
