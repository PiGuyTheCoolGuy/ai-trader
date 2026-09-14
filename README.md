# AI Trader

A complete Python project for **CPU-only crypto model training, historical backtesting, and forward paper trading**. It downloads years of public candles, fits a LightGBM model, evaluates it on later unseen data, and records paper trades using fresh market quotes. No GPU, exchange account, or API keys are needed. There is no real-money order submission code.

The default dataset is BTCUSDT, ETHUSDT and SOLUSDT, at five-minute intervals, from **January 2021 through yesterday UTC**: roughly 1.8 million candles as of September 2026. Each candle contributes market features; the model predicts a future return, not a text response. This is gradient-boosted machine learning, not an LLM or reinforcement-learning agent.

An [actual CPU reference run](examples/reference-run/README.md), including the trained model and full report, is included. It completed training and evaluation on 1,798,209 downloaded candles in 78 seconds after download on six CPU threads. Its calibration strategy lost 1.81% after costs, so it selected cash. This is a working research pipeline, not a demonstrated profitable trader.

## Start on your Ubuntu server

Python 3.10–3.12 is supported and tested by CI. Ubuntu 22.04's Python 3.10 works.

```bash
sudo apt update
sudo apt install -y git python3-venv libgomp1
git clone https://github.com/PiGuyTheCoolGuy/ai-trader.git
cd ai-trader
bash scripts/setup.sh
source .venv/bin/activate
ai-trader run --config configs/default.json --threads 40
```

Use a thread count appropriate to your CPU, or `--threads 0` to use all detected threads. On a dual-socket server, more threads are not automatically faster; try the physical-core count before assuming every logical thread is best. Training always sets LightGBM's `device_type=cpu`.

The last command downloads data, builds features, trains three model candidates, chooses a trading threshold on calibration data, runs the untouched test, and writes a report. It can take a while; the download logs each completed symbol-month, and training logs each trial and every 50 boosting rounds. The raw archive cache survives interruptions. Re-running skips checksum-valid downloads.

For a long server session with a persistent log:

```bash
tmux new -s ai-trader
bash scripts/train.sh configs/default.json 40
```

Detach with Ctrl+B, then D; reconnect with `tmux attach -t ai-trader`. The script creates a timestamped log under `artifacts/logs/`.

## Commands

Run from the repository root with the virtual environment active.

| Command | Purpose |
| --- | --- |
| `ai-trader run --config configs/default.json` | Full download, training and evaluation pipeline |
| `ai-trader download --config configs/default.json` | Download/cache/validate historical data only |
| `ai-trader train --config configs/default.json --threads 40` | Train on the existing downloaded dataset |
| `ai-trader run --start 2024-01-01 --end 2025-01-01` | Reproducible date range; end is exclusive |
| `ai-trader download --refresh` | Replace cached archives with current provider versions |
| `ai-trader report --run artifacts/RUN_ID` | Print training proof and final test results |
| `ai-trader paper --run artifacts/RUN_ID --once` | Verify the live feed and initialize paper state |
| `ai-trader paper --run artifacts/RUN_ID` | Run forward paper trading until stopped |
| `ai-trader status` | Show saved balance, open positions and recent paper trades |

Each training run has its own directory; the CLI prints its exact path. `artifacts/latest.json` also identifies the latest successful run. Failed runs never replace this pointer. Re-running download with a different date range replaces the processed Parquet files for those symbols/intervals; raw ZIP caches remain reusable. Keep a separate `data.directory` for experiments that need concurrent datasets.

## More historical data

`configs/large.json` changes the dataset to **five symbols at one-minute resolution from January 2021**, with six CPU model trials and a 60-minute prediction horizon. As of September 2026, that is roughly 15 million candles before gaps. It uses substantially more memory, disk and CPU time. Start with the default configuration to measure your server first.

```bash
ai-trader run --config configs/large.json --threads 40
```

You can edit dates, symbols, interval, workers, tree rounds, thread count and risk settings in JSON. Supported intervals: `1m`, `3m`, `5m`, `15m`, `30m`, `1h`. All symbols must be USDT spot pairs. A symbol must have data over at least 95% of the requested range; use a later start for a newly listed asset. `end: null` means today UTC, excluding today's incomplete daily archive.

Increasing `rounds` increases the maximum permitted training, not the minimum. Early stopping ends a trial when tuning error stops improving. More time and more candles do not guarantee a better strategy.

## What the model learns

The pooled model learns from all configured symbols using price-normalized features: returns, moving-average distances, volatility, price ranges, volume changes, taker-buy share, trade counts, RSI, ATR, candle shapes and cyclic UTC time features. Rolling features use at most 288 previous bars. They reset at missing-candle gaps and use the same implementation in training and live paper execution.

At the close of candle `t`, the target is the gross percentage return from `open[t+1]` to `open[t+1+horizon]`, expressed in **basis points**. One basis point is 0.01%; 100 basis points is 1%. A predicted `35` means an estimated 0.35% gross return, not 35% confidence.

The time split uses shared timestamps across every symbol:

| Period | Default share | Used for |
| --- | --- | --- |
| Training | First 55% | Fit trees |
| Tuning | Next 15% | Early stopping and model selection by return RMSE |
| Calibration | Next 15% | Select a trading threshold using simulated results after costs |
| Test | Final 15% | Evaluate the frozen model and threshold |

Labels whose future endpoint reaches the next period are purged. Features use only present/past candles; there is no shuffled train/test split and no model refit after examining the test results. This means the saved model is fitted to the training portion, **not to every downloaded row**: later rows are deliberately reserved for evaluation. The evaluation is a fixed chronological holdout, not a walk-forward retraining claim. If you change settings after looking at a test result, that test is no longer untouched evidence; assess future data as well.

## Reports and the zero-trade case

Open `artifacts/RUN_ID/report.html` in your browser. It contains an equity chart, per-symbol and monthly test results, chronological split sizes, each candidate's fitted tree count and training time, and the calibration decision table. No web server is required.

| Output | Contents |
| --- | --- |
| `model.txt` | Native LightGBM model; no pickle loading |
| `metadata.json`, `config.json` | Feature schema, threshold, model hash, exact settings |
| `training_report.json` | Split dates/counts, predictions, learning curves, baseline errors, candidate scores, versions, data hashes and final metrics |
| `test_trades.csv` | Entries, exits, sizes, fees, net P&L and reasons |
| `test_equity.csv` | Held-out strategy and equal-allocation buy-and-hold equity |
| `report.html` | Portable visual report |
| `data/manifest.json` | Source URLs, SHA256 checksums, row counts and coverage |

**Training succeeded and trading succeeded are different things.** `training_completed`, `chosen_trees`, per-trial learning curves and elapsed times prove a fit occurred. A short fit can be normal for tabular CPU models, especially when early stopping triggers.

The minimum entry forecast is `2 × (fee_bps + slippage_bps) + min_edge_bps`, or 31 bps with default settings. Calibration considers this floor plus prediction quantiles. A threshold needs at least 30 closed calibration trades, positive net return, no drawdown halt, and a positive `return_pct - 0.5 × max_drawdown_pct` score. Cash is an explicit alternative. If none qualify, `threshold_bps` is `null`, the report explains why, and paper trading observes the market without opening positions. The software never lowers the threshold automatically just to manufacture trades. Defaults are research assumptions, not an optimized strategy or a profitability promise.

## Execution and risk model

- Long-only spot positions, with no leverage or shorting. The 10,000 USDT starting balance is **total**, divided equally into independent symbol allocations. Capital is not reused across symbols.
- A position uses at most 25% of its symbol allocation and is additionally sized to a 0.5% stop-risk budget including estimated costs. Minimum notional is 10 USDT.
- Default stops are 1.5% below entry, targets 3% above, and maximum holding time is 12 five-minute bars. Positions close at the first bar/quote on a new UTC day. Stops and sampled circuit breakers can lose more than their configured threshold after gaps or slippage.
- Each symbol allocation has a 3% daily loss pause and a persistent 15% peak-to-equity drawdown halt. These are allocation-level limits; there is no separate shared portfolio circuit breaker.
- Backtests use only the preceding closed-candle signal, fill at the next candle open, charge fees/slippage on both sides, choose the stop if both stop and target are touched in one bar, and account for forced liquidation at the evaluation endpoint. Stops crossed by an opening gap fill at the worse opening price. Risk limits are checked at bar open/close.
- The backtest's slippage estimate includes an assumed spread; it has no order-book, queue, market-impact or historical execution-latency model. Buy-and-hold comparison includes estimated entry/exit costs and uses the same capital allocation.

## Forward paper trading

```bash
bash scripts/paper.sh --once
bash scripts/paper.sh
# In another terminal:
source .venv/bin/activate
ai-trader status
```

The script chooses the latest successful model. Paper execution buys at the newly observed ask and sells at the observed bid, plus configured adverse slippage and fees. It does not claim historical candle-open fills after those opens are gone. Initial startup/restart only primes signals; entries begin with the next fresh closed candle, no more than 60 seconds old by default. Positions are checked on each successful polling cycle, so an outage can delay a stop or exit. Quotes are fetched sequentially across symbols.

SQLite atomically saves cash, positions, risk halts, processed signal identity and events. A file lock prevents two processes from trading the same state. A state file is bound to a model/config fingerprint; use a new `--state` path when switching models. `status` reports timestamps of the last observations so an old balance is not mistaken for a fresh quote. Ctrl+C stops the process without discarding open paper positions.

`scripts/ai-trader-paper.service.example` is an optional systemd template for your server. Edit its paths and run ID, install it as `/etc/systemd/system/ai-trader-paper.service`, then enable it with systemctl. Logs are available with `journalctl -u ai-trader-paper -f`.

## Development

```bash
python -m pip install -e '.[dev]'
ruff check src tests
ruff format --check src tests
pytest -q
```

Tests cover timestamp-unit conversion, archive integrity, daily fallback and rate limits, data clipping, causal features, gap handling, purge boundaries, actual CPU fitting/model reload, next-open fills, conservative stop ordering, costs, risk resets, shared capital allocation and paper persistence. Tests use generated fixtures and mocked downloads; they require no API calls. CI runs them on Python 3.10 and 3.12.

## Primary source documentation

- [Binance public archives](https://github.com/binance/binance-public-data): monthly/daily layout, checksums, and the January 2025 change from millisecond to microsecond spot timestamps.
- [Binance public market-data endpoints](https://github.com/binance/binance-spot-api-docs/blob/master/faqs/market_data_only.md): unauthenticated market data, without private/order APIs.
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html): CPU device selection, threading and early stopping.

If a provider denies access, the downloader reports the denial and stops. It does not route around geographic or network restrictions.
