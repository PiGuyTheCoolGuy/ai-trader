# Measured CPU reference run

This directory contains an actual trained model and its results, produced while building the project. It is **not** a profitable-strategy example. The selected policy is cash because the calibration strategy lost money after costs.

| Measurement | Result |
| --- | --- |
| Source | Binance public spot archives, SHA256-verified |
| Range | 2021-01-01 through 2026-09-13 UTC |
| Markets | BTCUSDT, ETHUSDT, SOLUSDT |
| Interval | 5 minutes |
| Downloaded candles | 1,798,209 |
| Coverage per symbol | 99.9645%; 213 missing candles in 7 gaps |
| Labeled training rows | 985,002 |
| Tuning / calibration / test labeled rows | 268,608 / 268,608 / 268,650 |
| Features | 40 |
| Device | CPU, 6 threads; no GPU |
| Selected model | 42 fitted trees; 3 candidates evaluated |
| Pipeline time after download | 78.27 seconds, including features and evaluation |
| Peak process RSS | 2,493,492 KiB (about 2.38 GiB), measured on Linux |
| Calibration at 31 bps entry threshold | 122 closed trades; -1.8113% net return; 3.5774% maximum drawdown |
| Selected policy | Cash (`threshold_bps: null`) |
| Frozen-policy test | 0 trades, 0% return |

The machine running this verification is not the user's Dell server. Its timing is a measurement on this environment, not an estimate for the R620. Downloads are excluded from the training duration. Early stopping chose 84, 42 and 49 trees respectively from limits of 2,000 rounds per trial; the candidate with the lowest tuning RMSE was saved.

The model improved return RMSE slightly relative to a constant forecast but did not establish a tradable edge after costs. The 0% final policy result reflects remaining in cash, not successful market prediction. Most estimated returns were smaller than the cost threshold: the calibration 99th percentile prediction was approximately 5.44 bps, versus the 31 bps minimum entry forecast. Rare larger forecasts produced the 122 calibration trades.

`report.html` and `training_report.json` contain the complete results; `model.txt` is the native model; `data_manifest.json` identifies every archive and processed data hash. `test_equity.csv` includes every held-out candle and the buy-and-hold baseline. `test_trades.csv` has a stable header even though the selected policy made no test trades. The raw historical dataset is downloaded locally by the program and is not committed to Git.

From the repository root, inspect or run this model:

```bash
ai-trader report --run examples/reference-run
ai-trader paper --run examples/reference-run --state artifacts/reference-paper.sqlite --once
```

It will observe the feed without entering positions because its saved threshold is null. Train your own new run with the same historical range:

```bash
ai-trader run --config configs/default.json --threads 6 --end 2026-09-14
```

Versions are recorded in `training_report.json`. Hardware, library versions, and upstream archive corrections can affect reproducibility. Do not change the saved configuration or threshold and present it as this evaluated model; new experiments should produce their own run directories.
