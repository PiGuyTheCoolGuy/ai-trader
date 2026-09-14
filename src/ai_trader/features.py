"""Finite rolling windows; every feature at t uses only candles <= t."""

import numpy as np
import pandas as pd

LOOKBACK = 288
FEATURE_VERSION = 1


def _segment(frame: pd.DataFrame) -> pd.DataFrame:
    c, o, h, lo, v = (
        frame[name].astype(float) for name in ["close", "open", "high", "low", "volume"]
    )
    out = {}
    returns = c.pct_change(fill_method=None)
    for window in [1, 3, 6, 12, 24, 48, 96, 288]:
        out[f"return_{window}"] = c.pct_change(window, fill_method=None)
    for window in [12, 48, 96, 288]:
        mean, std = c.rolling(window).mean(), c.rolling(window).std()
        out[f"sma_distance_{window}"] = c / mean - 1
        out[f"volatility_{window}"] = returns.rolling(window).std()
        out[f"price_z_{window}"] = (c - mean) / std.replace(0, np.nan)
        out[f"range_position_{window}"] = (c - lo.rolling(window).min()) / (
            h.rolling(window).max() - lo.rolling(window).min()
        ).replace(0, np.nan)
        out[f"volume_ratio_{window}"] = v / v.rolling(window).mean().replace(0, np.nan)
    tr = pd.concat([h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1).max(axis=1)
    out["atr_14"] = tr.rolling(14).mean() / c
    delta = c.diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    out["rsi_14"] = (up / (up + down).replace(0, np.nan)).fillna(0.5)
    out["body"] = (c - o) / o
    out["range"] = (h - lo) / c
    out["upper_wick"] = (h - pd.concat([o, c], axis=1).max(axis=1)) / c
    out["lower_wick"] = (pd.concat([o, c], axis=1).min(axis=1) - lo) / c
    out["taker_fraction"] = (frame.taker_volume / v.replace(0, np.nan)).fillna(0.5)
    out["trade_ratio_48"] = frame.trades / frame.trades.rolling(48).mean().replace(0, np.nan)
    minutes = frame.timestamp.dt.hour * 60 + frame.timestamp.dt.minute
    out["time_sin"] = np.sin(2 * np.pi * minutes / 1440)
    out["time_cos"] = np.cos(2 * np.pi * minutes / 1440)
    out["week_sin"] = np.sin(2 * np.pi * frame.timestamp.dt.dayofweek / 7)
    out["week_cos"] = np.cos(2 * np.pi * frame.timestamp.dt.dayofweek / 7)
    features = pd.DataFrame(out, index=frame.index).replace([np.inf, -np.inf], np.nan)
    # Constant-price/zero-volume windows have neutral ratios, after warmup only.
    features.iloc[LOOKBACK:] = features.iloc[LOOKBACK:].fillna(0)
    features.iloc[:LOOKBACK] = np.nan
    return features.astype("float32")


def build_features(frame: pd.DataFrame, seconds: int) -> pd.DataFrame:
    gaps = frame.timestamp.diff().dt.total_seconds().ne(seconds).cumsum()
    return pd.concat([_segment(group) for _, group in frame.groupby(gaps, sort=False)]).sort_index()


def labeled(frame: pd.DataFrame, seconds: int, horizon: int) -> tuple[pd.DataFrame, list[str]]:
    features = build_features(frame, seconds)
    result = pd.concat([frame, features], axis=1)
    # Signal at close[t], simulated entry at open[t+1], horizon exit at open[t+1+h].
    result["target_bps"] = (frame.open.shift(-(horizon + 1)) / frame.open.shift(-1) - 1) * 10000
    result["label_end"] = frame.timestamp.shift(-(horizon + 1))
    continuous = (result.label_end - frame.timestamp).dt.total_seconds().eq((horizon + 1) * seconds)
    result.loc[~continuous, ["target_bps", "label_end"]] = [np.nan, pd.NaT]
    return result.dropna(subset=[*features.columns, "target_bps", "label_end"]), list(
        features.columns
    )


def split_times(timestamps: pd.Series, fractions: list[float]) -> list[pd.Timestamp]:
    times = pd.Series(timestamps.unique()).sort_values().reset_index(drop=True)
    indexes = [int(len(times) * v) for v in np.cumsum(fractions)]
    if len(set(indexes)) != 3 or not indexes or indexes[-1] >= len(times):
        raise ValueError("Not enough distinct timestamps for four chronological splits")
    return [times.iloc[i] for i in indexes]


def purged_split(data: pd.DataFrame, boundaries: list[pd.Timestamp]) -> dict[str, pd.DataFrame]:
    a, b, c = boundaries
    # label_end must be STRICTLY before the next split; no future target crosses a boundary.
    return {
        "train": data[(data.timestamp < a) & (data.label_end < a)],
        "tune": data[(data.timestamp >= a) & (data.timestamp < b) & (data.label_end < b)],
        "calibration": data[(data.timestamp >= b) & (data.timestamp < c) & (data.label_end < c)],
        "test": data[data.timestamp >= c],
    }
