import numpy as np
import pandas as pd

from ai_trader.features import LOOKBACK, build_features, labeled, purged_split, split_times


def test_future_candles_cannot_change_past_features(history):
    expected = build_features(history, 300)
    altered = history.copy()
    altered.loc[900:, ["open", "high", "low", "close", "volume"]] *= 10
    actual = build_features(altered, 300)
    pd.testing.assert_frame_equal(expected.iloc[:900], actual.iloc[:900])


def test_live_window_and_offline_features_match(history):
    expected = build_features(history, 300).iloc[-1]
    live = build_features(history.iloc[-1000:].reset_index(drop=True), 300).iloc[-1]
    pd.testing.assert_series_equal(expected, live, check_names=False)


def test_gaps_reset_features_and_invalidate_labels(history):
    gapped = history.drop(index=800).reset_index(drop=True)
    features = build_features(gapped, 300)
    assert features.iloc[800 : 800 + LOOKBACK].isna().all().all()
    result, _ = labeled(gapped, 300, 12)
    before = result[result.timestamp < history.timestamp.iloc[800]]
    assert before.label_end.max() < history.timestamp.iloc[800]


def test_label_matches_next_open_execution(history):
    result, _ = labeled(history, 300, 12)
    i = result.index[0]
    assert np.isclose(
        result.target_bps.iloc[0],
        (history.open.iloc[i + 13] / history.open.iloc[i + 1] - 1) * 10000,
    )


def test_shared_time_boundaries_and_purge(history):
    result, _ = labeled(history, 300, 12)
    boundaries = split_times(result.timestamp, [0.55, 0.15, 0.15])
    split = purged_split(result, boundaries)
    for earlier, later in zip(["train", "tune", "calibration"], ["tune", "calibration", "test"]):
        assert split[earlier].label_end.max() < split[later].timestamp.min()
