import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from backtest import (
    expanding_window_splits,
    rolling_window_splits,
    run_backtest,
    seasonal_naive_forecast,
)


def test_expanding_splits_last_fold_ends_exactly_at_n():
    splits = expanding_window_splits(n=100, n_folds=5, horizon=10, min_train_size=30)
    assert splits[-1][1].stop == 100


def test_expanding_splits_train_only_grows():
    splits = expanding_window_splits(n=100, n_folds=5, horizon=10, min_train_size=30)
    train_ends = [tr.stop for tr, _ in splits]
    assert train_ends == sorted(train_ends)
    assert len(set(train_ends)) == len(train_ends)  # strictly increasing


def test_expanding_splits_test_windows_never_overlap_and_chain_to_train():
    splits = expanding_window_splits(n=100, n_folds=5, horizon=10, min_train_size=30)
    for train_slice, test_slice in splits:
        assert train_slice.stop == test_slice.start  # no gap, no overlap
    test_windows = [(te.start, te.stop) for _, te in splits]
    for (s1, e1), (s2, e2) in zip(test_windows, test_windows[1:]):
        assert e1 == s2  # folds tile the holdout region with no overlap/gap


def test_expanding_splits_raises_when_not_enough_data():
    with pytest.raises(ValueError):
        expanding_window_splits(n=50, n_folds=5, horizon=10, min_train_size=30)


def test_rolling_splits_train_window_is_constant_size():
    splits = rolling_window_splits(n=100, n_folds=4, horizon=10, train_size=20)
    sizes = {tr.stop - tr.start for tr, _ in splits}
    assert sizes == {20}


def test_rolling_splits_last_fold_ends_exactly_at_n():
    splits = rolling_window_splits(n=100, n_folds=4, horizon=10, train_size=20)
    assert splits[-1][1].stop == 100


def test_run_backtest_calls_model_once_per_fold_with_correct_train_only():
    y = np.arange(100, dtype=float)
    splits = expanding_window_splits(n=100, n_folds=3, horizon=5, min_train_size=50)

    seen_train_lengths = []

    def fit_predict(y_train, horizon):
        seen_train_lengths.append(len(y_train))
        return np.repeat(y_train[-1], horizon)  # naive: repeat last observed value

    results = run_backtest(y, splits, fit_predict)
    assert len(results) == 3
    assert seen_train_lengths == [tr.stop for tr, _ in splits]
    for fold in results:
        # every predicted value equals the last TRAIN value, never a test value —
        # proof the model never saw its own answer.
        assert np.all(fold["y_pred"] == fold["y_train"][-1])
        assert not np.array_equal(fold["y_pred"], fold["y_true"])


def test_run_backtest_rejects_wrong_length_predictions():
    y = np.arange(60, dtype=float)
    splits = expanding_window_splits(n=60, n_folds=2, horizon=5, min_train_size=40)

    def bad_fit_predict(y_train, horizon):
        return np.zeros(horizon - 1)  # deliberately wrong length

    with pytest.raises(ValueError):
        run_backtest(y, splits, bad_fit_predict)


def test_seasonal_naive_repeats_last_cycle():
    y_train = np.array([1, 2, 3, 4, 5, 6, 7], dtype=float)  # one week, period 7
    forecast = seasonal_naive_forecast(y_train, horizon=7, period=7)
    assert np.array_equal(forecast, y_train)


def test_seasonal_naive_wraps_for_longer_horizon():
    y_train = np.array([1, 2, 3], dtype=float)
    forecast = seasonal_naive_forecast(y_train, horizon=7, period=3)
    assert np.array_equal(forecast, np.array([1, 2, 3, 1, 2, 3, 1]))


def test_seasonal_naive_period_one_repeats_last_value():
    y_train = np.array([10, 20, 30], dtype=float)
    forecast = seasonal_naive_forecast(y_train, horizon=4, period=1)
    assert np.array_equal(forecast, np.array([30, 30, 30, 30]))


def test_seasonal_naive_raises_when_train_shorter_than_period():
    with pytest.raises(ValueError):
        seasonal_naive_forecast(np.array([1.0, 2.0]), horizon=5, period=7)
