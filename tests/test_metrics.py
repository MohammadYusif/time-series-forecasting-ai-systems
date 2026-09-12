import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))

from metrics import coverage, interval_width, mae, mase, mape, pinball_loss, rmse, smape, wape


def test_mae_perfect_forecast_is_zero():
    y = [1, 2, 3, 4]
    assert mae(y, y) == 0.0


def test_mae_matches_hand_computation():
    assert mae([1, 2, 3], [2, 2, 2]) == pytest.approx(2 / 3)


def test_rmse_penalises_large_errors_harder_than_mae():
    y_true = [0, 0, 0, 0]
    y_pred_uniform = [1, 1, 1, 1]
    y_pred_spiky = [0, 0, 0, 4]
    # Same MAE (1.0) for both, RMSE must be higher for the spiky one.
    assert mae(y_true, y_pred_uniform) == pytest.approx(mae(y_true, y_pred_spiky))
    assert rmse(y_true, y_pred_spiky) > rmse(y_true, y_pred_uniform)


def test_mape_matches_hand_computation():
    assert mape([100, 200], [110, 180]) == pytest.approx((10 / 100 + 20 / 200) / 2 * 100)


def test_smape_is_symmetric_in_over_and_under_forecast():
    over = smape([100], [120])
    under = smape([120], [100])
    assert over == pytest.approx(under)


def test_smape_bounded_near_200_for_opposite_sign_extreme():
    assert smape([100], [-100]) == pytest.approx(200.0, abs=1e-6)


def test_wape_handles_zero_actuals_without_exploding():
    # A single near-zero actual should not dominate the score the way it
    # would under row-wise MAPE.
    y_true = [1000, 1000, 0]
    y_pred = [1000, 1000, 5]
    assert wape(y_true, y_pred) == pytest.approx(5 / 2000 * 100)


def test_wape_all_zero_actuals_and_zero_prediction_is_zero():
    assert wape([0, 0, 0], [0, 0, 0]) == 0.0


def test_mase_below_one_beats_naive_seasonal_baseline():
    rng = np.random.default_rng(0)
    y_train = 10 + 2 * np.sin(np.arange(200) * 2 * np.pi / 7) + rng.normal(0, 0.1, 200)
    y_true = 10 + 2 * np.sin(np.arange(200, 214) * 2 * np.pi / 7)
    y_pred = y_true + rng.normal(0, 0.05, 14)  # a near-perfect forecast
    score = mase(y_true, y_pred, y_train, seasonal_period=7)
    assert score < 1.0


def test_mase_naive_seasonal_forecast_scores_close_to_one():
    rng = np.random.default_rng(1)
    period = 7
    y_train = 10 + 2 * np.sin(np.arange(300) * 2 * np.pi / period) + rng.normal(0, 0.3, 300)
    y_true = y_train[-14:]
    y_pred_naive = y_train[-14 - period : -period]  # repeat last full cycle
    score = mase(y_true, y_pred_naive, y_train[:-14], seasonal_period=period)
    assert score == pytest.approx(1.0, rel=0.5)


def test_mase_raises_on_perfectly_flat_training_series():
    y_train = np.ones(50)
    with pytest.raises(ValueError):
        mase([1, 1], [1, 1], y_train, seasonal_period=1)


def test_pinball_loss_zero_for_perfect_quantile_hit():
    assert pinball_loss([10, 20], [10, 20], quantile=0.5) == pytest.approx(0.0)


def test_pinball_loss_penalises_underprediction_more_for_high_quantile():
    y_true = [10]
    under = pinball_loss(y_true, [5], quantile=0.9)
    over = pinball_loss(y_true, [15], quantile=0.9)
    assert under > over


def test_pinball_loss_rejects_out_of_range_quantile():
    with pytest.raises(ValueError):
        pinball_loss([1], [1], quantile=1.5)


def test_coverage_all_inside_interval_is_one():
    y_true = [5, 6, 7]
    assert coverage(y_true, [0, 0, 0], [10, 10, 10]) == 1.0


def test_coverage_half_inside_interval_is_half():
    y_true = [5, 15]
    assert coverage(y_true, [0, 0], [10, 10]) == pytest.approx(0.5)


def test_interval_width_matches_hand_computation():
    assert interval_width([0, 2], [4, 10]) == pytest.approx((4 + 8) / 2)
