"""Kiểm thử ``src.forecast.tomorrow_consensus`` — hàm thuần, không DB/mạng.

Phủ 3 nhóm yêu cầu:
- danh sách rỗng (và None) → ``{"has_forecast": False}``;
- giá trị None/NaN (model không cho dự báo giá/xác suất) → bị bỏ khỏi mean,
  min, max; hướng None không được đếm;
- đủ dữ liệu horizon=1 → mean/min/max, p_up_mean, n_up/n_down/n_neutral, và
  lọc đúng dòng horizon=5 (không tính vào đồng thuận "phiên mai").
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.forecast import tomorrow_consensus  # noqa: E402


def make_row(
    model: str = "lightgbm",
    horizon: int = 1,
    target_date: str = "2026-10-01",
    predicted_price: float | None = 24.5,
    predicted_direction: str | None = "up",
    probability_up: float | None = 0.6,
) -> dict:
    """Dòng predictions đúng schema trả về bởi forecast_summary (list[dict])."""
    return {
        "model": model,
        "horizon": horizon,
        "target_date": target_date,
        "predicted_price": predicted_price,
        "predicted_direction": predicted_direction,
        "probability_up": probability_up,
        "actual_price": None,
    }


# ---------------------------------------------------------------------------
# 1) Rỗng
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty", [[], None])
def test_empty_predictions_returns_only_has_forecast_false(empty):
    assert tomorrow_consensus(empty) == {"has_forecast": False}


def test_only_other_horizon_is_empty_consensus():
    # Chỉ có dự báo h=5 → không phải "phiên mai" → coi như rỗng.
    rows = [make_row(model="naive", horizon=5), make_row(model="arima", horizon=5)]
    assert tomorrow_consensus(rows) == {"has_forecast": False}


def test_horizon_missing_or_garbage_is_skipped():
    rows = [
        {"model": "x", "horizon": None, "target_date": "2026-10-01"},
        {"model": "y", "horizon": float("nan")},
        make_row(model="lightgbm", horizon=1, predicted_price=25.0),
    ]
    result = tomorrow_consensus(rows)
    assert result["has_forecast"] is True
    assert result["n_models"] == 1


# ---------------------------------------------------------------------------
# 2) None / NaN trong giá trị
# ---------------------------------------------------------------------------


def test_none_prices_and_probabilities_are_excluded():
    rows = [
        make_row(model="naive", predicted_price=24.5, predicted_direction="up", probability_up=None),
        make_row(model="arima", predicted_price=None, predicted_direction="down", probability_up=0.4),
    ]
    result = tomorrow_consensus(rows)
    assert result["has_forecast"] is True
    # price: chỉ naive có giá → mean = min = max = 24.5
    assert result["price_mean"] == pytest.approx(24.5)
    assert result["price_min"] == pytest.approx(24.5)
    assert result["price_max"] == pytest.approx(24.5)
    # p_up: chỉ arima có xác suất → mean = 0.4
    assert result["p_up_mean"] == pytest.approx(0.4)
    assert result["n_up"] == 1
    assert result["n_down"] == 1
    assert result["n_neutral"] == 0
    assert result["n_models"] == 2


def test_all_values_none_still_counts_directionless_models():
    rows = [
        make_row(model="naive", predicted_price=None, predicted_direction=None, probability_up=None),
        make_row(model="arima", predicted_price=None, predicted_direction=None, probability_up=None),
    ]
    result = tomorrow_consensus(rows)
    assert result["has_forecast"] is True
    assert result["n_models"] == 2
    assert result["price_mean"] is None
    assert result["price_min"] is None
    assert result["price_max"] is None
    assert result["p_up_mean"] is None
    assert result["n_up"] == 0 and result["n_down"] == 0 and result["n_neutral"] == 0
    assert result["target_date"] == "2026-10-01"


def test_nan_values_from_pandas_records_treated_as_missing():
    # forecast_summary trả records qua DataFrame → ô trống thành NaN thay vì None.
    nan = float("nan")
    rows = [
        make_row(model="naive", predicted_price=nan, predicted_direction=nan, probability_up=nan),
        make_row(model="arima", predicted_price=24.9, predicted_direction="up", probability_up=0.7),
    ]
    result = tomorrow_consensus(rows)
    assert result["price_mean"] == pytest.approx(24.9)
    assert result["p_up_mean"] == pytest.approx(0.7)
    assert result["n_up"] == 1
    assert result["n_models"] == 2


def test_infinite_values_are_excluded():
    rows = [
        make_row(model="naive", predicted_price=math.inf),
        make_row(model="arima", predicted_price=24.5),
    ]
    result = tomorrow_consensus(rows)
    assert result["price_mean"] == pytest.approx(24.5)


def test_sqlite_row_like_object_supported():
    """forecast_summary trả list[dict], nhưng hàm vẫn chịu được sqlite3.Row."""

    class RowLike:  # giả lập sqlite3.Row: chỉ hỗ trợ row["key"]
        def __init__(self, data: dict):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

    rows = [
        RowLike(make_row(model="naive", predicted_price=24.0, predicted_direction="neutral", probability_up=0.5)),
        RowLike(make_row(model="arima", predicted_price=24.4, predicted_direction="up", probability_up=0.55)),
    ]
    result = tomorrow_consensus(rows)
    assert result["has_forecast"] is True
    assert result["n_models"] == 2
    assert result["price_mean"] == pytest.approx(24.2)
    assert result["n_up"] == 1
    assert result["n_neutral"] == 1


# ---------------------------------------------------------------------------
# 3) Đủ dữ liệu
# ---------------------------------------------------------------------------


def test_full_consensus_across_models():
    rows = [
        make_row(model="naive", predicted_price=24.5, predicted_direction="up", probability_up=0.6),
        make_row(model="arima", predicted_price=24.7, predicted_direction="up", probability_up=0.55),
        make_row(model="lightgbm", predicted_price=None, predicted_direction="down", probability_up=None),
        make_row(model="lstm", predicted_price=24.3, predicted_direction="neutral", probability_up=0.4),
        # dòng h=5 phải bị lọc bỏ, KHÔNG tính vào n_models/mean:
        make_row(model="naive", horizon=5, target_date="2026-10-07",
                 predicted_price=26.0, predicted_direction="up", probability_up=0.9),
    ]
    result = tomorrow_consensus(rows)
    assert result["has_forecast"] is True
    assert result["target_date"] == "2026-10-01"
    assert result["n_models"] == 4
    assert result["price_mean"] == pytest.approx((24.5 + 24.7 + 24.3) / 3)
    assert result["price_min"] == pytest.approx(24.3)
    assert result["price_max"] == pytest.approx(24.7)
    assert result["p_up_mean"] == pytest.approx((0.6 + 0.55 + 0.4) / 3)
    assert result["n_up"] == 2
    assert result["n_down"] == 1
    assert result["n_neutral"] == 1


def test_target_date_trimmed_and_first_non_null():
    rows = [
        make_row(model="arima", target_date="2026-10-01T00:00:00+07:00"),
        make_row(model="lstm", target_date="2026-10-02"),
    ]
    result = tomorrow_consensus(rows)
    assert result["target_date"] == "2026-10-01"


def test_unknown_direction_not_counted():
    rows = [
        make_row(model="naive", predicted_direction="SIDEWAYS"),
        make_row(model="arima", predicted_direction="up"),
    ]
    result = tomorrow_consensus(rows)
    assert result["n_up"] == 1
    assert result["n_down"] == 0
    assert result["n_neutral"] == 0


def test_result_keys_are_stable():
    keys = set(tomorrow_consensus([make_row()]).keys())
    assert keys == {
        "has_forecast", "target_date", "n_models",
        "price_mean", "price_min", "price_max", "p_up_mean",
        "n_up", "n_down", "n_neutral",
    }
    assert set(tomorrow_consensus([]).keys()) == {"has_forecast"}
    assert not pd.isna(1)  # pandas được import trong module — smoke import
