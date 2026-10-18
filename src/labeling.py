"""Gán nhãn (labeling) — tạo target regression & classification cho từng horizon.

Định nghĩa (neo cứng, unit test kiểm chứng trong tests/test_features.py):
- Regression: ``target_ret_{h}d`` tại t = log(close_{t+h} / close_t) — cài bằng
  ``np.log(close.shift(-h) / close)``: dịch GIÁ TƯƠNG LAI về dòng t (shift(-h)
  là dịch DÂNG LÊN, đúng chiều duy nhất cho target forward).
- Classification: ``target_dir_{h}d`` tại t ∈ {+1, 0, -1} theo LỢI NHUẬN ĐƠN
  GIẢN t→t+h (đơn vị phần trăm, khớp ``neutral_band_pct`` của config):
      r >  +band → +1 (tăng)      r <  -band → -1 (giảm)
      |r| <= band → 0 (neutral — vùng đóng, quy ước neo)
  Với band = 0.3% (config.yaml: neutral_band_pct), vùng neutral rất hẹp so với
  biến động giá cổ phiếu → chọn dạng so sánh trực tiếp, không thêm epsilon.

Chống leakage (ưu tiên số 1):
- Target tại t chỉ dùng close_t và close_{t+h}; các phiên t+1..t+h-1 và mọi
  phiên khác KHÔNG ảnh hưởng (test perturbation kiểm chứng riêng cho từng vị trí).
- ``h`` phiên CUỐI của khung không có target (NaN — chưa biết tương lai); tầng
  walk-forward/evaluation tự loại. Không bao giờ fill target bằng anything.

Feature và target KHÔNG BAO GIỜ trộn làm một: assemble.py sinh ma trận
feature (chỉ quá khứ), labeling.py sinh cột target (chỉ tương lai).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Nhãn phân loại hướng.
UP: int = 1
NEUTRAL: int = 0
DOWN: int = -1


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """Log-return t→t+h: log(close_{t+h}/close_t); h phiên cuối là NaN."""
    close = _validate_close(close)
    if int(horizon) < 1:
        raise ValueError(f"horizon phải >= 1, nhận {horizon}")
    horizon = int(horizon)
    return np.log(close.shift(-horizon)) - np.log(close)


def forward_simple_return(close: pd.Series, horizon: int) -> pd.Series:
    """Lợi nhuận đơn giản t→t+h: close_{t+h}/close_t - 1; h phiên cuối là NaN."""
    close = _validate_close(close)
    if int(horizon) < 1:
        raise ValueError(f"horizon phải >= 1, nhận {horizon}")
    horizon = int(horizon)
    return close.shift(-horizon) / close - 1.0


def classification_target(
    close: pd.Series,
    horizon: int,
    neutral_band_pct: float,
) -> pd.Series:
    """Nhãn 3 lớp theo hướng đi sau ``horizon`` phiên, vùng neutral ±band.

    ``neutral_band_pct`` theo phần trăm (0.3 nghĩa là ±0.3%). Quy ước ranh giới:
    |r| <= band → neutral (vùng đóng). Nhãn int8: +1/0/-1; h phiên cuối NaN.
    """
    if neutral_band_pct <= 0:
        raise ValueError(f"neutral_band_pct phải > 0 (%), nhận {neutral_band_pct}")
    band = float(neutral_band_pct) / 100.0
    ret = forward_simple_return(close, horizon)

    label = pd.Series(np.nan, index=ret.index, dtype="float64")
    valid = ret.notna()
    label[valid & (ret > band)] = UP
    label[valid & (ret < -band)] = DOWN
    label[valid & (ret.abs() <= band)] = NEUTRAL
    return label.astype("Int64")  # nullable int: NaN được giữ ở h phiên cuối


def build_targets(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5),
    neutral_band_pct: float = 0.3,
) -> pd.DataFrame:
    """Sinh toàn bộ cột target cho các ``horizons`` (mặc định [1, 5] của config).

    Trả DataFrame cùng index với ``df`` với cột ``target_ret_{h}d`` (regression)
    và ``target_dir_{h}d`` (classification). Chỉ cần cột ``close``.
    """
    if "close" not in df.columns:
        raise ValueError(f"build_targets cần cột 'close' (các cột: {list(df.columns)})")
    close = df["close"]
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"target_ret_{h}d"] = forward_log_return(close, h)
        out[f"target_dir_{h}d"] = classification_target(close, h, neutral_band_pct)
    return out


def _validate_close(close: pd.Series) -> pd.Series:
    if not isinstance(close, pd.Series):
        raise ValueError(f"close phải là pandas Series, nhận {type(close).__name__}")
    if close.isna().any():
        raise ValueError(
            "close chứa NaN — tầng dữ liệu phải cho ra chuỗi close đầy đủ "
            "(normalize_ohlcv đã bỏ phiên không hợp lệ); không gán nhãn trên giá bịa."
        )
    if (close <= 0).any():
        raise ValueError("close phải > 0 để tính log-return.")
    return close.astype("float64")


__all__ = [
    "DOWN",
    "NEUTRAL",
    "UP",
    "build_targets",
    "classification_target",
    "forward_log_return",
    "forward_simple_return",
]
