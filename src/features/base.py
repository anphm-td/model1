"""Interface FeatureProvider — hợp đồng chung cho mọi nhóm đặc trưng.

Hợp đồng (bắt buộc với mọi hiện thực):
- ``provide(df)`` nhận khung OHLCV chuẩn (xem ``src/data_sources/base.py``:
  index DatetimeIndex tên ``date`` tăng dần không trùng, cột
  ``open/high/low/close/volume`` float64, chỉ chứa phiên giao dịch thật) và
  trả về DataFrame đặc trưng:
    * index TRÙNG với index của ``df`` (cùng thứ tự, cùng độ dài) — mỗi dòng
      là đặc trưng của phiên t, mỗi cột một feature;
    * KHÔNG chứa cột giá mức (open/high/low/close/volume...) — fit trên
      levels thì model học "giá đang ở đâu" chứ không học quá trình sinh giá
      (bài học từ các dự án tương tự; assemble.py còn chặn lại bằng blocklist);
    * giá trị tại t CHỈ được tính từ dữ liệu có thời điểm <= t. Mọi rolling
      phải khép về quá khứ, mọi dịch phải về phía quá khứ; tuyệt đối không
      ``shift(-k)`` ngoài labeling, không ``center=True``, không ``bfill``.

Chính sách NaN (thống nhất giữa các provider):
- "Warm-up": trước khi đủ cửa sổ indicator, giá trị là NaN (dự đoán được,
  unit test neo) — tầng walk-forward tự loại các dòng NaN.
- Với lịch sử ghép từ nguồn chỉ có close (open/high/low = NaN — xem
  ``src/data_loader.py``): feature chỉ dùng close vẫn có giá trị; feature
  bắt buộc cần high/low (Stochastic) thành NaN; ATR tự hạ cấp về biên độ
  close-to-close cho các phiên đó. KHÔNG ffill/bfill để "vá" NaN.
"""
from __future__ import annotations

import abc

import pandas as pd

#: Cột OHLCV bắt buộc của khung đầu vào (khớp hợp đồng data_sources).
REQUIRED_INPUT_COLUMNS: tuple[str, ...] = ("close", "high", "low", "open", "volume")


class FeatureProviderError(ValueError):
    """Đầu vào của provider không thoả hợp đồng OHLCV/thời gian."""


class FeatureProvider(abc.ABC):
    """Một nhóm đặc trưng bật/tắt theo ``features.feature_groups`` trong config."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Tên nhóm feature ("price", "market_context", ...), dùng trong log."""

    @abc.abstractmethod
    def provide(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sinh DataFrame đặc trưng theo index ngày giao dịch của ``df``.

        Xem hợp đồng ở docstring module. Không được thay đổi ``df`` truyền vào.
        """


def validate_ohlcv_frame(df: pd.DataFrame) -> None:
    """Kiểm tra ``df`` thoả hợp đồng đầu vào của provider; lỗi thì raise.

    - Phải là DataFrame có đủ cột bắt buộc;
    - index phải là DatetimeIndex tăng dần, không trùng ngày (hợp đồng của
      ``normalize_ohlcv`` ở tầng dữ liệu — vi phạm là lỗi tầng trên, không
      tự sửa âm thầm vì sort lại có thể đổi nghĩa target/feature).
    """
    if not isinstance(df, pd.DataFrame):
        raise FeatureProviderError(
            f"FeatureProvider cần DataFrame OHLCV, nhận được {type(df).__name__}"
        )
    if df.empty:
        raise FeatureProviderError(
            "FeatureProvider nhận khung OHLCV rỗng — kiểm tra nguồn dữ liệu/"
            "cache trước khi sinh đặc trưng (lỗi phải báo rõ, không im lặng)."
        )
    missing = [col for col in REQUIRED_INPUT_COLUMNS if col not in df.columns]
    if missing:
        raise FeatureProviderError(
            f"Khung OHLCV thiếu cột {missing} (các cột hiện có: {list(df.columns)})"
        )
    if not isinstance(df.index, pd.DatetimeIndex):
        raise FeatureProviderError(
            f"Index phải là DatetimeIndex, nhận được {type(df.index).__name__}"
        )
    if not df.index.is_monotonic_increasing:
        raise FeatureProviderError(
            "Index ngày phải tăng dần (chuỗi thời gian) — nhận index lộn xộn."
        )
    if df.index.has_duplicates:
        raise FeatureProviderError(
            "Index ngày không được trùng (mỗi phiên một dòng) — kiểm tra tầng dữ liệu."
        )


__all__ = [
    "FeatureProvider",
    "FeatureProviderError",
    "REQUIRED_INPUT_COLUMNS",
    "validate_ohlcv_frame",
]
