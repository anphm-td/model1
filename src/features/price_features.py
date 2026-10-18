"""PriceFeatures — nhóm đặc trưng tính từ OHLCV của chính mã (nhóm `price`).

Tất cả feature đều stationary (tỷ lệ/độ litt/%, KHÔNG có giá mức) và causal
(tại t chỉ dùng dữ liệu <= t). Danh sách feature:

Lợi nhuận & độ trễ (log-return):
- ``ret_1d`` / ``ret_5d`` / ``ret_20d``  : log(close_t / close_{t-k})
- ``lag_ret_1..5``                       : log-return của phiên t-k (ret_1d.shift(k))

Xu hướng — vị trí giá vs trung bình động (không phát ra SMA/EMA thô vì đó là
giá mức; dạng ratio ``close/MA - 1`` mới là thông tin chuẩn hoá được học):
- ``sma_{5,10,20,50}_ratio`` : close/SMA_N - 1
- ``ema_{5,10,20,50}_ratio`` : close/EMA_N - 1

Vị trí trong dải 52 tuần (252 phiên — dùng close, chịu được high/low NaN):
- ``dist_52w_high`` : close/max_252 - 1 (<= 0)
- ``dist_52w_low``  : close/min_252 - 1 (>= 0)

Dao động/biến động:
- ``rsi_14``         : RSI Wilder 14 phiên (xấp xỉ ewm alpha=1/14) trong [0, 100]
- ``macd_norm`` / ``macd_signal_norm`` / ``macd_hist_norm``:
  (EMA12-EMA26), tín hiệu EMA9, histogram — đều CHIA cho close để bỏ đơn vị giá
- ``stoch_k_14`` / ``stoch_d_14`` : %K = 100*(C - minL14)/(maxH14 - minL14), %D = SMA3(%K)
  (cần high/low → NaN trên lịch sử chỉ có close)
- ``atr_14_pct``     : ATR(14) Wilder / close. True Range = max(high-low,
  |high-prev_close|, |low-prev_close|, |close-prev_close|) — thành phần cuối
  không đổi TR chuẩn khi high/low có mặt (luôn <= max 3 thành phần đầu) nhưng
  cho TR = |close-prev_close| khi high/low NaN (lịch sử nguồn HOSE chính thức)
- ``bb_pos_20``      : vị trí Bollinger (close - SMA20)/(2*std20), std ddof=0
- ``bb_width_20``    : 4*std20/SMA20 (độ rộng dải, chuẩn hoá)

Khối lượng:
- ``vol_ratio_20``   : volume / SMA20(volume)
- ``obv_delta_5`` / ``obv_delta_20``: (OBV_t - OBV_{t-n}) / SMA20(volume) —
  OBV thô là tích luỹ từ điểm gốc nên phụ thuộc điểm bắt đầu (non-stationary,
  cùng lý do không phát giá mức); chỉ phát phần THAY ĐỔI chuẩn hoá. Phiên thiếu
  volume đóng góp 0 vào OBV.

Lịch: ``dow`` (0=Thứ 2..4=Thứ 6), ``month`` (1..12) — suy trực tiếp từ ngày phiên.

Chính sách NaN: warm-up (chưa đủ cửa sổ) → NaN; mọi inf (chia 0) → NaN.
Không ffill/bfill ở bất kỳ đâu trong module này.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureProvider, validate_ohlcv_frame

#: Cửa sổ SMA/EMA theo yêu cầu thiết kế.
MA_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)
#: Số phiên "một năm giao dịch" cho đỉnh/đáy 52 tuần.
YEAR_BARS: int = 252


def _mask_warmup(series: pd.Series, n: int) -> pd.Series:
    """Đặt NaN cho n dòng đầu (chưa đủ cửa sổ của indicator EWM/RSI)."""
    out = series.copy()
    if n > 0:
        out.iloc[:n] = np.nan
    return out


class PriceFeatures(FeatureProvider):
    """Đặc trưng kỹ thuật từ OHLCV của chính mã — causal, không giá mức."""

    @property
    def name(self) -> str:
        return "price"

    def provide(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv_frame(df)
        close = df["close"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        volume = df["volume"].astype("float64")
        log_close = np.log(close)

        out = pd.DataFrame(index=df.index)

        # --- Lợi nhuận log 1/5/20 phiên (chỉ nhìn quá khứ) ---
        for k in (1, 5, 20):
            out[f"ret_{k}d"] = log_close.diff(k)

        # --- Lag return 1..5: log-return của các phiên t-1..t-5 ---
        ret_1d = out["ret_1d"]
        for k in range(1, 6):
            out[f"lag_ret_{k}"] = ret_1d.shift(k)

        # --- SMA/EMA dưới dạng vị trí giá vs MA (close/MA - 1) ---
        for n in MA_WINDOWS:
            sma = close.rolling(n, min_periods=n).mean()
            out[f"sma_{n}_ratio"] = close / sma - 1.0
            ema = close.ewm(span=n, adjust=False).mean()
            out[f"ema_{n}_ratio"] = close / _mask_warmup(ema, n - 1) - 1.0

        # --- Vị trí trong dải 52 tuần (dùng close — chịu được high/low NaN) ---
        high_52w = close.rolling(YEAR_BARS, min_periods=YEAR_BARS).max()
        low_52w = close.rolling(YEAR_BARS, min_periods=YEAR_BARS).min()
        out["dist_52w_high"] = close / high_52w - 1.0
        out["dist_52w_low"] = close / low_52w - 1.0

        # --- RSI(14) kiểu Wilder (ewm alpha=1/14) ---
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        avg_loss = loss.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        rs = avg_gain / avg_loss
        out["rsi_14"] = 100.0 - 100.0 / (1.0 + rs)
        # avg_loss = 0 (chuỗi chỉ tăng) → rs = inf → RSI = 100; cả hai = 0 (đứng
        # giá) → NaN — chính sách chung, xử lý inf cuối module.

        # --- MACD (chuẩn hoá theo close) ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd = _mask_warmup(macd, 25)  # cần EMA26 đủ cửa sổ
        signal = _mask_warmup(signal, 33)  # 26 + 9 - 1
        out["macd_norm"] = macd / close
        out["macd_signal_norm"] = signal / close
        out["macd_hist_norm"] = (macd - signal) / close

        # --- Stochastic %K/%D (cần high/low; NaN trên lịch sử chỉ có close) ---
        min_low = low.rolling(14, min_periods=14).min()
        max_high = high.rolling(14, min_periods=14).max()
        stoch_k = 100.0 * (close - min_low) / (max_high - min_low)
        out["stoch_k_14"] = stoch_k
        out["stoch_d_14"] = stoch_k.rolling(3, min_periods=3).mean()

        # --- ATR(14) Wilder, chuẩn hoá theo close ---
        prev_close = close.shift(1)
        tr_parts = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
                (close - prev_close).abs(),
            ],
            axis=1,
        )
        true_range = tr_parts.max(axis=1, skipna=True)
        atr = true_range.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        out["atr_14_pct"] = atr / close

        # --- Bollinger: vị trí trong dải và độ rộng dải ---
        sma20 = close.rolling(20, min_periods=20).mean()
        std20 = close.rolling(20, min_periods=20).std(ddof=0)
        out["bb_pos_20"] = (close - sma20) / (2.0 * std20)
        out["bb_width_20"] = 4.0 * std20 / sma20

        # --- Khối lượng ---
        vol_ma20 = volume.rolling(20, min_periods=20).mean()
        out["vol_ratio_20"] = volume / vol_ma20
        signed_volume = np.sign(close.diff()).fillna(0.0) * volume.fillna(0.0)
        obv = signed_volume.cumsum()
        for n in (5, 20):
            out[f"obv_delta_{n}"] = (obv - obv.shift(n)) / vol_ma20

        # --- Lịch (thứ/mô hình trong tuần, tháng trong năm) ---
        out["dow"] = df.index.dayofweek.to_numpy().astype("int8")
        out["month"] = df.index.month.to_numpy().astype("int8")

        # Chuẩn hoá chung: mọi inf (chia 0) thành NaN — không ffill/bfill.
        # Chỉ đụng cột float để giữ nguyên dtype int8 của dow/month.
        float_cols = out.select_dtypes(include=["floating"]).columns
        out[float_cols] = out[float_cols].replace([np.inf, -np.inf], np.nan)
        return out


__all__ = ["MA_WINDOWS", "YEAR_BARS", "PriceFeatures"]
