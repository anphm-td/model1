"""MarketContext — đặc trưng bối cảnh thị trường (nhóm `market_context`).

Hiện tại gồm hai feature từ VN-Index (config: ``data.index_ticker``):
- ``vni_ret_1d``  : log-return của VN-Index TRÊN CÙNG PHIÊN t với mã;
- ``vni_beta_60`` : beta trượt 60 phiên của mã vs VN-Index =
  Cov(r_mã, r_index) / Var(r_index) tính trên 60 phiên gần nhất (mỗi phiên
  phải có CẢ HAI chuỗi — "cặp hoàn chỉnh"); window trượt chỉ gồm các cặp
  trong quá khứ nên vẫn causal.

Ràng buộc chống leakage & lệch lịch:
- Giá trị tại t của mã chỉ dùng dữ liệu index tại các phiên <= t;
- Index thiếu dữ liệu phiên nào của mã (index nghỉ/ticker vẫn giao dịch, hoặc
  cache index ngắn) → feature index tại phiên đó là NaN. TUYỆT ĐỐI không
  ffill/bfill: return của phiên khác không phải return "cùng phiên", và việc
  điền giá trị tương lai/quá khứ giả tạo quan sát low-volatility giả (bài học
  ffill ngày nghỉ).
- Index frame rỗng (nguồn lỗi — xem ``load_vnindex``) → hai cột toàn NaN +
  log cảnh báo; pipeline không chết.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import FeatureProvider, validate_ohlcv_frame

logger = logging.getLogger(__name__)

#: Số phiên trượt của beta (theo yêu cầu thiết kế: beta 60 phiên).
BETA_WINDOW: int = 60


class MarketContext(FeatureProvider):
    """Bối cảnh thị trường cho từng mã, join theo ngày phiên giao dịch."""

    def __init__(self, index_ohlcv: pd.DataFrame | None = None) -> None:
        """``index_ohlcv``: khung OHLCV của VN-Index (hợp đồng như ticker).

        Truyền khung rỗng/None khi nguồn index lỗi — provider trả feature NaN.
        """
        self._index = index_ohlcv if index_ohlcv is not None else pd.DataFrame()

    @property
    def name(self) -> str:
        return "market_context"

    def provide(self, df: pd.DataFrame) -> pd.DataFrame:
        validate_ohlcv_frame(df)
        out = pd.DataFrame(index=df.index, columns=["vni_ret_1d", "vni_beta_60"])
        out = out.astype("float64")

        if self._index is None or self._index.empty or "close" not in self._index.columns:
            logger.warning(
                "MarketContext: không có dữ liệu VN-Index — hai feature "
                "vni_ret_1d/vni_beta_60 thành NaN cho %d phiên.",
                len(df),
            )
            return out

        index_close = self._index["close"].astype("float64")
        index_ret = np.log(index_close).diff()  # log-return trên lịch của index

        # --- vni_ret_1d: log-return của index TRÊN PHIÊN t của mã ---
        # reindex theo lịch MÃ, không ffill: index thiếu phiên → NaN.
        out["vni_ret_1d"] = index_ret.reindex(df.index)

        # --- vni_beta_60: beta trượt trên các cặp (mã, index) hoàn chỉnh ---
        ticker_ret = np.log(df["close"].astype("float64")).diff()
        pairs = pd.concat(
            {"ticker": ticker_ret, "index": index_ret}, axis=1, join="inner"
        ).dropna(how="any")
        if len(pairs) >= 2:
            cov = pairs["ticker"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).cov(
                pairs["index"]
            )
            var = pairs["index"].rolling(BETA_WINDOW, min_periods=BETA_WINDOW).var(ddof=1)
            beta = cov / var
            out["vni_beta_60"] = beta.reindex(df.index)

        return out.replace([np.inf, -np.inf], np.nan)


__all__ = ["BETA_WINDOW", "MarketContext"]
