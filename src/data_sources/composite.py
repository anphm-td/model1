"""Ghép nhiều nguồn giá theo thứ tự ưu tiên (chuỗi dự phòng có merge).

Mục đích: CAFEF (nguồn chính, OHLCV) hiện chỉ phục vụ ~63 phiên gần nhất, còn
HOSE chính thức phục vụ lịch sử đầy đủ nhưng chỉ close+volume — chuỗi ghép cho
phản hợp lệ nhất: phần trùng ngày lấy số liệu nguồn ưu tiên cao hơn (combine_first),
phần chỉ nguồn sau có thì lấp thêm, không bịa dữ liệu.

Quy tắc:
- Gọi nguồn theo thứ tự; nguồn kế tiếp chỉ được gọi khi nguồn trước:
    * trả khung rỗng, HOẶC
    * ``start`` được chỉ định và phiên cũ nhất của khung > start
      (chưa phủ đến đầu khoảng yêu cầu).
- Merge: ``primary.combine_first(backup)`` — ngày trùng lấy giá trị nguồn ưu
  tiên cao hơn (kể cả khi nguồn ưu tiên có NaN ở ô đó thì lấy nguồn sau).
- Mọi nguồn đều không có dữ liệu → khung rỗng + cảnh báo tổng hợp (không raise).
"""
from __future__ import annotations

import logging
from collections.abc import Sequence

import pandas as pd

from .base import FetchOutcome, PriceDataSource, normalize_ohlcv

logger = logging.getLogger(__name__)


class PriorityMergeSource(PriceDataSource):
    """Chuỗi nguồn giá có thứ tự ưu tiên + merge theo ngày."""

    def __init__(self, sources: Sequence[PriceDataSource]) -> None:
        if not sources:
            raise ValueError("PriorityMergeSource cần ít nhất một nguồn")
        self._sources = list(sources)

    @property
    def name(self) -> str:
        return "+".join(source.name for source in self._sources)

    @property
    def sources(self) -> list[PriceDataSource]:
        """Các nguồn con theo thứ tự ưu tiên (chẩn đoán/test)."""
        return list(self._sources)

    def fetch_outcome(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> FetchOutcome:
        warnings: list[str] = []
        merged: pd.DataFrame | None = None
        used: list[str] = []

        for source in self._sources:
            if merged is not None and self._covers(merged, start):
                break  # nguồn đã phủ đủ khoảng — không cần nguồn sau
            outcome = source.fetch_outcome(symbol, start=start, end=end)
            warnings.extend(outcome.warnings)
            if outcome.frame.empty:
                continue
            used.append(outcome.source)
            merged = outcome.frame if merged is None else merged.combine_first(outcome.frame)

        if merged is None:
            warnings.append(
                f"Mọi nguồn trong chuỗi ({self.name}) đều không có dữ liệu cho {symbol} [{start}, {end}]"
            )
            for warning in warnings:
                logger.warning("%s", warning)
            return FetchOutcome(frame=normalize_ohlcv(None), source="none", warnings=warnings)
        return FetchOutcome(frame=normalize_ohlcv(merged), source="+".join(used), warnings=warnings)

    @staticmethod
    def _covers(frame: pd.DataFrame, start: str | None) -> bool:
        """Khung đã phủ đến đầu khoảng chưa (start=None → coi như chưa, thử nguồn sau)."""
        if start is None:
            return False
        return bool(frame.index.min() <= pd.Timestamp(start))
