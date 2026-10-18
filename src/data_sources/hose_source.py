"""Sàn HOSE chính thức (api.hsx.vn) — nguồn lịch sử đầy đủ (close + volume).

Hợp đồng — xác minh trực tiếp từ máy này (29/09/2026):
- GET https://api.hsx.vn/mk/api/v1/market/securities/chart/{symbol}
      ?fromDate=YYYY-MM-DD&toDate=YYYY-MM-DD
  header: User-Agent trình duyệt + Referer/Origin https://www.hsx.vn/.
  Đây là endpoint biểu đồ của website chính thức HOSE (bóc từ JS bundle
  main.d430e296.js của www.hsx.vn).
- Response JSON: {"data": [ {"reportdate": <epoch giây>, "closeprice": <đồng>,
  "totalshare": <cổ phiếu>}, ... ], "success": bool, "message": str}
- Lịch sử ĐẦY ĐỦ từ ngày niêm yết (TCB: 2.078 phiên từ 04/06/2018; FPT: 2.926
  phiên từ 05/01/2015).

GIỚI HẠN đã xác minh (nêu rõ, không bịa):
- Endpoint chỉ có giá ĐÓNG CỬA + khối lượng — open/high/low được điền NaN.
- closeprice ở đơn vị ĐỒNG → chuẩn hoá về NGHÌN ĐỒNG (÷1000) để khớp đơn vị
  CAFEF (nguồn chính theo spec).
- totalshare gồm cả khối lượng khớp lệnh lớn (big lot) — hơi khác khối lượng
  khớp lệnh thường của CAFEF; ngày trùng nhau ưu tiên số liệu CAFEF.
- Giá CHƯA điều chỉnh hành động công ty (đổi mệnh giá/cổ phiếu thưởng tạo bước
  nhảy giá giả trong chuỗi — lớp feature/labeling phải biết điều này).
- Chỉ số: "VNINDEX" NẰM TRONG danh sách UNSUPPORTED_SYMBOLS — chuỗi "VNI" của
  endpoint chart là chuỗi thưa (≈73 điểm cách nhau hàng tháng), ghép vào cache
  ngày sẽ làm sai lợi nhuận giữa 2 phiên cách xa, nên nguồn này chủ động trả
  khung rỗng cho VNINDEX và để CAFEF đảm nhiệm (~64 phiên gần nhất, ngày liên tục).
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests

from .base import (
    FetchOutcome,
    PriceDataSource,
    SourceError,
    filter_ohlcv_range,
    normalize_ohlcv,
)
from .cafef_source import (
    BROWSER_USER_AGENT,
    TCBS_DEFAULT_START,
    _clean_symbol,
    _get_json_with_retry,
)

logger = logging.getLogger(__name__)

HOSE_CHART_URL = "https://api.hsx.vn/mk/api/v1/market/securities/chart/{symbol}"

HOSE_HEADERS: dict[str, str] = {
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.hsx.vn/",
    "Origin": "https://www.hsx.vn",
}

#: Mã KHÔNG được lấy qua endpoint chart: "VNI" của HOSE là chuỗi THƯA (≈73 điểm
#: cách nhau hàng tháng trải 2015→nay) — ghép vào cache NGÀY sẽ làm lợi nhuận
#: giữa 2 điểm cách xa bị tính nhầm là lợi nhuận 1 phiên. VN-Index chỉ lấy qua
#: CAFEF (khoảng ~64 phiên gần nhất, ngày liên tục).
UNSUPPORTED_SYMBOLS: frozenset[str] = frozenset({"VNINDEX"})


def _hose_rows_to_frame(rows: list[Mapping[str, Any]]) -> pd.DataFrame:
    """Đổi hàng HOSE (đồng, epoch giây) → frame OHLCV chuẩn (nghìn đồng, NaN OHL)."""
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_ts = row.get("reportdate", row.get("reportDate"))
        if raw_ts in (None, ""):
            continue
        try:
            ts = pd.to_datetime(float(raw_ts), unit="s", utc=True)
        except (TypeError, ValueError):
            continue
        # ngày phiên theo giờ Việt Nam (timestamp có thể là 00:00 hoặc giờ tối VN)
        session_date = ts.tz_convert("Asia/Ho_Chi_Minh").tz_localize(None).normalize()
        close_vnd = row.get("closeprice", row.get("closePrice"))
        records.append(
            {
                "date": session_date,
                # HOSE chỉ cung cấp close + volume: OHL = NaN (không bịa số)
                "open": None,
                "high": None,
                "low": None,
                # đồng → nghìn đồng (khớp đơn vị CAFEF, nguồn chính)
                "close": float(close_vnd) / 1000.0 if close_vnd not in (None, "") else None,
                "volume": row.get("totalshare", row.get("totalShare")),
            }
        )
    return pd.DataFrame.from_records(records)


class HoseOfficialSource(PriceDataSource):
    """Nguồn dự phòng lịch sử đầy đủ từ SÀN HOSE chính thức (close + volume).

    Dùng để lấp khoảng lịch sử mà CAFEF không còn phục vụ (hiện CAFEF chỉ trả
    ~63 phiên gần nhất) và khi TCBS không truy cập được. Không raise khi lỗi
    mạng ở mức gọi đơn lẻ — raise SourceError để tầng ghép (PriorityMergeSource)
    gom cảnh báo.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep_fn or time.sleep

    @property
    def name(self) -> str:
        return "hose_official"

    def fetch_outcome(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> FetchOutcome:
        cleaned = _clean_symbol(symbol)
        if cleaned in UNSUPPORTED_SYMBOLS:
            message = (
                f"HOSE official không phục vụ lịch sử NGÀY liên tục cho {cleaned} "
                "(chuỗi 'VNI' của endpoint chart là chuỗi thưa, ghép vào cache ngày "
                "sẽ làm sai lợi nhuận) — trả khung rỗng để nguồn khác đảm nhiệm."
            )
            logger.warning("%s", message)
            return FetchOutcome(frame=normalize_ohlcv(None), source=self.name, warnings=[message])
        start_ts = pd.Timestamp(start) if start else pd.Timestamp(TCBS_DEFAULT_START)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp.now(tz="UTC").tz_convert("Asia/Ho_Chi_Minh").tz_localize(None)
        url = HOSE_CHART_URL.format(symbol=cleaned)
        params = {
            "fromDate": start_ts.strftime("%Y-%m-%d"),
            "toDate": end_ts.strftime("%Y-%m-%d"),
        }
        payload = _get_json_with_retry(
            session=self._session,
            url=url,
            params=params,
            headers=HOSE_HEADERS,
            timeout=self._timeout,
            max_retries=self._max_retries,
            sleep_fn=self._sleep,
            label=f"HOSE official chart {cleaned}",
        )
        rows = payload.get("data") if isinstance(payload, dict) else None
        if rows is None:
            raise SourceError(f"HOSE official trả phản hồi thiếu 'data' cho {cleaned}: {str(payload)[:200]}")
        if not isinstance(rows, list):
            raise SourceError(f"HOSE official trả 'data' không phải danh sách cho {cleaned}")
        frame = normalize_ohlcv(_hose_rows_to_frame(rows))
        return FetchOutcome(
            frame=filter_ohlcv_range(frame, start, end),
            source=self.name,
            warnings=[],
        )


__all__ = ["HOSE_CHART_URL", "HOSE_HEADERS", "UNSUPPORTED_SYMBOLS", "HoseOfficialSource"]
