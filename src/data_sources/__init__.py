"""Nguồn dữ liệu giá: hợp đồng chung, nguồn chính CAFEF (fallback TCBS) và
nguồn chính thức HOSE (lịch sử đầy đủ, close+volume).

Module:
- base: interface PriceDataSource, SourceError, FetchOutcome, normalize_ohlcv —
  hợp đồng khung OHLCV chuẩn (index ngày, cột open/high/low/close/volume,
  chỉ chứa phiên giao dịch thực tế, KHÔNG ffill ngày nghỉ).
- cafef_source: CafefClient (CAFEF PriceHistory — phân trang 20 dòng/trang,
  retry 3 lần, hai host failover), TcbsClient (fallback OHLC theo epoch giây),
  CafefSource (ghép CAFEF → TCBS; cả hai fail → khung rỗng + cảnh báo),
  HoseSymbolDirectory (danh sách mã HOSE, cache data/symbols.parquet).
- hose_source: HoseOfficialSource — API chính thức SÀN HOSE (api.hsx.vn):
  lịch sử đầy đủ từ ngày niêm yết nhưng CHỈ close + volume (OHL = NaN, đơn vị
  nghìn đồng khớp CAFEF). Cần thiết vì CAFEF hiện chỉ phục vụ ~63 phiên gần nhất
  (xác minh 29/09/2026).
- composite: PriorityMergeSource — ghép chuỗi nguồn theo thứ tự ưu tiên, merge
  theo ngày (ngày trùng lấy nguồn ưu tiên cao hơn).

Lưu ý: vnstock/vnstock3/vnai KHÔNG dùng (đã chết trên PyPI). Cấu hình nguồn
trong config.yaml (data.source, data.fallback_sources) cần cập nhật khi bổ sung
nguồn chính thức HOSE.
"""
from .base import (
    OHLCV_COLUMNS,
    FetchOutcome,
    PriceDataSource,
    SourceError,
    filter_ohlcv_range,
    normalize_ohlcv,
)
from .cafef_source import (
    CafefClient,
    CafefSource,
    HoseSymbolDirectory,
    SymbolsResult,
    TcbsClient,
)
from .composite import PriorityMergeSource
from .hose_source import HoseOfficialSource

__all__ = [
    "CafefClient",
    "CafefSource",
    "FetchOutcome",
    "HoseOfficialSource",
    "HoseSymbolDirectory",
    "OHLCV_COLUMNS",
    "PriceDataSource",
    "PriorityMergeSource",
    "SourceError",
    "SymbolsResult",
    "TcbsClient",
    "filter_ohlcv_range",
    "normalize_ohlcv",
]
