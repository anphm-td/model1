"""Điều phối tải dữ liệu giá + cache parquet, cập nhật tăng dần.

Chuỗi nguồn mặc định (PriorityMergeSource, theo thứ tự ưu tiên):
1. CafefSource — CAFEF PriceHistory (nguồn chính theo spec; OHLCV đầy đủ) với
   fallback TCBS bên trong. ĐÃ XÁC MINH 29/09/2026: CAFEF chỉ còn phục vụ ~63
   phiên gần nhất (TotalCount=63, mọi tham số ngày bị bỏ qua) và DNS TCBS chết
   từ máy này.
2. HoseOfficialSource — API chính thức của SÀN HOSE (api.hsx.vn, bóc từ website
   www.hsx.vn): lịch sử ĐẦY ĐỦ từ ngày niêm yết nhưng CHỈ có close + volume
   (open/high/low = NaN, đơn vị đã quy về nghìn đồng khớp CAFEF). Nguồn này chỉ
   được gọi khi nguồn trước chưa phủ đến đầu khoảng yêu cầu.

Kết quả ghép: khung OHLCV tối đa dữ liệu thật có thể có — phần cũ hơn cửa sổ
CAFEF có close/volume thật, open/high/low NaN (tài liệu rõ, KHÔNG bịa, KHÔNG
ffill ngày nghỉ). Lớp feature phải xử lý được NaN của open/high/low.

- Cache thô:    data/raw/{ticker}.parquet       — khung OHLCV chuẩn từ nguồn.
- Cache sạch:   data/processed/{ticker}.parquet — khung OHLCV đã chuẩn hoá/kiểm định
  (sort, dedupe, dtype float64, bỏ phiên close không hợp lệ). Tại đây KHÔNG sinh
  feature nào — lớp feature (src/features) đọc processed để sinh đặc trưng.
- Cập nhật tăng dần: chỉ tải phần thiếu kể từ phiên cuối đã cache (overlap đúng
  1 phiên cuối để bắt chỉnh sửa trong ngày), không tải lại toàn bộ lịch sử.
- Nguồn lỗi/không có dữ liệu → trả khung rỗng + log cảnh báo từng nguyên nhân
  (không raise) — pipeline/dashboard phải sống được với khung rỗng.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .data_sources.base import (
    FetchOutcome,
    PriceDataSource,
    SourceError,
    filter_ohlcv_range,
    normalize_ohlcv,
)
from .data_sources.cafef_source import CafefSource
from .data_sources.composite import PriorityMergeSource
from .data_sources.hose_source import HoseOfficialSource

logger = logging.getLogger(__name__)

#: Thư mục data/ của dự án.
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

#: Tải lịch sử 1D từ 2015 (khớp config.yaml: data.start_date).
DEFAULT_START = "2015-01-01"


def default_source() -> PriceDataSource:
    """Chuỗi nguồn mặc định: CAFEF (chính) → HOSE chính thức (lấp lịch sử đầy đủ)."""
    return PriorityMergeSource([CafefSource(), HoseOfficialSource()])


def raw_path(ticker: str, data_dir: Path = DATA_DIR) -> Path:
    """Đường dẫn cache thô data/raw/{ticker}.parquet."""
    return Path(data_dir) / "raw" / f"{ticker.upper()}.parquet"


def processed_path(ticker: str, data_dir: Path = DATA_DIR) -> Path:
    """Đường dẫn cache sạch data/processed/{ticker}.parquet."""
    return Path(data_dir) / "processed" / f"{ticker.upper()}.parquet"


def last_cached_date(ticker: str, data_dir: Path = DATA_DIR) -> pd.Timestamp | None:
    """Phiên mới nhất đã cache cho ``ticker`` (None nếu chưa có cache)."""
    frame = _read_parquet(raw_path(ticker, data_dir))
    if frame is None or frame.empty:
        return None
    return frame.index.max()


def load_history(
    ticker: str,
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
    source: PriceDataSource | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """Tải (tăng dần) và cache OHLCV theo ngày cho ``ticker``; trả khung [start, end].

    - Lần đầu (chưa có cache / cache hỏng / refresh=True): tải từ ``start``
      (mặc định 2015-01-01, khớp config.yaml), ghi cả raw lẫn processed. Khi
      nguồn chính (CAFEF) không phủ đến ``start``, nguồn sau (HOSE chính thức,
      close+volume) được gọi để lấp phần lịch sử cũ hơn.
    - Các lần sau: chỉ tải từ phiên cuối đã cache (cập nhật tăng dần); các
      phiên trùng bị đè bằng dữ liệu mới tải về (bắt chỉnh sửa trong ngày).
    - Nguồn lỗi/không có dữ liệu: trả khung rỗng (nếu chưa từng có cache) hoặc
      cache hiện có (khi refresh=True mà nguồn lỗi) — kèm log cảnh báo.
    """
    ticker = str(ticker).strip().upper()
    src = source or default_source()
    path = raw_path(ticker, data_dir)
    existing = _read_parquet(path)
    cached = None if refresh else existing

    if cached is None or cached.empty:
        outcome = src.fetch_outcome(ticker, start=start, end=end)
        if outcome.frame.empty:
            _log_warnings(ticker, outcome)
            if existing is not None and not existing.empty:
                logger.warning(
                    "load_history(%s): nguồn lỗi khi refresh — trả cache hiện có (%d phiên)",
                    ticker,
                    len(existing),
                )
                return filter_ohlcv_range(existing, start, end)
            return outcome.frame
        _write_cache(path, outcome.frame, data_dir)
        logger.info("load_history(%s): tải %d phiên (%s→%s) qua nguồn %s",
                    ticker, len(outcome.frame), start, end or "nay", outcome.source)
        return filter_ohlcv_range(outcome.frame, start, end)

    last = cached.index.max()
    if end is not None and pd.Timestamp(end) < last:
        return filter_ohlcv_range(cached, start, end)  # cache đã phủ quá end

    fetch_from = last.strftime("%Y-%m-%d")  # overlap đúng phiên cuối
    outcome = src.fetch_outcome(ticker, start=fetch_from, end=end)
    if outcome.frame.empty:
        _log_warnings(ticker, outcome, note="giữ nguyên cache hiện có")
        return filter_ohlcv_range(cached, start, end)
    combined = _merge_frames(cached, outcome.frame)
    _write_cache(path, combined, data_dir)
    logger.info(
        "load_history(%s): cập nhật tăng dần từ %s qua nguồn %s (cache: %d phiên)",
        ticker, fetch_from, outcome.source, len(combined),
    )
    return filter_ohlcv_range(combined, start, end)


def load_vnindex(
    start: str = DEFAULT_START,
    end: str | None = None,
    refresh: bool = False,
    source: PriceDataSource | None = None,
    data_dir: Path = DATA_DIR,
) -> pd.DataFrame:
    """VN-Index cho MarketContext: cùng cơ chế cache/tăng dần như ticker thường.

    Khi nguồn lỗi, trả khung rỗng + cảnh báo — MarketContext phải chịu được
    (feature bối cảnh thành NaN, pipeline không chết).
    """
    return load_history("VNINDEX", start=start, end=end, refresh=refresh, source=source, data_dir=data_dir)


# ---------------------------------------------------------------------------
# Nội bộ
# ---------------------------------------------------------------------------


def _read_parquet(path: Path) -> pd.DataFrame | None:
    """Đọc cache parquet; file thiếu/hỏng/sai schema → None (tải lại từ nguồn)."""
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — cache hỏng không được giết pipeline
        logger.warning("Đọc cache lỗi (%s): %s — sẽ tải lại từ nguồn", path, exc)
        return None
    try:
        return normalize_ohlcv(frame)
    except SourceError as exc:
        logger.warning("Cache %s sai schema (%s) — sẽ tải lại từ nguồn", path, exc)
        return None


def _write_cache(path: Path, frame: pd.DataFrame, data_dir: Path) -> None:
    """Ghi cache thô + cache sạch (processed) cho ``ticker``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    processed = Path(data_dir) / "processed" / path.name
    processed.parent.mkdir(parents=True, exist_ok=True)
    normalize_ohlcv(frame).to_parquet(processed)


def _merge_frames(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    """Ghép cache cũ + phần mới: các phiên có trong ``fresh`` đè lên bản cache."""
    if fresh.empty:
        return cached
    if cached.empty:
        return fresh
    cut = fresh.index.min()
    head = cached.loc[cached.index < cut]
    combined = pd.concat([head, fresh])
    combined = combined.loc[~combined.index.duplicated(keep="last")].sort_index(kind="stable")
    return normalize_ohlcv(combined)


def _log_warnings(ticker: str, outcome: FetchOutcome, note: str = "") -> None:
    prefix = f"load_history({ticker})" + (f" [{note}]" if note else "")
    for warning in outcome.warnings:
        logger.warning("%s: %s", prefix, warning)


__all__ = [
    "DATA_DIR",
    "DEFAULT_START",
    "PROCESSED_DIR",
    "RAW_DIR",
    "default_source",
    "last_cached_date",
    "load_history",
    "load_vnindex",
    "processed_path",
    "raw_path",
]
