"""Nguồn dữ liệu giá: CAFEF PriceHistory (nguồn chính) + fallback TCBS OHLC.

Hợp đồng CAFEF — xác minh trực tiếp từ máy này (29/09/2026):
- GET https://cafef.vn/du-lieu/ajax/pagenew/datahistory/pricehistory.ashx
      ?Symbol=TCB&StartDate=&EndDate=&PageSize=20&PageIndex=1
  header bắt buộc: ``request-with: XMLHttpRequest`` + User-Agent trình duyệt.
  URL hợp đồng cũ https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx
  hiện 301 sang host mới VÀ LÀM RƠI query string → client thử cả hai URL theo
  thứ tự, URL nào trả dữ liệu hợp lệ thì dùng và ghi nhớ cho các lần sau.
- Response JSON: {"Data": {"Data": [hàng giá...], "TotalCount": ...}, "Message", "Success"}.
- Hàng giá hiện tại dùng key tiếng Việt: ``Ngay`` "28/09/2026" (dd/mm/yyyy),
  ``GiaMoCua``/``GiaCaoNhat``/``GiaThapNhat``/``GiaDongCua``,
  ``KhoiLuongKhopLenh`` (volume khớp lệnh). Parser cũng chấp nhận key tiếng Anh
  theo hợp đồng cũ (PriceOpen/PriceHigh/PriceLow/PriceClose/Volume,
  TradingDate ISO) để chủ động trước khi server đổi trở lại.
- PageSize bị server giới hạn tối đa 20 dòng/trang; phân trang bằng PageIndex
  (mới nhất → cũ nhất); StartDate/EndDate hiện bị server bỏ qua nên client tự
  lọc theo khoảng [start, end] sau khi tải và dừng sớm khi đã đi qua ``start``.
- Giá CAFEF báo theo nghìn đồng (vd. TCB close 32.85 = 32.850 đồng) — chuỗi
  nhất quán nên không ảnh hưởng feature tỷ lệ/lợi nhuận.

Fallback TCBS (dùng khi CAFEF lỗi/đổi hợp đồng):
- GET https://api-finance-tcbs.tcbs.com.vn/stock/ohlc/{symbol}/{start_epoch_giây}/{end_epoch_giây}
  → {"data": [{"open":..., "high":..., "low":..., "close":..., "volume":...,
  "tradingdate":"2026-08-05"}, ...]}
  LƯU Ý (đã xác minh 29/09/2026): DNS api-finance-tcbs.tcbs.com.vn KHÔNG phân
  giải được từ máy này — client vẫn hiện thực đúng hợp đồng, retry đầy đủ và
  báo lỗi rõ khi gọi; fallback chỉ có tác dụng khi DNS/API sống lại.

Danh sách mã HOSE (HoseSymbolDirectory):
- GET https://banggia.cafef.vn/stockhandler.ashx?...curFloorCode=HOSE... trả
  mảng JSON các mã niêm yết (key "a" = mã CK) — đã xác minh 29/09/2026.
- Nguồn này KHÔNG kèm tên công ty/ngành; các nguồn tên/ngành khác (VNDirect
  finfo — DNS trỏ IP nội bộ, TCBS — DNS chết/404, hsx.vn — trang JS rỗng) đều
  không truy cập được từ máy này. Directory chỉ trả cột [symbol] và nêu rõ
  điều đó trong message; TUYỆT ĐỐI không bịa tên/ngành.
"""
from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .base import (
    FetchOutcome,
    OHLCV_COLUMNS,
    PriceDataSource,
    SourceError,
    filter_ohlcv_range,
    normalize_ohlcv,
)

logger = logging.getLogger(__name__)

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
)

#: Thứ tự thử: URL hợp đồng cũ (spec dự án) trước, rồi host mới mà redirect
#: đang trỏ tới (đã xác minh hoạt động 29/09/2026).
CAFEF_HISTORY_URLS: tuple[str, ...] = (
    "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx",
    "https://cafef.vn/du-lieu/ajax/pagenew/datahistory/pricehistory.ashx",
)

CAFEF_HEADERS: dict[str, str] = {
    "request-with": "XMLHttpRequest",
    "User-Agent": BROWSER_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://cafef.vn/",
}

#: Server giới hạn PageSize tối đa 20 dòng/trang (đã xác minh 29/09/2026).
CAFEF_MAX_PAGE_SIZE = 20

TCBS_OHLC_URL = "https://api-finance-tcbs.tcbs.com.vn/stock/ohlc/{symbol}/{start_epoch}/{end_epoch}"
TCBS_HEADERS: dict[str, str] = {"User-Agent": BROWSER_USER_AGENT, "Accept": "application/json"}
TCBS_DEFAULT_START = "2015-01-01"  # khớp config.yaml: data.start_date

BANGGIA_URL = "https://banggia.cafef.vn/stockhandler.ashx"

#: Thư mục data/ của dự án (src/data_sources/cafef_source.py → root/data).
PROJECT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
SYMBOLS_CACHE_PATH = PROJECT_DATA_DIR / "symbols.parquet"

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}$")


def _clean_symbol(symbol: str) -> str:
    """Chuẩn hoá mã (upper, strip) và kiểm tra hình thức hợp lệ."""
    cleaned = str(symbol).strip().upper()
    if not _SYMBOL_RE.match(cleaned):
        raise SourceError(f"Mã không hợp lệ: {symbol!r} (chỉ A–Z/0–9, 2–10 ký tự)")
    return cleaned


def _get_json_with_retry(
    *,
    session: requests.Session,
    url: str,
    params: Mapping[str, Any] | None,
    headers: Mapping[str, str],
    timeout: float,
    max_retries: int,
    sleep_fn: Callable[[float], None],
    label: str,
) -> Any:
    """GET JSON với retry ``max_retries`` lần cho lỗi mạng/5xx/429; lỗi khác báo ngay.

    Sau khi cạn retry → SourceError với thông điệp tổng hợp (label + url + lỗi gốc).
    """
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=dict(params or {}), headers=dict(headers), timeout=timeout)
            if resp.status_code >= 500 or resp.status_code == 429:
                last_error = SourceError(f"HTTP {resp.status_code}")
            elif resp.status_code >= 400:
                raise SourceError(f"{label} HTTP {resp.status_code}: {resp.text[:120]!r}")
            else:
                try:
                    payload = resp.json()
                except ValueError as exc:
                    raise SourceError(f"{label} trả nội dung không phải JSON: {str(exc)[:120]}") from exc
                if not isinstance(payload, (dict, list)):
                    raise SourceError(f"{label} trả JSON không phải object/danh sách")
                return payload
        except requests.RequestException as exc:
            last_error = exc
        if attempt < max_retries:
            sleep_fn(0.5 * attempt)  # backoff tuyến tính
    raise SourceError(f"{label} thất bại sau {max_retries} lần thử ({url}): {last_error!r}")


# ---------------------------------------------------------------------------
# Client CAFEF
# ---------------------------------------------------------------------------

#: Ánh xạ cột chuẩn → các key nguồn (tiếng Việt hiện tại, tiếng Anh hợp đồng cũ).
_CAFE_ROW_FIELDS: dict[str, tuple[str, ...]] = {
    "open": ("GiaMoCua", "PriceOpen"),
    "high": ("GiaCaoNhat", "PriceHigh"),
    "low": ("GiaThapNhat", "PriceLow"),
    "close": ("GiaDongCua", "PriceClose"),
    "volume": ("KhoiLuongKhopLenh", "Volume"),
}


def _coalesce(row: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_cafef_date(raw: Any) -> pd.Timestamp | None:
    """Parse ngày CAFEF: dd/mm/yyyy (hiện tại) hoặc ISO (hợp đồng cũ)."""
    if raw in (None, ""):
        return None
    ts = pd.to_datetime(raw, format="%d/%m/%Y", errors="coerce")
    if pd.isna(ts):
        ts = pd.to_datetime(raw, errors="coerce")
    if pd.isna(ts):
        return None
    ts = ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.normalize()


def _cafef_rows_to_frame(rows: Iterable[Mapping[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        date_raw = _coalesce(row, ("Ngay", "TradingDate", "Date"))
        ts = _parse_cafef_date(date_raw)
        if ts is None:
            continue
        record: dict[str, Any] = {"date": ts}
        for col, keys in _CAFE_ROW_FIELDS.items():
            record[col] = _coalesce(row, keys)
        records.append(record)
    return pd.DataFrame.from_records(records)


class CafefClient:
    """Client CAFEF PriceHistory tự chứa: requests, phân trang, retry 3 lần.

    - Phân trang PageIndex 1..N (mới nhất → cũ nhất), PageSize tối đa 20/trang
      (server giới hạn), dừng khi trang rỗng / trang ngắn hơn PageSize / đã đi
      qua ``start``; chặn tối đa ``max_pages`` trang để tránh vòng lặp.
    - Retry ``max_retries`` lần cho lỗi mạng và HTTP 5xx/429 (backoff 0.5s×lần
      thử) trên MỖI request; cạn retry → SourceError.
    - Thử lần lượt các URL trong ``CAFEF_HISTORY_URLS``; URL đầu tiên trả dữ
      liệu hợp lệ sẽ được ghi nhớ (preferred) cho các lần gọi sau.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        page_size: int = CAFEF_MAX_PAGE_SIZE,
        max_pages: int = 800,
        page_delay: float = 0.15,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= page_size <= CAFEF_MAX_PAGE_SIZE:
            raise ValueError(
                f"page_size phải trong [1, {CAFEF_MAX_PAGE_SIZE}] — server giới hạn "
                f"{CAFEF_MAX_PAGE_SIZE} dòng/trang (đã xác minh 29/09/2026)"
            )
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._page_size = page_size
        self._max_pages = max_pages
        self._page_delay = page_delay
        self._sleep = sleep_fn
        self._preferred_url: str | None = None

    @property
    def preferred_url(self) -> str | None:
        """URL CAFEF gần nhất trả dữ liệu hợp lệ (chẩn đoán/test)."""
        return self._preferred_url

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """Tải toàn bộ trang khả dụng rồi lọc [start, end]; SourceError nếu mọi URL hỏng."""
        symbol = _clean_symbol(symbol)
        urls = self._urls_to_try()
        rows: list[Mapping[str, Any]] = []
        errors: list[str] = []
        for url in urls:
            try:
                rows = self._fetch_all_pages(url, symbol, start, end)
            except SourceError as exc:
                errors.append(f"{url} → {exc}")
                continue
            self._preferred_url = url
            break
        else:
            raise SourceError("CAFEF thất bại trên mọi host đã thử: " + " | ".join(errors))

        frame = normalize_ohlcv(_cafef_rows_to_frame(rows))
        return filter_ohlcv_range(frame, start, end)

    # ---- nội bộ ----
    def _urls_to_try(self) -> tuple[str, ...]:
        if self._preferred_url is not None:
            others = [u for u in CAFEF_HISTORY_URLS if u != self._preferred_url]
            return (self._preferred_url, *others)
        return CAFEF_HISTORY_URLS

    def _fetch_all_pages(
        self,
        url: str,
        symbol: str,
        start: str | None,
        end: str | None,
    ) -> list[Mapping[str, Any]]:
        rows: list[Mapping[str, Any]] = []
        start_ts = pd.Timestamp(start) if start else None
        for page_index in range(1, self._max_pages + 1):
            page = self._fetch_page(url, symbol, page_index, start, end)
            if not page:
                break  # đi quá phiên cũ nhất → hết dữ liệu
            rows.extend(page)
            if start_ts is not None:
                page_dates = [
                    d
                    for d in (_parse_cafef_date(_coalesce(r, ("Ngay", "TradingDate", "Date"))) for r in page)
                    if d is not None
                ]
                if page_dates and min(page_dates) < start_ts:
                    break  # đã đi qua start → các trang sau chỉ cũ hơn
            if len(page) < self._page_size:
                break  # trang cuối
            if self._page_delay:
                self._sleep(self._page_delay)
        else:
            raise SourceError(
                f"CAFEF: vượt quá số trang tối đa {self._max_pages} cho {symbol} — dừng để tránh vòng lặp"
            )
        return rows

    def _fetch_page(
        self,
        url: str,
        symbol: str,
        page_index: int,
        start: str | None,
        end: str | None,
    ) -> list[Mapping[str, Any]]:
        params = {
            "Symbol": symbol,
            "StartDate": start or "",
            "EndDate": end or "",
            "PageSize": self._page_size,
            "PageIndex": page_index,
        }
        payload = _get_json_with_retry(
            session=self._session,
            url=url,
            params=params,
            headers=CAFEF_HEADERS,
            timeout=self._timeout,
            max_retries=self._max_retries,
            sleep_fn=self._sleep,
            label=f"CAFEF PriceHistory {symbol} (trang {page_index})",
        )
        data = (payload.get("Data") or {}).get("Data") if isinstance(payload, dict) else None
        if data is None:
            message = payload.get("Message") if isinstance(payload, dict) else None
            raise SourceError(
                f"phản hồi thiếu Data.Data (trang {page_index}): {message or str(payload)[:160]}"
            )
        if not isinstance(data, list):
            raise SourceError(f"Data.Data không phải danh sách (trang {page_index}): {type(data).__name__}")
        return [row for row in data if isinstance(row, dict)]


# ---------------------------------------------------------------------------
# Client TCBS (fallback)
# ---------------------------------------------------------------------------


def _tcbs_rows_to_frame(rows: Iterable[Any]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("tradingdate") or row.get("tradingDate") or row.get("date")
        if raw_date in (None, ""):
            continue
        records.append(
            {
                "date": raw_date,
                "open": row.get("open"),
                "high": row.get("high"),
                "low": row.get("low"),
                "close": row.get("close"),
                "volume": row.get("volume"),
            }
        )
    return pd.DataFrame.from_records(records)


class TcbsClient:
    """Fallback OHLC từ TCBS theo epoch giây (hợp đồng trong spec dự án).

    Epoch tính theo UTC; ngày mặc định: start 2015-01-01 (khớp config.yaml),
    end = hiện tại. DNS api-finance-tcbs.tcbs.com.vn đã chết từ máy này khi
    xác minh (29/09/2026) — lỗi sẽ được bọc thành SourceError rõ ràng.
    """

    def __init__(
        self,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep_fn

    def fetch(self, symbol: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        symbol = _clean_symbol(symbol)
        start_ts = pd.Timestamp(start, tz="UTC") if start else pd.Timestamp(TCBS_DEFAULT_START, tz="UTC")
        end_ts = pd.Timestamp(end, tz="UTC") if end else pd.Timestamp.now(tz="UTC")
        url = TCBS_OHLC_URL.format(symbol=symbol, start_epoch=int(start_ts.timestamp()), end_epoch=int(end_ts.timestamp()))
        payload = _get_json_with_retry(
            session=self._session,
            url=url,
            params=None,
            headers=TCBS_HEADERS,
            timeout=self._timeout,
            max_retries=self._max_retries,
            sleep_fn=self._sleep,
            label=f"TCBS OHLC {symbol}",
        )
        rows = payload.get("data") if isinstance(payload, dict) else None
        if rows is None:
            raise SourceError(f"TCBS trả phản hồi thiếu 'data' cho {symbol}: {str(payload)[:200]}")
        frame = normalize_ohlcv(_tcbs_rows_to_frame(rows))
        return filter_ohlcv_range(frame, start, end)


# ---------------------------------------------------------------------------
# Nguồn ghép: CAFEF (chính) → TCBS (fallback)
# ---------------------------------------------------------------------------


class CafefSource(PriceDataSource):
    """Nguồn chính CAFEF + fallback TCBS; cả hai fail → khung rỗng + cảnh báo.

    Lớp này KHÔNG raise khi nguồn lỗi: mọi lỗi được gom vào ``warnings`` (và
    log) để pipeline/dashboard luôn nhận được DataFrame hợp lệ (có thể rỗng).
    VN-Index dùng đúng cơ chế này với Symbol=VNINDEX (fallback TCBS "VNINDEX").
    """

    def __init__(self, cafef: CafefClient | None = None, tcbs: TcbsClient | None = None) -> None:
        self.cafef = cafef or CafefClient()
        self.tcbs = tcbs or TcbsClient()

    @property
    def name(self) -> str:
        return "cafef+tcbs"

    def fetch_outcome(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> FetchOutcome:
        symbol = _clean_symbol(symbol)
        warnings: list[str] = []

        try:
            frame = self.cafef.fetch(symbol, start=start, end=end)
            if not frame.empty:
                return FetchOutcome(frame=frame, source="cafef", warnings=warnings)
            warnings.append(f"CAFEF trả 0 phiên cho {symbol} trong khoảng [{start}, {end}]")
        except SourceError as exc:
            warnings.append(f"CAFEF lỗi cho {symbol}: {exc}")

        try:
            frame = self.tcbs.fetch(symbol, start=start, end=end)
            if not frame.empty:
                for warning in warnings:
                    logger.warning("%s — chuyển fallback TCBS", warning)
                return FetchOutcome(frame=frame, source="tcbs", warnings=warnings)
            warnings.append(f"TCBS trả 0 phiên cho {symbol} trong khoảng [{start}, {end}]")
        except SourceError as exc:
            warnings.append(f"TCBS lỗi cho {symbol}: {exc}")

        warnings.append(
            f"Cả CAFEF và TCBS đều không có dữ liệu cho {symbol} [{start}, {end}] — "
            "trả khung rỗng; pipeline/dashboard phải xử lý được tình trạng này."
        )
        for warning in warnings:
            logger.warning("%s", warning)
        return FetchOutcome(frame=normalize_ohlcv(None), source="none", warnings=warnings)

    def fetch_vnindex(self, start: str | None = None, end: str | None = None) -> pd.DataFrame:
        """VN-Index: cùng cơ chế với Symbol=VNINDEX (fallback TCBS "VNINDEX")."""
        return self.fetch_ohlcv("VNINDEX", start=start, end=end)

    def fetch_vnindex_outcome(self, start: str | None = None, end: str | None = None) -> FetchOutcome:
        return self.fetch_outcome("VNINDEX", start=start, end=end)


# ---------------------------------------------------------------------------
# Danh sách mã HOSE
# ---------------------------------------------------------------------------


def _empty_symbols_frame() -> pd.DataFrame:
    return pd.DataFrame({"symbol": pd.Series(dtype="string")})


def normalize_symbol_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Chuẩn hoá frame danh sách mã: cột [symbol] (+ name/industry nếu có thật).

    - symbol: strip/upper, khớp ^[A-Z0-9]{2,10}$, dedupe, sort.
    - Cột name/industry chỉ giữ lại nếu nguồn cung cấp ít nhất một giá trị thật.
    - Thiếu cột symbol → ValueError (nguồn sai hợp đồng).
    """
    if frame is None or len(frame) == 0:
        return _empty_symbols_frame()
    df = frame.copy()
    df.columns = [str(col).strip().lower() for col in df.columns]
    if "symbol" not in df.columns:
        raise ValueError(f"Nguồn danh sách mã không có cột 'symbol' (các cột: {list(df.columns)})")
    df["symbol"] = df["symbol"].astype("string").str.strip().str.upper()
    df = df.loc[df["symbol"].notna() & df["symbol"].str.match(r"^[A-Z0-9]{2,10}$", na=False)]
    df = df.drop_duplicates(subset="symbol").sort_values("symbol", kind="stable").reset_index(drop=True)

    keep = ["symbol"]
    for col in ("name", "industry"):
        if col in df.columns and df[col].notna().any():
            keep.append(col)
    out = df.loc[:, keep].copy()
    for col in keep[1:]:
        out[col] = out[col].astype("string").str.strip()
    return out if not out.empty else _empty_symbols_frame()


@dataclass
class SymbolsResult:
    """Kết quả tra danh sách mã: frame + nguồn + message (khi rỗng/dùng cache cũ)."""

    frame: pd.DataFrame
    source: str | None = None
    message: str | None = None

    @property
    def symbols(self) -> list[str]:
        if self.frame.empty:
            return []
        return self.frame["symbol"].astype(str).tolist()


class HoseSymbolDirectory:
    """Danh sách mã HOSE scrape từ nguồn công khai, cache data/symbols.parquet.

    Nguồn mặc định: banggia.cafef.vn (mảng JSON các mã, key "a"). Nguồn này
    chỉ có mã — KHÔNG có tên công ty/ngành; các nguồn tên/ngành khác đều
    không truy cập được từ máy này khi xác minh (29/09/2026), nên kết quả chỉ
    chứa [symbol] kèm message nêu rõ. Không có nguồn nào (hoặc toàn lỗi) →
    trả frame rỗng + message rõ (dashboard cần cho phép nhập mã thủ công);
    nếu có cache cũ thì trả cache cũ + message, không bịa dữ liệu.
    """

    def __init__(
        self,
        cache_path: Path | str = SYMBOLS_CACHE_PATH,
        session: requests.Session | None = None,
        timeout: float = 15.0,
        max_retries: int = 3,
        sleep_fn: Callable[[float], None] = time.sleep,
        max_age_days: float = 7.0,
        providers: Iterable[Callable[[], pd.DataFrame]] | None = None,
    ) -> None:
        self._cache_path = Path(cache_path)
        self._session = session or requests.Session()
        self._timeout = timeout
        self._max_retries = max_retries
        self._sleep = sleep_fn
        self._max_age_days = max_age_days
        self._providers = list(providers) if providers is not None else [self._fetch_banggia_hose]

    def get(self, refresh: bool = False) -> SymbolsResult:
        """Tra danh sách mã: cache tươi → trả ngay; ngược lại chạy nguồn, cache kết quả."""
        if not refresh:
            cached = self._read_cache()
            if cached is not None and not cached.empty and self._cache_is_fresh():
                return SymbolsResult(frame=cached, source="cache")

        errors: list[str] = []
        for provider in self._providers:
            name = getattr(provider, "__name__", "provider")
            try:
                frame = normalize_symbol_frame(provider())
            except Exception as exc:  # noqa: BLE001 — nguồn scrape có thể fail đủ kiểu
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                continue
            if not frame.empty:
                self._write_cache(frame)
                return SymbolsResult(frame=frame, source=name)

        cached = self._read_cache()
        if cached is not None and not cached.empty:
            message = (
                "Không tải được danh sách mã từ nguồn công khai ("
                + "; ".join(errors)
                + f") — dùng cache cũ tại {self._cache_path.name}."
            )
            logger.warning("%s", message)
            return SymbolsResult(frame=cached, source="cache", message=message)

        message = (
            "Không có nguồn công khai nào cung cấp danh sách mã HOSE ("
            + "; ".join(errors)
            + ") — trả danh sách rỗng; dashboard cần cho phép nhập mã thủ công."
        )
        logger.warning("%s", message)
        return SymbolsResult(frame=_empty_symbols_frame(), source=None, message=message)

    # ---- nội bộ ----
    def _cache_is_fresh(self) -> bool:
        try:
            age_seconds = time.time() - self._cache_path.stat().st_mtime
        except OSError:
            return False
        return age_seconds <= self._max_age_days * 86400.0

    def _read_cache(self) -> pd.DataFrame | None:
        if not self._cache_path.exists():
            return None
        try:
            return pd.read_parquet(self._cache_path)
        except Exception as exc:  # noqa: BLE001 — cache hỏng thì coi như không có
            logger.warning("Đọc cache symbols lỗi (%s): %s", self._cache_path, exc)
            return None

    def _write_cache(self, frame: pd.DataFrame) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(self._cache_path)

    def _fetch_banggia_hose(self) -> pd.DataFrame:
        """Scrape bảng giá CAFEF — trả về toàn bộ mã HOSE (key 'a' = mã CK)."""
        params = {
            "center": 1,
            "index": "false",
            "curFloorCode": "HOSE",
            "ColumnWhitelist": "",
            "Term": "",
            "pageSize": 1000,
            "orderBy": "Code",
            "orderDir": "ASC",
            "total": 0,
            "offset": 0,
            "limit": 1000,
        }
        payload = _get_json_with_retry(
            session=self._session,
            url=BANGGIA_URL,
            params=params,
            headers=CAFEF_HEADERS,
            timeout=self._timeout,
            max_retries=self._max_retries,
            sleep_fn=self._sleep,
            label="CAFEF banggia (danh sách mã HOSE)",
        )
        rows = payload if isinstance(payload, list) else (payload.get("data") if isinstance(payload, dict) else None)
        if rows is None:
            raise SourceError(f"banggia trả phản hồi không có danh sách mã: {str(payload)[:200]}")
        records: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            records.append(
                {
                    "symbol": _coalesce(row, ("a", "Code", "Symbol")),
                    "name": _coalesce(row, ("name", "Name", "CompanyName")),
                    "industry": _coalesce(row, ("industry", "Industry", "Nganh")),
                }
            )
        return pd.DataFrame.from_records(records)


__all__ = [
    "BANGGIA_URL",
    "BROWSER_USER_AGENT",
    "CAFEF_HISTORY_URLS",
    "CAFEF_HEADERS",
    "CAFEF_MAX_PAGE_SIZE",
    "CafefClient",
    "CafefSource",
    "HoseSymbolDirectory",
    "OHLCV_COLUMNS",
    "SymbolsResult",
    "TcbsClient",
    "TCBS_OHLC_URL",
]
