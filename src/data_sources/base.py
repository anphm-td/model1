"""Interface PriceDataSource — hợp đồng chung cho mọi nguồn dữ liệu giá.

Hợp đồng khung OHLCV chuẩn mà mọi nguồn phải trả về:
- index: DatetimeIndex naive (đã bỏ múi giờ, chuẩn hoá 00:00), tên ``date``,
  sắp tăng dần, không trùng ngày;
- cột: ``open, high, low, close, volume`` (float64);
- CHỈ chứa các phiên giao dịch thực tế — KHÔNG ffill ngày nghỉ/lễ (ffill tạo
  ngày "phẳng" giả và model học sai; với HOSE ngày lễ nhiều nên càng nguy hiểm);
- giá theo đúng đơn vị nguồn trả (CAFEF báo theo nghìn đồng), volume theo cổ
  phiếu khớp lệnh.

Quy ước lỗi:
- Tầng client thấp (CafefClient, TcbsClient) phải raise SourceError với thông
  điệp rõ ràng (đã thử gì, lỗi gốc) — không được trả khung rỗng một cách im lặng.
- Tầng ghép (CafefSource) mới là nơi biến lỗi thành khung rỗng + cảnh báo để
  pipeline/dashboard không chết.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)

#: Cột bắt buộc của khung OHLCV chuẩn, đúng thứ tự này.
OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")

#: Tên cột ngày được chấp nhận khi chuẩn hoá.
_DATE_COLUMNS: tuple[str, ...] = ("date", "time", "tradingdate", "ngay")


class SourceError(RuntimeError):
    """Nguồn dữ liệu lỗi: mạng, phản hồi sai hợp đồng, dữ liệu không hợp lệ."""


@dataclass
class FetchOutcome:
    """Kết quả một lần tải: khung dữ liệu + nguồn thực dùng + các cảnh báo."""

    frame: pd.DataFrame
    source: str  # "cafef" | "tcbs" | "none" | tên nguồn khác
    warnings: list[str] = field(default_factory=list)


class PriceDataSource(abc.ABC):
    """Nguồn giá OHLCV theo ngày cho cổ phiếu/chỉ số (HOSE)."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Tên nguồn, dùng trong log và provenance."""

    @abc.abstractmethod
    def fetch_outcome(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> FetchOutcome:
        """Tải OHLCV theo ngày cho ``symbol`` trong khoảng [start, end].

        Tham số ``start``/``end`` là chuỗi "YYYY-MM-DD" (hoặc None = không chặn
        hai đầu). Trả về FetchOutcome: khi mọi nguồn trong hiện thực đều lỗi,
        trả khung rỗng + warnings (không raise) để tầng gọi tự xử lý.
        """

    def fetch_ohlcv(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        """Tiện ích: chỉ lấy khung OHLCV (bỏ provenance/cảnh báo)."""
        return self.fetch_outcome(symbol, start=start, end=end).frame


def _empty_ohlcv() -> pd.DataFrame:
    """Khung OHLCV rỗng nhưng đúng schema (index ngày, cột float64)."""
    index = pd.DatetimeIndex([], name="date")
    data = {col: pd.Series(dtype="float64") for col in OHLCV_COLUMNS}
    return pd.DataFrame(data, index=index, columns=list(OHLCV_COLUMNS))


def normalize_ohlcv(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Chuẩn hoá khung OHLCV về hợp đồng chung (xem docstring module).

    - Tìm cột ngày trong (date/time/tradingdate/ngay) hoặc dùng index nếu là
      DatetimeIndex; bỏ múi giờ nếu có; chuẩn hoá về mốc 00:00.
    - Ép kiểu số; sort tăng dần (ổn định); trùng ngày giữ bản ghi sau cùng;
      bỏ phiên có close rỗng/<=0 (không ffill ngày nghỉ — không bịa dữ liệu).
    - Khung rỗng đầu vào → trả khung rỗng đúng schema.
    - Thiếu cột bắt buộc → SourceError.
    """
    if frame is None or len(frame) == 0:
        return _empty_ohlcv()

    df = frame.copy()
    lower_cols = {str(col).strip().lower(): col for col in df.columns}
    date_col = next((lower_cols[c] for c in _DATE_COLUMNS if c in lower_cols), None)

    if date_col is not None:
        dates = pd.to_datetime(df[date_col], errors="coerce")
        if dates.isna().any():
            # pandas 2.x suy luận format từ phần tử đầu và áp dụng nghiêm ngặt:
            # cột ngày trộn định dạng (vd. "2025-01-03" + "2025-01-06T00:00:00")
            # phải được parse lại phần lỗi với format="mixed".
            retry = pd.to_datetime(df.loc[dates.isna(), date_col], errors="coerce", format="mixed")
            dates = dates.copy()
            dates.loc[retry.index] = retry
    elif isinstance(df.index, pd.DatetimeIndex):
        dates = pd.Series(df.index)
    else:
        raise SourceError(
            "Chuẩn hoá OHLCV: không tìm thấy cột ngày "
            f"{_DATE_COLUMNS} và index không phải DatetimeIndex (các cột: {list(df.columns)})"
        )

    if getattr(dates.dtype, "tz", None) is not None:
        dates = dates.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None)
    index = pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce")).normalize()

    data: dict[str, object] = {}
    for col in OHLCV_COLUMNS:
        src = lower_cols.get(col)
        if src is None:
            raise SourceError(
                f"Chuẩn hoá OHLCV: thiếu cột bắt buộc '{col}' (các cột hiện có: {list(df.columns)})"
            )
        data[col] = pd.to_numeric(df[src].to_numpy(), errors="coerce")

    out = pd.DataFrame(data, index=index, columns=list(OHLCV_COLUMNS))
    unparseable = out.index.isna()
    if unparseable.any():
        logger.debug("normalize_ohlcv: bỏ %d dòng không parse được ngày", int(unparseable.sum()))
        out = out.loc[~unparseable]
    out = out.sort_index(kind="stable")
    out = out.loc[~out.index.duplicated(keep="last")]

    invalid = out["close"].isna() | (out["close"] <= 0)
    if invalid.any():
        logger.debug("normalize_ohlcv: bỏ %d phiên có close không hợp lệ", int(invalid.sum()))
        out = out.loc[~invalid]
    return out.astype("float64")


def filter_ohlcv_range(
    frame: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Lọc khung OHLCV theo khoảng [start, end] — bao cả hai đầu (ngày lịch)."""
    if frame.empty or (start is None and end is None):
        return frame
    mask = pd.Series(True, index=frame.index)
    if start is not None:
        mask &= frame.index >= pd.Timestamp(start)
    if end is not None:
        mask &= frame.index <= pd.Timestamp(end)
    return frame.loc[mask]
