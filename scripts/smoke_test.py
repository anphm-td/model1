#!/usr/bin/env python
"""Smoke test nguồn dữ liệu CAFEF PriceHistory — kiểm chứng hợp đồng trực tiếp.

Tự chứa (chỉ dùng requests + thư viện chuẩn), KHÔNG phụ thuộc vnquant/vnstock.

Việc cần làm: tải 30 phiên giao dịch gần nhất của mã TCB qua CAFEF
PriceHistory theo hợp đồng đã xác minh:

    GET https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx
        ?Symbol=TCB&StartDate=&EndDate=&PageSize=20&PageIndex=1
    Header: "request-with: XMLHttpRequest" + User-Agent trình duyệt
    Response JSON: {"Data": {"Data": [<các phiên>]}}
    Phiên (hợp đồng gốc): {"PriceOpen", "PriceHigh", "PriceLow", "PriceClose",
        "Volume", "TradingDate": "2026-08-05T00:00:00"}

Hai điểm khác biệt so với hợp đồng gốc được xác minh trực tiếp từ máy này
ngày 2026-09-29 và được xử lý ở dưới (cả hai đều báo trong output):

1. Host hợp đồng 301-redirect về https://cafef.vn/du-lieu/ajax/pagenew/
   datahistory/pricehistory.ashx và DROP hết query string — phải gửi lại
   các tham số (Symbol/StartDate/EndDate/PageSize/PageIndex) cho đích redirect.
2. API hiện trả tên trường tiếng Việt trong cùng cấu trúc {"Data": {"Data"}}:
   {"Ngay": "dd/mm/yyyy", "GiaMoCua", "GiaCaoNhat", "GiaThapNhat",
    "GiaDongCua", "KhoiLuongKhopLenh"}. Parser nhận CẢ biến thể tiếng Anh
   (hợp đồng) và biến thể tiếng Việt (đang chạy).

- Phân trang qua PageSize/PageIndex (PageSize bị giới hạn 20 dòng/trang nên
  cần 30 phiên phải tải ≥ 2 trang); dừng khi trang trả về ít dòng hơn PageSize.
- Retry 3 lần khi lỗi mạng (mỗi lần cách nhau 2 giây); lỗi API báo rõ.
- API trả phiên mới nhất trước; "5 dòng đầu" là 5 dòng đầu của dữ liệu tải về
  (tức 5 phiên gần nhất).

Mã thoát:
- 0 khi tải đủ dữ liệu và in được 5 dòng đầu.
- 1 kèm thông báo rõ ràng khi lỗi mạng/API (không in traceback dài).

Cách chạy: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from typing import Any, TypedDict
from urllib.parse import urljoin

try:
    import requests
except ImportError:  # pragma: no cover — môi trường chuẩn của dự án luôn có requests
    print(
        "LỖI SMOKE TEST: thư viện 'requests' chưa được cài "
        "(requests đã có sẵn trong Anaconda — kiểm tra lại môi trường python).",
        file=sys.stderr,
    )
    sys.exit(1)

TICKER = "TCB"
SESSIONS = 30          # số phiên gần nhất cần tải
PAGE_SIZE = 20         # theo hợp đồng CAFEF; API hiện giới hạn tối đa 20 dòng/trang
MAX_PAGES = 5          # chặn vòng lặp: 5 trang × 20 dòng > 30 phiên cần
MAX_REDIRECTS = 5      # chặn vòng lặp redirect
RETRY_ATTEMPTS = 3     # số lần thử lại khi lỗi mạng (theo hợp đồng nguồn dữ liệu)
RETRY_SLEEP_SECONDS = 2.0
HTTP_TIMEOUT_SECONDS = 20.0

# URL hợp đồng (điểm vào) — host này 301 về cafef.vn/du-lieu/... (xem docstring)
CAFEF_URL = "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"

# Hợp đồng CAFEF yêu cầu header AJAX + User-Agent trình duyệt.
CAFEF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "request-with": "XMLHttpRequest",
}

# Hai biến thể tên trường của một phiên: (ngày, mở, cao, thấp, đóng, khối lượng).
# 1) Tiếng Anh — hợp đồng gốc; 2) Tiếng Việt — API đang phục vụ (2026-09-29).
ROW_FIELD_VARIANTS: tuple[tuple[str, str, str, str, str, str], ...] = (
    ("TradingDate", "PriceOpen", "PriceHigh", "PriceLow", "PriceClose", "Volume"),
    ("Ngay", "GiaMoCua", "GiaCaoNhat", "GiaThapNhat", "GiaDongCua", "KhoiLuongKhopLenh"),
)


class PriceRow(TypedDict):
    """Một phiên giá đã chuẩn hoá (ngày ISO "YYYY-MM-DD", giá float)."""

    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def _get_with_retry(url: str, params: dict[str, Any]) -> "requests.Response":
    """GET (có đi qua redirect, gửi lại params ở từng bước) — retry khi lỗi mạng.

    - requests tự theo redirect nhưng redirect của CAFEF DROP query string, nên
      tắt auto-redirect và tự bước qua từng hop, luôn gắn lại `params`.
    - Retry RETRY_ATTEMPTS lần cho lỗi mạng (RequestException). Lỗi HTTP status
      là lỗi API — không retry, báo rõ qua exception của caller.
    """
    current = url
    last_error: Exception | None = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            for _ in range(MAX_REDIRECTS + 1):
                response = requests.get(
                    current,
                    params=params,
                    headers=CAFEF_HEADERS,
                    timeout=HTTP_TIMEOUT_SECONDS,
                    allow_redirects=False,
                )
                if response.is_redirect:
                    # Bỏ query của đích redirect (nếu có) — params luôn gắn lại.
                    current = urljoin(current, response.headers["Location"]).split("?")[0]
                    continue
                return response
            raise RuntimeError(f"quá {MAX_REDIRECTS} lần redirect: {current}")
        except requests.RequestException as err:
            last_error = err
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_SLEEP_SECONDS)
    raise RuntimeError(
        f"lỗi mạng sau {RETRY_ATTEMPTS} lần thử: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error


def _get_json_with_retry(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET và trả về dict JSON; lỗi HTTP/không-JSON báo rõ bằng RuntimeError."""
    response = _get_with_retry(url, params)
    if response.status_code != 200:
        raise RuntimeError(f"CAFEF trả về HTTP {response.status_code}")
    try:
        payload: Any = response.json()
    except ValueError as err:
        raise RuntimeError(
            f"CAFEF trả về nội dung không phải JSON (HTTP {response.status_code})"
        ) from err
    if not isinstance(payload, dict):
        raise RuntimeError(f"CAFEF trả về JSON không phải object: {type(payload).__name__}")
    return payload


def _normalize_date(raw: str) -> str:
    """Chuẩn hoá ngày về ISO "YYYY-MM-DD".

    Hợp đồng gốc: "2026-08-05T00:00:00"; biến thể hiện tại: "dd/mm/yyyy".
    """
    raw = raw.strip()
    if "T" in raw:  # ISO có thời gian
        return raw.split("T", 1)[0]
    if "/" in raw:  # dd/mm/yyyy
        day, month, year = raw.split("/")[:3]
        return f"{year}-{int(month):02d}-{int(day):02d}"
    return raw[:10]  # đã là YYYY-MM-DD


def _parse_row(item: dict[str, Any]) -> PriceRow:
    """Parse một phiên theo một trong hai biến thể tên trường (xem ROW_FIELD_VARIANTS)."""
    errors: list[str] = []
    for date_key, open_key, high_key, low_key, close_key, volume_key in ROW_FIELD_VARIANTS:
        if date_key not in item or close_key not in item:
            continue
        try:
            return PriceRow(
                date=_normalize_date(str(item[date_key])),
                open=float(item[open_key]),
                high=float(item[high_key]),
                low=float(item[low_key]),
                close=float(item[close_key]),
                volume=float(item.get(volume_key) or 0),
            )
        except (KeyError, TypeError, ValueError) as err:
            errors.append(f"{type(err).__name__}: {err}")
    raise RuntimeError(
        "Dòng dữ liệu CAFEF không khớp biến thể trường nào đã biết "
        f"(keys={sorted(item.keys())}); chi tiết: {'; '.join(errors) or 'thiếu trường giá/ngày'}"
    )


def _parse_rows(payload: dict[str, Any]) -> list[PriceRow]:
    """Trích {"Data": {"Data": [...]}} thành danh sách phiên đã chuẩn hoá."""
    data_block = payload.get("Data")
    data = data_block.get("Data") if isinstance(data_block, dict) else None
    if not data:
        message = payload.get("Message") if isinstance(payload, dict) else None
        raise RuntimeError(f"CAFEF không trả dữ liệu: {message or 'Data rỗng'}")
    return [_parse_row(item) for item in data]


def fetch_recent_sessions(symbol: str, sessions: int = SESSIONS) -> list[PriceRow]:
    """Tải `sessions` phiên gần nhất của `symbol`, phân trang qua PageSize/PageIndex.

    Trả về các phiên theo đúng thứ tự API trả về (mới nhất trước), đã khử
    trùng lặp theo ngày.
    """
    all_rows: list[PriceRow] = []
    seen_dates: set[str] = set()
    page_index = 1
    while len(all_rows) < sessions and page_index <= MAX_PAGES:
        payload = _get_json_with_retry(
            CAFEF_URL,
            {
                "Symbol": symbol,
                "StartDate": "",
                "EndDate": "",
                "PageSize": PAGE_SIZE,
                "PageIndex": page_index,
            },
        )
        rows = _parse_rows(payload)
        for row in rows:
            if row["date"] not in seen_dates:
                seen_dates.add(row["date"])
                all_rows.append(row)
        if len(rows) < PAGE_SIZE:  # trang cuối — hết dữ liệu
            break
        page_index += 1
    return all_rows[:sessions]  # trang cuối có thể dư — cắt đúng số phiên cần


def _print_rows(rows: list[PriceRow]) -> None:
    """In bảng giá cho người dùng đọc nhanh."""
    header = (
        f"{'Ngày':<12}{'Open':>10}{'High':>10}{'Low':>10}{'Close':>10}{'Volume':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['date']:<12}{row['open']:>10.2f}{row['high']:>10.2f}"
            f"{row['low']:>10.2f}{row['close']:>10.2f}{row['volume']:>14.0f}"
        )


def main() -> int:
    print(f"[smoke test] nguồn: CAFEF PriceHistory — mã {TICKER}, {SESSIONS} phiên gần nhất")
    try:
        rows = fetch_recent_sessions(TICKER)
    except (RuntimeError, requests.RequestException) as err:
        print(f"LỖI SMOKE TEST: không tải được dữ liệu {TICKER} từ CAFEF — {err}", file=sys.stderr)
        print(
            "Gợi ý: kiểm tra kết nối mạng; nếu CAFEF đổi hợp đồng API (redirect, "
            "tên trường, phân trang), so lại với các biến thể trong "
            "ROW_FIELD_VARIANTS và xử lý redirect trong _get_with_retry.",
            file=sys.stderr,
        )
        return 1

    if len(rows) < SESSIONS:
        print(
            f"CẢNH BÁO: chỉ lấy được {len(rows)} phiên < {SESSIONS} "
            "(API trả ít dữ liệu hơn yêu cầu).",
            file=sys.stderr,
        )
    print(
        f"Thành công: {len(rows)} phiên gần nhất "
        f"({rows[-1]['date']} → {rows[0]['date']}, mới nhất trước)."
    )
    print(f"\n--- {TICKER}: 5 dòng đầu của dữ liệu tải về ---")
    _print_rows(rows[:5])
    return 0


if __name__ == "__main__":
    sys.exit(main())
