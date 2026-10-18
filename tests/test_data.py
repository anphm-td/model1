"""Kiểm thử tầng dữ liệu (src/data_sources, src/data_loader, src/watchlist, src/db).

Toàn bộ OFFLINE: phần gọi mạng được mock bằng session/nguồn giả (không có
request thật), SQLite dùng tệp tạm trong tmp_path, cache parquet ghi vào
thư mục tạm — không đụng tới data/ thật của dự án.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data_loader, db
from src.data_sources.base import (
    OHLCV_COLUMNS,
    FetchOutcome,
    PriceDataSource,
    SourceError,
    filter_ohlcv_range,
    normalize_ohlcv,
)
from src.data_sources.cafef_source import (
    CafefClient,
    CafefSource,
    HoseSymbolDirectory,
    TcbsClient,
    normalize_symbol_frame,
)
from src.data_sources.composite import PriorityMergeSource
from src.data_sources.hose_source import HoseOfficialSource
from src.watchlist import Watchlist, WatchlistError

NO_SLEEP = lambda _seconds: None  # noqa: E731


# ---------------------------------------------------------------------------
# Giả lập HTTP / nguồn dữ liệu
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = "") -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text if text else (json.dumps(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json body")
        return self._payload


class FakeSession:
    """Session giả: ``handler(url, params)`` → FakeResponse hoặc raise Exception."""

    def __init__(self, handler) -> None:
        self.handler = handler
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": dict(headers or {})})
        result = self.handler(url, params or {})
        if isinstance(result, Exception):
            raise result
        return result


class ScriptedSource(PriceDataSource):
    """Nguồn giả theo kịch bản: mỗi lần fetch trả FetchOutcome kế tiếp; ghi lại lời gọi."""

    def __init__(self, outcomes: list[FetchOutcome]) -> None:
        self.script = list(outcomes)
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return "scripted"

    def fetch_outcome(self, symbol, start=None, end=None) -> FetchOutcome:
        self.calls.append({"symbol": symbol, "start": start, "end": end})
        if self.script:
            return self.script.pop(0)
        return FetchOutcome(frame=normalize_ohlcv(None), source="scripted", warnings=["hết kịch bản"])


def ohlcv_frame(dates, closes) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(dates), name="date")
    return pd.DataFrame(
        {
            "open": [c - 0.1 for c in closes],
            "high": [c + 0.5 for c in closes],
            "low": [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * len(closes),
        },
        index=index,
    )


def cafef_row(ngay: str, open_, high, low, close, volume: int = 10_000) -> dict:
    """Một hàng giá theo hợp đồng CAFEF hiện tại (key tiếng Việt, Ngay dd/mm/yyyy)."""
    return {
        "Symbol": "TCB",
        "Ngay": ngay,
        "GiaDieuChinh": close,
        "GiaDongCua": close,
        "ThayDoi": "+0,00 (+0,00%)",
        "KhoiLuongKhopLenh": volume,
        "GiaTriKhopLenh": 1.0,
        "KLThoaThuan": 0,
        "GtThoaThuan": 0,
        "GiaMoCua": open_,
        "GiaCaoNhat": high,
        "GiaThapNhat": low,
    }


def cafef_ok_session(pages: dict[int, list[dict]]):
    def handler(url, params):
        return FakeResponse(
            200,
            {"Data": {"Data": pages[params["PageIndex"]], "TotalCount": 3}, "Message": "", "Success": True},
        )

    return FakeSession(handler)


# ---------------------------------------------------------------------------
# normalize_ohlcv / filter
# ---------------------------------------------------------------------------


def test_normalize_ohlcv_enforces_contract():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-03", "2025-01-02", "2025-01-02", "2025-01-06T00:00:00"],
            "open": [2.0, 1.0, 1.0, 3.0],
            "high": [3.0, 2.0, 2.0, 4.0],
            "low": [1.0, 0.5, 0.5, 2.5],
            "close": [2.5, 1.5, 1.6, 3.5],
            "volume": [10, 20, 20, 30],
            "extra": ["x"] * 4,
        }
    )
    out = normalize_ohlcv(raw)
    assert list(out.columns) == list(OHLCV_COLUMNS)  # cột thừa bị bỏ, đúng thứ tự
    assert out.index.name == "date"
    assert out.index.is_monotonic_increasing and out.index.is_unique
    assert list(out.index) == list(pd.to_datetime(["2025-01-02", "2025-01-03", "2025-01-06"]))
    assert out["close"].tolist() == [1.6, 2.5, 3.5]  # trùng ngày → giữ bản sau
    assert (out.dtypes == "float64").all()


def test_normalize_ohlcv_drops_invalid_close_and_rejects_missing_columns():
    raw = pd.DataFrame(
        {
            "date": ["2025-01-02", "2025-01-03"],
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.0],
            "close": [1.5, 0.0],  # close = 0 → phiên rác, phải bỏ
            "volume": [10, 20],
        }
    )
    out = normalize_ohlcv(raw)
    assert list(out.index) == list(pd.to_datetime(["2025-01-02"]))

    with pytest.raises(SourceError, match="thiếu cột bắt buộc 'low'"):
        normalize_ohlcv(raw.drop(columns=["low"]))


def test_normalize_ohlcv_empty_and_tz():
    empty = normalize_ohlcv(None)
    assert empty.empty and list(empty.columns) == list(OHLCV_COLUMNS)
    raw = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-02", "2025-01-03"], utc=True),
            "open": [1, 2],
            "high": [2, 3],
            "low": [0.5, 1],
            "close": [1.5, 2.5],
            "volume": [10, 20],
        }
    )
    out = normalize_ohlcv(raw)
    assert out.index.tz is None  # bỏ múi giờ (giá trị ngày)


def test_filter_ohlcv_range_inclusive_both_ends():
    frame = ohlcv_frame(pd.bdate_range("2025-01-02", periods=5), [1, 2, 3, 4, 5])
    out = filter_ohlcv_range(frame, "2025-01-03", "2025-01-07")
    assert list(out.index) == list(pd.to_datetime(["2025-01-03", "2025-01-06", "2025-01-07"]))


# ---------------------------------------------------------------------------
# CafefClient — hợp đồng CAFEF thật (mock HTTP)
# ---------------------------------------------------------------------------


def test_cafef_client_parses_live_contract_and_paginates():
    pages = {
        1: [cafef_row("28/09/2026", 33.2, 33.25, 32.4, 32.85), cafef_row("25/09/2026", 33.0, 33.4, 32.9, 33.1)],
        2: [cafef_row("24/09/2026", 33.0, 33.5, 32.8, 33.2)],
    }
    session = cafef_ok_session(pages)
    client = CafefClient(session=session, page_size=2, page_delay=0.0, sleep_fn=NO_SLEEP)
    frame = client.fetch("tcb", start="2026-09-24", end="2026-09-28")

    assert list(frame.columns) == list(OHLCV_COLUMNS)
    assert [d.strftime("%Y-%m-%d") for d in frame.index] == ["2026-09-24", "2026-09-25", "2026-09-28"]
    assert frame["close"].tolist() == [33.2, 33.1, 32.85]
    assert frame["volume"].tolist() == [10_000.0, 10_000.0, 10_000.0]

    # đúng hợp đồng: URL host cũ được thử đầu, header + tham số phân trang
    first = session.calls[0]
    assert first["url"] == "https://s.cafef.vn/Ajax/PageNew/DataHistory/PriceHistory.ashx"
    assert first["headers"]["request-with"] == "XMLHttpRequest"
    assert "User-Agent" in first["headers"]
    assert first["params"]["Symbol"] == "TCB"
    assert first["params"]["PageSize"] == 2
    assert first["params"]["PageIndex"] == 1
    assert session.calls[1]["params"]["PageIndex"] == 2  # trang 2 vì trang 1 đầy


def test_cafef_client_accepts_legacy_english_contract():
    row = {
        "PriceOpen": 33.2,
        "PriceHigh": 33.25,
        "PriceLow": 32.4,
        "PriceClose": 32.85,
        "Volume": 21_588_800,
        "TradingDate": "2026-09-28T00:00:00",
    }
    session = FakeSession(
        lambda url, params: FakeResponse(200, {"Data": {"Data": [row]}, "Success": True})
    )
    client = CafefClient(session=session, sleep_fn=NO_SLEEP, page_delay=0.0)
    frame = client.fetch("TCB")
    assert list(frame.index) == [pd.Timestamp("2026-09-28")]
    assert frame.iloc[0]["close"] == 32.85
    assert frame.iloc[0]["volume"] == 21_588_800


def test_cafef_client_stops_early_when_passing_start():
    pages = {
        1: [cafef_row("28/09/2026", 1, 2, 0.5, 1.9), cafef_row("27/09/2026", 1, 2, 0.5, 1.9)],
        2: [cafef_row("26/09/2026", 1, 2, 0.5, 1.9), cafef_row("25/09/2026", 1, 2, 0.5, 1.9)],
    }
    session = cafef_ok_session(pages)
    client = CafefClient(session=session, page_size=2, page_delay=0.0, sleep_fn=NO_SLEEP)
    frame = client.fetch("TCB", start="2026-09-26")
    # trang 2 chứa ngày 25/09 (< start) → dừng ngay, không gọi trang 3
    assert len(session.calls) == 2
    assert [d.strftime("%Y-%m-%d") for d in frame.index] == ["2026-09-26", "2026-09-27", "2026-09-28"]


def test_cafef_client_retries_on_network_error_then_succeeds():
    attempts = {"n": 0}

    def handler(url, params):
        attempts["n"] += 1
        if attempts["n"] <= 2:
            raise requests.ConnectionError("mạng đứt (giả lập)")
        return FakeResponse(200, {"Data": {"Data": [cafef_row("28/09/2026", 1, 2, 0.5, 1.9)]}, "Success": True})

    client = CafefClient(session=FakeSession(handler), sleep_fn=NO_SLEEP, page_delay=0.0)
    frame = client.fetch("TCB")
    assert attempts["n"] == 3  # 2 lỗi + 1 thành công
    assert len(frame) == 1


def test_cafef_client_retries_on_5xx():
    attempts = {"n": 0}

    def handler(url, params):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return FakeResponse(503, {"nothing": 1})
        return FakeResponse(200, {"Data": {"Data": [cafef_row("28/09/2026", 1, 2, 0.5, 1.9)]}, "Success": True})

    client = CafefClient(session=FakeSession(handler), sleep_fn=NO_SLEEP, page_delay=0.0)
    assert len(client.fetch("TCB")) == 1
    assert attempts["n"] == 2


def test_cafef_client_raises_aggregated_error_after_exhausting_retries():
    def handler(url, params):
        raise requests.ConnectionError("mạng đứt (giả lập)")

    client = CafefClient(session=FakeSession(handler), sleep_fn=NO_SLEEP, max_retries=2, page_delay=0.0)
    with pytest.raises(SourceError, match="mọi host"):
        client.fetch("TCB")
    # 2 host × 2 lần thử = 4 request, kèm thông điệp từng đường đã thử
    assert len(client._urls_to_try()) == 2


def test_cafef_client_fails_over_to_new_domain_and_remembers_it():
    def handler(url, params):
        if url.startswith("https://s.cafef.vn"):
            # giả lập redirect 301 làm rơi query string: server báo thiếu Symbol
            return FakeResponse(200, {"Data": None, "Message": "symbol is null or empty", "Success": False})
        return FakeResponse(
            200, {"Data": {"Data": [cafef_row("28/09/2026", 1, 2, 0.5, 1.9)]}, "Success": True}
        )

    session = FakeSession(handler)
    client = CafefClient(session=session, sleep_fn=NO_SLEEP, page_delay=0.0)
    assert len(client.fetch("TCB")) == 1
    assert client.preferred_url == "https://cafef.vn/du-lieu/ajax/pagenew/datahistory/pricehistory.ashx"
    client.fetch("TCB")
    # lần sau nhảy thẳng host đã biết tốt — host cũ chỉ bị thử đúng 1 lần
    assert sum(1 for c in session.calls if c["url"].startswith("https://s.cafef.vn")) == 1


def test_cafef_client_raises_on_contract_break():
    def handler(url, params):
        # Data tồn tại nhưng không có Data.Data → hợp đồng đã đổi
        return FakeResponse(200, {"Data": {"TotalCount": 0}, "Message": "ok", "Success": True})

    client = CafefClient(session=FakeSession(handler), sleep_fn=NO_SLEEP, page_delay=0.0)
    with pytest.raises(SourceError, match="thiếu Data.Data"):
        client.fetch("TCB")


def test_cafef_client_rejects_bad_page_size():
    with pytest.raises(ValueError, match="20"):
        CafefClient(page_size=1000)


# ---------------------------------------------------------------------------
# TcbsClient — hợp đồng fallback
# ---------------------------------------------------------------------------


def test_tcbs_client_parses_contract_and_builds_epoch_url():
    captured: dict = {}

    def handler(url, params):
        captured["url"] = url
        return FakeResponse(
            200,
            {
                "data": [
                    {"open": 33.0, "high": 33.4, "low": 32.9, "close": 33.1, "volume": 100, "tradingdate": "2015-03-10"},
                    {"open": 33.1, "high": 33.6, "low": 33.0, "close": 33.2, "volume": 200, "tradingdate": "2015-03-11"},
                ]
            },
        )

    client = TcbsClient(session=FakeSession(handler), sleep_fn=NO_SLEEP)
    frame = client.fetch("TCB", start="2015-01-01", end="2016-01-01")
    # epoch giây UTC của 2015-01-01 và 2016-01-01
    assert captured["url"] == "https://api-finance-tcbs.tcbs.com.vn/stock/ohlc/TCB/1420070400/1451606400"
    # ngày phiên trong khoảng [start, end] để kiểm chứng lọc hai đầu
    assert [d.strftime("%Y-%m-%d") for d in frame.index] == ["2015-03-10", "2015-03-11"]
    assert frame["close"].tolist() == [33.1, 33.2]


def test_tcbs_client_raises_on_missing_data():
    client = TcbsClient(
        session=FakeSession(lambda url, params: FakeResponse(200, {"error": "no data"})),
        sleep_fn=NO_SLEEP,
    )
    with pytest.raises(SourceError, match="thiếu 'data'"):
        client.fetch("TCB")


def test_tcbs_client_retries_on_dns_failure():
    attempts = {"n": 0}

    def handler(url, params):
        attempts["n"] += 1
        raise requests.ConnectionError("Failed to resolve 'api-finance-tcbs.tcbs.com.vn' (giả lập DNS chết)")

    client = TcbsClient(session=FakeSession(handler), sleep_fn=NO_SLEEP, max_retries=3)
    with pytest.raises(SourceError, match="3 lần"):
        client.fetch("TCB")
    assert attempts["n"] == 3


# ---------------------------------------------------------------------------
# CafefSource — ghép nguồn + fallback
# ---------------------------------------------------------------------------


def broken_session() -> FakeSession:
    def broken(url, params):
        raise requests.ConnectionError("đứt (giả lập)")

    return FakeSession(broken)


def test_cafef_source_implements_interface():
    source = CafefSource()
    assert isinstance(source, PriceDataSource)
    assert source.name


def test_cafef_source_falls_back_to_tcbs_when_cafef_breaks():
    def ok_tcbs(url, params):
        assert "/stock/ohlc/TCB/" in url
        return FakeResponse(
            200,
            {"data": [{"open": 1, "high": 2, "low": 0.5, "close": 1.9, "volume": 10, "tradingdate": "2026-08-05"}]},
        )

    source = CafefSource(
        cafef=CafefClient(session=broken_session(), sleep_fn=NO_SLEEP, max_retries=1, page_delay=0.0),
        tcbs=TcbsClient(session=FakeSession(ok_tcbs), sleep_fn=NO_SLEEP),
    )
    outcome = source.fetch_outcome("TCB", start="2026-08-01")
    assert outcome.source == "tcbs"
    assert len(outcome.frame) == 1
    assert any("CAFEF lỗi" in w for w in outcome.warnings)
    assert len(source.fetch_ohlcv("TCB")) == 1  # qua interface chung


def test_cafef_source_returns_empty_frame_with_warning_when_both_fail():
    source = CafefSource(
        cafef=CafefClient(session=broken_session(), sleep_fn=NO_SLEEP, max_retries=1, page_delay=0.0),
        tcbs=TcbsClient(session=broken_session(), sleep_fn=NO_SLEEP),
    )
    outcome = source.fetch_outcome("TCB")  # KHÔNG được raise
    assert outcome.source == "none"
    assert outcome.frame.empty
    assert list(outcome.frame.columns) == list(OHLCV_COLUMNS)
    assert any("Cả CAFEF và TCBS" in w for w in outcome.warnings)


def test_cafef_source_empty_when_both_return_no_rows():
    def empty_cafef(url, params):
        return FakeResponse(200, {"Data": {"Data": []}, "Success": True})

    def empty_tcbs(url, params):
        return FakeResponse(200, {"data": []})

    source = CafefSource(
        cafef=CafefClient(session=FakeSession(empty_cafef), sleep_fn=NO_SLEEP, page_delay=0.0),
        tcbs=TcbsClient(session=FakeSession(empty_tcbs), sleep_fn=NO_SLEEP),
    )
    outcome = source.fetch_outcome("TCB")
    assert outcome.source == "none" and outcome.frame.empty
    assert any("CAFEF trả 0 phiên" in w for w in outcome.warnings)


def test_cafef_source_vnindex_uses_vnindex_symbol():
    session = cafef_ok_session({1: [cafef_row("29/09/2026", 1, 2, 0.5, 1.9)]})
    source = CafefSource(
        cafef=CafefClient(session=session, sleep_fn=NO_SLEEP, page_delay=0.0),
        tcbs=TcbsClient(session=broken_session(), sleep_fn=NO_SLEEP),
    )
    frame = source.fetch_vnindex(start="2026-09-01")
    assert session.calls[0]["params"]["Symbol"] == "VNINDEX"
    assert list(frame.index) == [pd.Timestamp("2026-09-29")]


# ---------------------------------------------------------------------------
# Danh sách mã HOSE
# ---------------------------------------------------------------------------


def test_normalize_symbol_frame_dedupes_and_keeps_real_metadata():
    frame = pd.DataFrame(
        {
            "symbol": ["TCB", "tcb ", "VCB", "AAM", "!!!"],
            "name": [None, None, None, None, None],
            "industry": ["Ngân hàng", None, None, None, None],
        }
    )
    out = normalize_symbol_frame(frame)
    assert out["symbol"].tolist() == ["AAM", "TCB", "VCB"]
    assert list(out.columns) == ["symbol", "industry"]  # name toàn rỗng → bỏ; industry có thật → giữ
    assert out.loc[out["symbol"] == "TCB", "industry"].iloc[0] == "Ngân hàng"


def test_banggia_provider_parses_codes_and_caches(tmp_path):
    payload = [{"a": "AAA", "b": 7.17}, {"a": "TCB", "b": 32.85}, {"bad": "row"}, {"a": "AAM"}]
    session = FakeSession(lambda url, params: FakeResponse(200, payload))
    cache = tmp_path / "symbols.parquet"
    directory = HoseSymbolDirectory(cache_path=cache, session=session, providers=None)

    result = directory.get(refresh=True)
    assert result.source == "_fetch_banggia_hose"
    assert result.message is None
    assert result.symbols == ["AAA", "AAM", "TCB"]  # upper, dedupe, sort
    assert session.calls[0]["params"]["curFloorCode"] == "HOSE"

    # lần 2 còn tươi → đọc cache, không gọi mạng
    result2 = directory.get()
    assert result2.source == "cache"
    assert len(session.calls) == 1
    assert pd.read_parquet(cache)["symbol"].tolist() == ["AAA", "AAM", "TCB"]


def test_symbols_directory_empty_with_message_when_all_fail(tmp_path):
    def broken():
        raise requests.ConnectionError("đứt (giả lập)")

    directory = HoseSymbolDirectory(cache_path=tmp_path / "symbols.parquet", providers=[broken])
    result = directory.get()
    assert result.frame.empty
    assert result.source is None
    assert result.message and "nhập mã thủ công" in result.message


def test_symbols_directory_falls_back_to_stale_cache_with_message(tmp_path):
    cache = tmp_path / "symbols.parquet"
    pd.DataFrame({"symbol": ["TCB", "VCB"]}).to_parquet(cache)
    old = time.time() - 30 * 86400
    os.utime(cache, (old, old))

    def broken():
        raise requests.ConnectionError("đứt (giả lập)")

    result = HoseSymbolDirectory(cache_path=cache, providers=[broken]).get()
    assert result.symbols == ["TCB", "VCB"]
    assert result.message and "cache cũ" in result.message


# ---------------------------------------------------------------------------
# data_loader — cache parquet + cập nhật tăng dần
# ---------------------------------------------------------------------------


def _seed_cache(tmp_path: Path) -> pd.DatetimeIndex:
    dates = pd.bdate_range("2025-01-02", periods=10)  # 02/01 → 15/01/2025
    closes = [10 + i for i in range(10)]
    src = ScriptedSource([FetchOutcome(frame=ohlcv_frame(dates, closes), source="scripted")])
    out = data_loader.load_history("tcb", data_dir=tmp_path, source=src)
    assert len(out) == 10
    return dates


def test_load_history_first_load_caches_raw_and_processed(tmp_path):
    dates = pd.bdate_range("2025-01-02", periods=10)
    src = ScriptedSource([FetchOutcome(frame=ohlcv_frame(dates, [10 + i for i in range(10)]), source="scripted")])
    out = data_loader.load_history("tcb", data_dir=tmp_path, source=src)

    assert len(out) == 10
    assert (tmp_path / "raw" / "TCB.parquet").exists()
    assert (tmp_path / "processed" / "TCB.parquet").exists()
    stored = pd.read_parquet(tmp_path / "raw" / "TCB.parquet")
    assert len(stored) == 10
    assert stored.index.max() == pd.Timestamp("2025-01-15")
    assert src.calls == [{"symbol": "TCB", "start": "2015-01-01", "end": None}]


def test_load_history_incremental_fetches_only_missing_part(tmp_path):
    _seed_cache(tmp_path)

    new_dates = pd.bdate_range("2025-01-15", periods=6)  # overlap phiên cuối + 5 phiên mới
    src = ScriptedSource(
        [FetchOutcome(frame=ohlcv_frame(new_dates, [19.0, 20.0, 21.0, 22.0, 23.0, 24.0]), source="scripted")]
    )
    out = data_loader.load_history("TCB", data_dir=tmp_path, source=src)

    assert src.calls[0]["start"] == "2025-01-15"  # chỉ tải phần thiếu (overlap 1 phiên)
    assert len(out) == 15
    assert out.index.max() == pd.Timestamp("2025-01-22")
    assert out.iloc[-1]["close"] == 24.0
    assert pd.read_parquet(tmp_path / "raw" / "TCB.parquet").shape[0] == 15
    assert pd.read_parquet(tmp_path / "processed" / "TCB.parquet").shape[0] == 15
    assert data_loader.last_cached_date("TCB", tmp_path) == pd.Timestamp("2025-01-22")


def test_load_history_keeps_cache_when_source_returns_nothing(tmp_path):
    _seed_cache(tmp_path)
    src = ScriptedSource([FetchOutcome(frame=normalize_ohlcv(None), source="none",
                                       warnings=["Cả CAFEF và TCBS đều không có dữ liệu"])])
    out = data_loader.load_history("TCB", data_dir=tmp_path, source=src)
    assert len(out) == 10  # cache được giữ nguyên
    assert pd.read_parquet(tmp_path / "raw" / "TCB.parquet").shape[0] == 10


def test_load_history_refresh_redownloads_full(tmp_path):
    _seed_cache(tmp_path)
    dates = pd.bdate_range("2025-01-02", periods=12)
    src = ScriptedSource([FetchOutcome(frame=ohlcv_frame(dates, [10 + i for i in range(12)]), source="scripted")])
    out = data_loader.load_history("TCB", data_dir=tmp_path, source=src, refresh=True)
    assert src.calls[0]["start"] == "2015-01-01"  # refresh → tải lại toàn bộ
    assert len(out) == 12


def test_load_history_returns_empty_and_writes_nothing_when_source_has_no_data(tmp_path):
    src = ScriptedSource([FetchOutcome(frame=normalize_ohlcv(None), source="none", warnings=["không có dữ liệu"])])
    out = data_loader.load_history("XYZ", data_dir=tmp_path, source=src)
    assert out.empty
    assert not (tmp_path / "raw" / "XYZ.parquet").exists()
    assert not (tmp_path / "processed" / "XYZ.parquet").exists()


def test_load_history_skips_network_when_end_within_cache(tmp_path):
    _seed_cache(tmp_path)
    src = ScriptedSource([])  # không có kịch bản → nếu bị gọi sẽ trả rỗng
    out = data_loader.load_history("TCB", data_dir=tmp_path, source=src, end="2025-01-10")
    assert src.calls == []  # cache đã phủ quá end → không gọi nguồn
    assert out.index.max() == pd.Timestamp("2025-01-10")


# ---------------------------------------------------------------------------
# watchlist — CRUD
# ---------------------------------------------------------------------------


def test_watchlist_crud_roundtrip(tmp_path):
    path = tmp_path / "watchlist.json"
    wl = Watchlist(path)

    assert wl.list_tickers() == []
    assert wl.add("tcb") is True  # lưu HOA
    assert wl.add("TCB") is False  # trùng (không phân biệt hoa/thường)
    assert wl.add("VCI") is True
    assert wl.contains("vci") is True
    assert wl.list_tickers() == ["TCB", "VCI"]

    entries = wl.entries()
    assert entries[0] == {"ticker": "TCB", "added_at": entries[0]["added_at"]}
    assert entries[0]["added_at"]

    assert wl.remove("tcb") is True
    assert wl.remove("TCB") is False  # đã xoá
    assert wl.list_tickers() == ["VCI"]

    # instance mới đọc lại cùng file (bền vững)
    assert Watchlist(path).list_tickers() == ["VCI"]
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 1 and data["watchlist"][0]["ticker"] == "VCI"


def test_watchlist_rejects_invalid_ticker(tmp_path):
    wl = Watchlist(tmp_path / "watchlist.json")
    with pytest.raises(ValueError, match="không hợp lệ"):
        wl.add("TC B!")
    with pytest.raises(ValueError, match="không hợp lệ"):
        wl.add("")
    assert wl.list_tickers() == []


def test_watchlist_corrupt_file_raises_instead_of_silently_resetting(tmp_path):
    path = tmp_path / "watchlist.json"
    path.write_text("{không phải json", encoding="utf-8")
    with pytest.raises(WatchlistError):
        Watchlist(path).list_tickers()


# ---------------------------------------------------------------------------
# db — schema + settle
# ---------------------------------------------------------------------------


@pytest.fixture
def conn(tmp_path):
    connection = db.connect(tmp_path / "app.db")
    db.init_db(connection)
    yield connection
    connection.close()


def test_db_init_idempotent_and_schema_complete(conn):
    db.init_db(conn)  # chạy lại không lỗi
    tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"runs", "predictions", "outcomes"} <= tables


def test_db_record_run_and_prediction_roundtrip(conn):
    run_id = db.record_run(
        conn,
        "tcb",
        data_until="2025-01-15",
        metrics={"mae": 0.5, "skill_vs_persistence": 0.1, "edge_vs_base_rate": 2.3},
        model_dir="models/TCB/20250115",
    )
    assert run_id == 1
    runs = db.list_runs(conn)
    assert runs[0]["ticker"] == "TCB" and runs[0]["data_until"] == "2025-01-15"
    assert json.loads(runs[0]["metrics_json"])["mae"] == 0.5
    assert runs[0]["trained_at"] and runs[0]["model_dir"] == "models/TCB/20250115"

    pred_id = db.record_prediction(
        conn, run_id, "TCB", "lgbm_reg", "2025-01-16", 1,
        predicted_price=34.1, predicted_direction="up", probability_up=0.61,
    )
    preds = db.list_predictions(conn, run_id=run_id)
    assert preds[0]["id"] == pred_id and preds[0]["target_date"] == "2025-01-16"
    assert preds[0]["horizon"] == 1 and preds[0]["probability_up"] == pytest.approx(0.61)

    with pytest.raises(sqlite3.IntegrityError):
        db.record_prediction(conn, 999, "TCB", "m", "2025-01-16", 1)  # run không tồn tại
    with pytest.raises(sqlite3.IntegrityError):
        db.record_prediction(conn, run_id, "TCB", "m", "2025-01-16", 1, predicted_direction="sideways")


def test_db_classify_direction():
    assert db.classify_direction(0.005, 0.003) == "up"
    assert db.classify_direction(-0.005, 0.003) == "down"
    assert db.classify_direction(0.002, 0.003) == "neutral"
    assert db.classify_direction(-0.002, 0.003) == "neutral"


SETTLE_PRICES = ohlcv_frame(
    pd.bdate_range("2025-01-06", periods=10),  # 06,07,08,09,10,13,14,15,16,17/01/2025
    [100.0, 101.0, 102.0, 99.0, 100.0, 103.0, 103.3, 102.9, 101.0, 105.0],
)


def test_db_settle_predictions_matches_direction_abs_error_and_hit(conn):
    provider = lambda ticker: SETTLE_PRICES  # noqa: E731
    run_id = db.record_run(conn, "TCB")
    p1 = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-10", 1,
                              predicted_price=101.0, predicted_direction="up")
    # 100/99 − 1 = +1.01% > 0.3% → 'up' → hit=1, |101−100| = 1.0
    p2 = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-10", 1,
                              predicted_price=99.0, predicted_direction="down")
    # cùng target → 'up' → hit=0, |99−100| = 1.0
    p3 = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-14", 1, predicted_direction="neutral")
    # 103.3/103 − 1 = +0.29% nằm trong ±0.3% → 'neutral' → hit=1; predicted_price None → abs_error None
    p4 = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-17", 5, predicted_direction="up")
    # horizon 5: so close[17/01]=105 với close cách 5 phiên = close[10/01]=100 → +5% → 'up' → hit=1

    settled = db.settle_predictions(conn, price_provider=provider, neutral_band=0.003)
    assert settled == 4

    rows = {r["prediction_id"]: r for r in conn.execute("SELECT * FROM outcomes")}
    assert rows[p1]["actual_price"] == pytest.approx(100.0)
    assert rows[p1]["actual_direction"] == "up"
    assert rows[p1]["hit"] == 1
    assert rows[p1]["abs_error"] == pytest.approx(1.0)
    assert rows[p2]["hit"] == 0 and rows[p2]["abs_error"] == pytest.approx(1.0)
    assert rows[p3]["actual_direction"] == "neutral"
    assert rows[p3]["abs_error"] is None
    assert rows[p3]["hit"] == 1
    assert rows[p4]["actual_direction"] == "up" and rows[p4]["hit"] == 1
    assert all(rows[p]["settled_at"] for p in (p1, p2, p3, p4))

    # idempotent: chạy lại không settle thêm dòng nào
    assert db.settle_predictions(conn, price_provider=provider) == 0


def test_db_settle_leaves_unsettled_when_no_price_for_target(conn):
    provider = lambda ticker: SETTLE_PRICES  # noqa: E731
    run_id = db.record_run(conn, "TCB")
    p_future = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-31", 1, predicted_direction="up")
    p_weekend = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-11", 1, predicted_direction="up")  # thứ Bảy
    p_neutral_pred = db.record_prediction(conn, run_id, "TCB", "m", "2025-01-15", 1,
                                          predicted_price=102.5, predicted_direction="neutral")
    # 102.9/103.3 − 1 = −0.39% < −0.3% → 'down' ≠ 'neutral' → hit=0; |102.5−102.9| = 0.4

    assert db.settle_predictions(conn, price_provider=provider) == 1
    assert sorted(r["id"] for r in db.unsettled_predictions(conn)) == sorted([p_future, p_weekend])
    rows = {r["prediction_id"]: r for r in conn.execute("SELECT * FROM outcomes")}
    assert rows[p_neutral_pred]["actual_direction"] == "down"
    assert rows[p_neutral_pred]["hit"] == 0
    assert rows[p_neutral_pred]["abs_error"] == pytest.approx(0.4)


def test_db_settle_skips_ticker_with_no_prices_and_survives_provider_error(conn):
    run_id = db.record_run(conn, "XYZ")
    db.record_prediction(conn, run_id, "XYZ", "m", "2025-01-10", 1, predicted_direction="up")

    assert db.settle_predictions(conn, price_provider=lambda t: normalize_ohlcv(None)) == 0
    assert len(db.unsettled_predictions(conn)) == 1

    def exploding(ticker):
        raise RuntimeError("nguồn giá nổ (giả lập)")

    assert db.settle_predictions(conn, price_provider=exploding) == 0  # không raise
    assert len(db.unsettled_predictions(conn)) == 1


def test_db_settle_default_provider_uses_data_loader(conn, monkeypatch):
    """Không truyền provider → settle phải đi qua data_loader.load_history (đã mock)."""
    run_id = db.record_run(conn, "TCB")
    db.record_prediction(conn, run_id, "TCB", "m", "2025-01-10", 1, predicted_direction="up")
    dates = pd.bdate_range("2025-01-02", periods=8)  # ...,09/01,10/01,...
    calls: list[str] = []

    def fake_load_history(ticker, **kwargs):
        calls.append(ticker)
        return ohlcv_frame(dates, [50.0 + i for i in range(8)])

    monkeypatch.setattr(data_loader, "load_history", fake_load_history)
    assert db.settle_predictions(conn) == 1
    assert calls == ["TCB"]
    outcomes = conn.execute("SELECT * FROM outcomes").fetchall()
    # close[10/01]=56, close[09/01]=55 → +1.82% > 0.3% → 'up', trùng hướng dự báo
    assert outcomes[0]["actual_direction"] == "up"
    assert outcomes[0]["hit"] == 1


# ---------------------------------------------------------------------------
# HoseOfficialSource — API chính thức HOSE (close + volume)
# ---------------------------------------------------------------------------


def hose_ok_session(rows: list[dict]):
    def handler(url, params):
        assert "/market/securities/chart/" in url
        return FakeResponse(200, {"data": rows, "success": True, "message": ""})

    return FakeSession(handler)


def test_hose_official_parses_chart_contract_and_converts_units():
    rows = [
        {"reportdate": 1528125271, "closeprice": 102400.0, "totalshare": 2811840.0},  # 2018-06-04 (giờ tối VN)
        {"reportdate": 1790553600, "closeprice": 32850.0, "totalshare": 23313354.0},  # 2026-09-28 00:00 UTC
    ]
    session = hose_ok_session(rows)
    source = HoseOfficialSource(session=session, sleep_fn=NO_SLEEP)
    frame = source.fetch_ohlcv("TCB", start="2015-01-01", end="2026-09-29")

    assert [d.strftime("%Y-%m-%d") for d in frame.index] == ["2018-06-04", "2026-09-28"]
    # đồng → nghìn đồng (khớp đơn vị CAFEF)
    assert frame["close"].tolist() == [102.4, 32.85]
    assert frame["volume"].tolist() == [2811840.0, 23313354.0]
    # nguồn này KHÔNG có open/high/low → NaN, không bịa số
    assert frame["open"].isna().all() and frame["high"].isna().all() and frame["low"].isna().all()

    first = session.calls[0]
    assert first["url"] == "https://api.hsx.vn/mk/api/v1/market/securities/chart/TCB"
    assert first["params"]["fromDate"] == "2015-01-01"
    assert first["params"]["toDate"] == "2026-09-29"
    assert "hsx.vn" in first["headers"]["Referer"]
    assert "User-Agent" in first["headers"]


def test_hose_official_returns_empty_for_vnindex_sparse_series():
    """VNINDEX bị chặn khỏi nguồn này: chuỗi 'VNI' của HOSE là chuỗi thưa (không phải ngày),
    ghép vào cache ngày sẽ làm sai lợi nhuận — nguồn chủ động trả rỗng + cảnh báo."""
    session = hose_ok_session([])
    source = HoseOfficialSource(session=session, sleep_fn=NO_SLEEP)
    outcome = source.fetch_outcome("VNINDEX")
    assert outcome.frame.empty
    assert outcome.warnings and "chuỗi thưa" in outcome.warnings[0]
    assert session.calls == []  # không gọi mạng cho mã bị chặn


def test_hose_official_raises_on_missing_data_and_http_error():
    source_missing = HoseOfficialSource(
        session=FakeSession(lambda url, params: FakeResponse(200, {"success": False, "message": "no data"})),
        sleep_fn=NO_SLEEP,
    )
    with pytest.raises(SourceError, match="thiếu 'data'"):
        source_missing.fetch_outcome("TCB")

    def bad(url, params):
        raise requests.ConnectionError("đứt (giả lập)")

    source_broken = HoseOfficialSource(session=FakeSession(bad), sleep_fn=NO_SLEEP)
    with pytest.raises(SourceError, match="HOSE official chart TCB"):
        source_broken.fetch_outcome("TCB")


# ---------------------------------------------------------------------------
# PriorityMergeSource — chuỗi nguồn ưu tiên
# ---------------------------------------------------------------------------


def partial_ohlcv_frame(dates, closes, with_ohl: bool) -> pd.DataFrame:
    frame = ohlcv_frame(dates, closes)
    if not with_ohl:
        frame[["open", "high", "low"]] = float("nan")  # mô phỏng nguồn chỉ có close
    return frame


def test_priority_merge_prefers_primary_on_overlap_and_fills_history():
    primary_dates = pd.bdate_range("2025-01-13", periods=2)  # chỉ 2 phiên gần
    backup_dates = pd.bdate_range("2025-01-08", periods=5)  # phủ sâu hơn, trùng 2 phiên cuối
    primary = ScriptedSource(
        [FetchOutcome(frame=partial_ohlcv_frame(primary_dates, [30.0, 31.0], with_ohl=True), source="primary")]
    )
    backup = ScriptedSource(
        [FetchOutcome(frame=partial_ohlcv_frame(backup_dates, [28.0, 29.0, 30.2, 31.4, 32.0], with_ohl=False),
                      source="backup")]
    )
    chain = PriorityMergeSource([primary, backup])
    outcome = chain.fetch_outcome("TCB", start="2025-01-08")

    assert outcome.source == "primary+backup"
    assert len(outcome.frame) == 5  # backup lấp phần lịch sử cũ hơn
    # ngày trùng: giá trị primary thắng (có OHLC thật)
    overlap = outcome.frame.loc[[pd.Timestamp("2025-01-13"), pd.Timestamp("2025-01-14")]]
    assert overlap["close"].tolist() == [30.0, 31.0]
    assert overlap["open"].notna().all()
    # ngày chỉ backup có: close thật, open/high/low NaN (không bịa)
    old = outcome.frame.loc[[pd.Timestamp("2025-01-08"), pd.Timestamp("2025-01-09")]]
    assert old["close"].tolist() == [28.0, 29.0]
    assert old["open"].isna().all()


def test_priority_merge_skips_backup_when_primary_covers_range():
    primary_dates = pd.bdate_range("2025-01-02", periods=10)
    primary = ScriptedSource([FetchOutcome(frame=ohlcv_frame(primary_dates, [10.0 + i for i in range(10)]), source="primary")])
    backup = ScriptedSource([])  # nếu bị gọi sẽ trả rỗng
    chain = PriorityMergeSource([primary, backup])
    outcome = chain.fetch_outcome("TCB", start="2025-01-02")  # primary phủ đủ → backup không bị gọi
    assert backup.calls == []
    assert outcome.source == "primary"
    assert len(outcome.frame) == 10


def test_priority_merge_empty_when_all_sources_empty():
    chain = PriorityMergeSource(
        [
            ScriptedSource([FetchOutcome(frame=normalize_ohlcv(None), source="primary", warnings=["primary hết dữ liệu"])]),
            ScriptedSource([FetchOutcome(frame=normalize_ohlcv(None), source="backup", warnings=["backup hết dữ liệu"])]),
        ]
    )
    outcome = chain.fetch_outcome("TCB", start="2015-01-01")
    assert outcome.source == "none"
    assert outcome.frame.empty
    assert any("primary hết dữ liệu" in w for w in outcome.warnings)
    assert any("Mọi nguồn trong chuỗi" in w for w in outcome.warnings)


def test_data_loader_default_source_is_the_documented_chain():
    chain = data_loader.default_source()
    assert isinstance(chain, PriorityMergeSource)
    assert [type(s).__name__ for s in chain.sources] == ["CafefSource", "HoseOfficialSource"]


def test_load_history_uses_backup_to_reach_requested_start(tmp_path):
    primary_dates = pd.bdate_range("2025-01-13", periods=2)
    backup_dates = pd.bdate_range("2025-01-02", periods=10)
    chain = PriorityMergeSource(
        [
            ScriptedSource([FetchOutcome(frame=partial_ohlcv_frame(primary_dates, [30.0, 31.0], with_ohl=True), source="primary")]),
            ScriptedSource([FetchOutcome(frame=partial_ohlcv_frame(backup_dates, [20.0 + i for i in range(10)], with_ohl=False), source="backup")]),
        ]
    )
    out = data_loader.load_history("TCB", data_dir=tmp_path, source=chain)
    assert len(out) == 10
    assert out.index.min() == pd.Timestamp("2025-01-02")
    stored = pd.read_parquet(tmp_path / "raw" / "TCB.parquet")
    assert len(stored) == 10
    # phần cũ hơn cửa sổ primary: open/high/low NaN trong cache (thật, không bịa)
    assert stored.loc[: pd.Timestamp("2025-01-10"), "open"].isna().all()
    assert stored.loc[: pd.Timestamp("2025-01-10"), "close"].notna().all()
