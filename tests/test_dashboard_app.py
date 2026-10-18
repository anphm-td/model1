"""Kiểm thử dashboard Streamlit qua ``streamlit.testing.v1.AppTest`` — OFFLINE.

Nguyên tắc test:
- KHÔNG gọi mạng: dữ liệu giá (``src.data_loader.load_history``), danh sách mã
  (``HoseSymbolDirectory.get``), watchlist và config (``src.pipeline.load_config``)
  đều được monkeypatch; SQLite trỏ tới tệp tạm ``tmp_path``.
- AppTest.from_file("app/dashboard.py").run() phải KHÔNG exception ở mọi tab.
- "Dữ liệu từ cache": chạy trang 2 lần — nguồn giả lập chỉ được gọi ĐÚNG 1 lần
  (lần sau trúng st.cache_data), chứng minh dashboard đọc cache chứ không đọc
  lại nguồn.
- Widget TradingView xuất hiện dưới dạng phần tử ``iframe`` trong AppTest
  (st.components.v1.html); tắt config → fallback plotly_chart nội bộ.

Hành vi đối chiếu đã xác minh bằng chạy thử AppTest trên dashboard thật trước
khi viết file này (banner, iframe, plotly_chart, dataframe DNSE, warning 401).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db as src_db  # noqa: E402
from src.dnse_client import DnseAuthError, DnseClient  # noqa: E402

DASHBOARD_PATH = PROJECT_ROOT / "app" / "dashboard.py"


# ---------------------------------------------------------------------------
# Môi trường test — config tạm + nguồn dữ liệu giả lập (không mạng, không đụng
# data/ của dự án thật)
# ---------------------------------------------------------------------------


def make_cfg(tmp_path: Path, *, tradingview: bool = True, dnse: bool = False,
             db_name: str = "app.db") -> dict[str, Any]:
    """Config thu nhỏ: DB + watchlist trong tmp_path, cờ TradingView/DNSE."""
    return {
        "project": {"title": "Dự báo cổ phiếu sàn HOSE (test)"},
        "default_ticker": "TCB",
        "app": {
            "watchlist_path": str(tmp_path / "watchlist.json"),
            "tradingview": {"enabled": tradingview, "exchange": "HOSE"},
            "dnse": {
                "enabled": dnse,
                "jwt_token": "test-jwt-token" if dnse else "",
                "account_no": "0001234567" if dnse else "",
            },
        },
        "database": {"path": str(tmp_path / db_name)},
        "neutral_band_pct": 0.3,
        "walk_forward": {"retrain_every": 21, "embargo": 5},
    }


def fake_price_frame(periods: int = 120) -> pd.DataFrame:
    """OHLCV giả lập 120 phiên — giá nghìn đồng, đủ cột cho PriceFeatures."""
    idx = pd.bdate_range("2026-03-02", periods=periods)
    rng = np.random.default_rng(42)
    base = 24.0 + np.cumsum(rng.normal(0, 0.05, periods))
    return pd.DataFrame(
        {
            "open": base,
            "high": base + 0.15,
            "low": base - 0.15,
            "close": base + 0.05,
            "volume": rng.integers(500_000, 5_000_000, periods).astype(float),
        },
        index=idx,
    )


@pytest.fixture
def app_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Gắn nguồn dữ liệu giả lập; ``install(cfg)`` gắn config + clear cache."""
    state: dict[str, Any] = {"price_calls": [], "symbols_calls": 0}

    def fake_load_history(ticker, *args, **kwargs):
        state["price_calls"].append(ticker)
        return fake_price_frame()

    def fake_get_symbols(self):
        state["symbols_calls"] += 1
        return SimpleNamespace(
            frame=pd.DataFrame(
                {
                    "symbol": ["TCB", "FPT", "MBB", "VCB"],
                    "name": ["Techcombank", "FPT Corp", "MB", "Vietcombank"],
                }
            ),
            source="fake-offline",
            message=None,
        )

    monkeypatch.setattr("src.data_loader.load_history", fake_load_history)
    monkeypatch.setattr("src.data_sources.cafef_source.HoseSymbolDirectory.get", fake_get_symbols)
    monkeypatch.setattr("src.watchlist.Watchlist.list_tickers", lambda self: ["TCB", "FPT"])
    monkeypatch.setattr(
        "src.watchlist.Watchlist.entries",
        lambda self: [
            {"ticker": "TCB", "added_at": "2026-09-01T10:00:00"},
            {"ticker": "FPT", "added_at": "2026-09-02T10:00:00"},
        ],
    )

    def install(cfg: dict[str, Any]) -> dict[str, Any]:
        monkeypatch.setattr("src.pipeline.load_config", lambda path=None: cfg)
        st.cache_data.clear()  # cache_data là toàn tiến trình — phải reset giữa test
        return state

    yield install
    st.cache_data.clear()


def run_app(cfg: dict[str, Any], *, timeout: int = 60) -> AppTest:
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=timeout)
    at.run()
    return at


def run_app_tab(cfg: dict[str, Any], tab: str, *, timeout: int = 60) -> AppTest:
    """Mở dashboard thẳng vào một tab qua session_state["nav"]."""
    at = AppTest.from_file(str(DASHBOARD_PATH), default_timeout=timeout)
    at.session_state["nav"] = tab
    at.run()
    return at


def iframes(at: AppTest) -> list[Any]:
    """Các phần tử ``iframe`` mức top-level của main (widget st.components)."""
    return [child for child in at.main.children.values() if child.type == "iframe"]


def markdown_values(at: AppTest) -> list[str]:
    return [str(m.value) for m in at.markdown]


def seed_forecast_db(db_path: Path, ticker: str = "TCB") -> None:
    """Ghi 1 run + 3 dự báo horizon=1 (2 tăng, 1 giảm; lstm thiếu giá/xác suất).

    Đồng thuận kỳ vọng: mean giá = (24.5+24.7)/2 = 24.60 → ≈ 24,600 VND;
    P(tăng) trung bình = (0.6+0.55)/2 = 57.5% → "58%"; n_up/n_models = 2/3.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        src_db.init_db(conn)
        run_id = src_db.record_run(conn, ticker, data_until="2026-09-29", metrics={"forecasts": []})
        rows = [
            ("naive", 24.5, "up", 0.6),
            ("arima", 24.7, "up", 0.55),
            ("lstm", None, "down", None),
        ]
        for model, price, direction, prob in rows:
            src_db.record_prediction(
                conn, run_id, ticker, model, "2026-09-30", 1,
                predicted_price=price, predicted_direction=direction, probability_up=prob,
            )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 1) App chạy không exception (mặc định: tab Thị trường) + banner chưa dự báo
# ---------------------------------------------------------------------------


def test_app_runs_without_exception_default_tab(app_env, tmp_path):
    cfg = make_cfg(tmp_path)
    app_env(cfg)
    at = run_app(cfg)

    assert not at.exception
    # Nav 4 tab nguyên vẹn
    assert at.radio[0].options == [
        "Thị trường", "Danh mục", "Phân tích & dự báo", "Lịch sử dự đoán"
    ]
    # Banner toàn cục hiện đúng trạng thái DB trống (chưa huấn luyện)
    assert any("⚠️ Chưa có dự báo cho TCB" in md for md in markdown_values(at))
    assert any("tham khảo, không phải khuyến nghị" in md for md in markdown_values(at))


# ---------------------------------------------------------------------------
# 2) Tab Thị trường: Ticker Tape + Market Overview (TradingView) + bảng mã
# ---------------------------------------------------------------------------


def test_tab1_renders_tradingview_widgets_and_symbol_table(app_env, tmp_path):
    cfg = make_cfg(tmp_path)
    app_env(cfg)
    at = run_app(cfg)

    assert not at.exception
    assert not at.warning
    # 2 widget TradingView: Ticker Tape (dưới header tab) + Market Overview (cuối tab)
    assert len(iframes(at)) == 2
    assert any("Băng giá trực tiếp từ TradingView" in str(c.value) for c in at.caption)
    assert any("HOSE:VNINDEX" in str(c.value) for c in at.caption)
    # Bảng mã render bình thường (4 mã giả lập, phân trang 1 trang)
    assert any("**Trang 1/1** — 4 mã khớp" in md for md in markdown_values(at))
    assert at.text_input[0].label == "Tìm kiếm theo mã hoặc tên"


def test_tab1_reads_symbols_from_cache_on_rerun(app_env, tmp_path):
    cfg = make_cfg(tmp_path)
    install = app_env(cfg)
    at = run_app(cfg)
    assert not at.exception
    assert install["symbols_calls"] == 1

    at2 = run_app(cfg)  # chạy lại — nguồn danh sách mã KHÔNG được gọi nữa (cache hit)
    assert not at2.exception
    assert install["symbols_calls"] == 1


# ---------------------------------------------------------------------------
# 3) Banner đồng thuận với DB đã có dự báo (tomorrow_consensus qua dashboard)
# ---------------------------------------------------------------------------


def test_banner_consensus_with_seeded_db(app_env, tmp_path):
    cfg = make_cfg(tmp_path, db_name="seeded.db")
    seed_forecast_db(Path(cfg["database"]["path"]))
    app_env(cfg)
    at = run_app(cfg)

    assert not at.exception
    banners = [md for md in markdown_values(at) if "DỰ BÁO PHIÊN MAI" in md]
    assert len(banners) == 1
    banner = banners[0]
    assert "DỰ BÁO PHIÊN MAI 2026-09-30 — TCB" in banner
    assert "giá <b>24.60</b> nghìn đ (≈ 24,600 VND)" in banner
    assert "hướng: 2 tăng / 1 giảm / 0 ngang" in banner
    # (0.6+0.55)/2 = 0.575 — Python "{:.0%}" in "57%" (float 0.575 ≈ 0.5749...)
    assert "P(tăng) <b>57%</b>" in banner
    assert "đồng thuận 2/3 model báo tăng" in banner
    assert "Nguồn: run #1" in banner
    assert "dữ liệu đến 2026-09-29" in banner


# ---------------------------------------------------------------------------
# 4) Tab Phân tích & dự báo: TradingView Advanced Chart + Technical Analysis
# ---------------------------------------------------------------------------


def test_tab3_tradingview_widgets_kpis_and_price_cache(app_env, tmp_path):
    cfg = make_cfg(tmp_path)
    install = app_env(cfg)
    at = run_app_tab(cfg, "Phân tích & dự báo")

    assert not at.exception
    # 4 KPI nguyên vẹn
    labels = [m.label for m in at.metric]
    assert labels == [
        "Giá đóng cửa (nghìn đ)", "RSI(14)", "Khối lượng (triệu cp)", "Số phiên trong khung xem",
    ]
    # 2 widget TradingView: Advanced Chart (620px) + Technical Analysis gauge (400px)
    assert len(iframes(at)) == 2
    assert any("Technical Analysis — gauge tín hiệu" in md for md in markdown_values(at))
    assert any("HOSE:TCB" in str(c.value) for c in at.caption)
    # Không vẽ plotly giá nội bộ khi TradingView bật
    assert not [ch for ch in at.main.children.values() if ch.type == "plotly_chart"]

    # Dữ liệu giá từ cache: chạy lại — load_history KHÔNG được gọi lần nữa
    at2 = run_app_tab(cfg, "Phân tích & dự báo")
    assert not at2.exception
    assert install["price_calls"] == ["TCB"]


def test_tab3_plotly_fallback_when_tradingview_disabled(app_env, tmp_path):
    cfg = make_cfg(tmp_path, tradingview=False)
    app_env(cfg)
    at = run_app_tab(cfg, "Phân tích & dự báo")

    assert not at.exception
    # Tắt widget → KHÔNG iframe nào, quay về plotly nội bộ
    assert len(iframes(at)) == 0
    assert [ch for ch in at.main.children.values() if ch.type == "plotly_chart"]
    assert any("plotly nội bộ" in str(c.value) for c in at.caption)
    assert not any("Technical Analysis — gauge tín hiệu" in md for md in markdown_values(at))


# ---------------------------------------------------------------------------
# 5) Tab Danh mục: mục "Tài khoản DNSE" (chỉ-đọc) theo cấu hình
# ---------------------------------------------------------------------------


def test_tab2_dnse_section_shows_positions_table(app_env, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, dnse=True)
    app_env(cfg)
    monkeypatch.setattr(
        DnseClient,
        "get_positions",
        lambda self: [
            {
                "symbol": "TCB", "status": "OPEN", "side": "NB", "quantity": 800,
                "accumulate_quantity": 1000, "closed_quantity": 200,
                "avg_cost_price": 24.5, "market_price": 25.2, "break_even_price": None,
                "unrealized_profit": 560.0, "price_unit": "nghìn đồng",
            }
        ],
    )
    monkeypatch.setattr(DnseClient, "get_account_summary", lambda self: {"account_no": "0001234567"})

    at = run_app_tab(cfg, "Danh mục")

    assert not at.exception
    assert any("Tài khoản DNSE" in md for md in markdown_values(at))
    dnse_tables = [
        df for df in at.dataframe
        if list(df.value.columns) == [
            "Mã", "KL nắm giữ", "Giá vốn TB (nghìn đ)", "Giá TT (nghìn đ)", "Dự báo ngày mai",
        ]
    ]
    assert len(dnse_tables) == 1
    rows = dnse_tables[0].value
    assert rows.iloc[0]["Mã"] == "TCB"
    assert rows.iloc[0]["KL nắm giữ"] == 800
    assert rows.iloc[0]["Giá vốn TB (nghìn đ)"] == pytest.approx(24.5)
    # DB test chưa có dự báo cho TCB → ô dự báo là "—"
    assert rows.iloc[0]["Dự báo ngày mai"] == "—"


def test_tab2_dnse_disabled_renders_nothing(app_env, tmp_path):
    cfg = make_cfg(tmp_path, dnse=False)
    app_env(cfg)
    at = run_app_tab(cfg, "Danh mục")

    assert not at.exception
    assert not any("Tài khoản DNSE" in md for md in markdown_values(at))


def test_tab2_dnse_api_error_shows_warning_not_crash(app_env, tmp_path, monkeypatch):
    cfg = make_cfg(tmp_path, dnse=True)
    app_env(cfg)
    monkeypatch.setattr(
        DnseClient, "get_positions",
        lambda self: (_ for _ in ()).throw(DnseAuthError("DNSE từ chối truy cập (HTTP 401)")),
    )

    at = run_app_tab(cfg, "Danh mục")

    assert not at.exception
    warnings = [str(w.value) for w in at.warning]
    assert any("Không lấy được vị thế từ DNSE" in w for w in warnings)


def test_tab2_dnse_positions_use_seeded_forecast(app_env, tmp_path, monkeypatch):
    """Cột 'Dự báo ngày mai' lấy từ forecast_summary + tomorrow_consensus."""
    cfg = make_cfg(tmp_path, dnse=True, db_name="seeded_dnse.db")
    seed_forecast_db(Path(cfg["database"]["path"]))
    app_env(cfg)
    monkeypatch.setattr(
        DnseClient,
        "get_positions",
        lambda self: [
            {"symbol": "TCB", "status": "OPEN", "side": "NB", "quantity": 800,
             "accumulate_quantity": 1000, "closed_quantity": 200, "avg_cost_price": 24.5,
             "market_price": 25.2, "break_even_price": None, "unrealized_profit": 560.0,
             "price_unit": "nghìn đồng"},
            {"symbol": "FPT", "status": "OPEN", "side": "NB", "quantity": 100,
             "accumulate_quantity": 100, "closed_quantity": 0, "avg_cost_price": 130.0,
             "market_price": 131.0, "break_even_price": None, "unrealized_profit": 100.0,
             "price_unit": "nghìn đồng"},
        ],
    )

    at = run_app_tab(cfg, "Danh mục")

    assert not at.exception
    dnse_tables = [
        df for df in at.dataframe
        if "Dự báo ngày mai" in list(df.value.columns)
    ]
    assert len(dnse_tables) == 1
    rows = dnse_tables[0].value.set_index("Mã")
    # TCB: DB seeded → "24.60 nghìn đ · P(tăng) 57% · 2/3 tăng" (0.575 → "{:.0%}")
    assert rows.loc["TCB", "Dự báo ngày mai"] == "24.60 nghìn đ · P(tăng) 57% · 2/3 tăng"
    # FPT: chưa huấn luyện → "—"
    assert rows.loc["FPT", "Dự báo ngày mai"] == "—"


# ---------------------------------------------------------------------------
# 6) Tab Lịch sử dự đoán — bảng predictions↔outcomes từ DB (đã seed)
# ---------------------------------------------------------------------------


def test_tab4_history_renders_with_seeded_db(app_env, tmp_path):
    cfg = make_cfg(tmp_path, db_name="seeded_history.db")
    seed_forecast_db(Path(cfg["database"]["path"]))
    app_env(cfg)
    at = run_app_tab(cfg, "Lịch sử dự đoán")

    assert not at.exception
    assert len(at.dataframe) >= 1
    history = at.dataframe[0].value
    assert "Giá dự báo (nghìn đ)" in history.columns
    assert (history["Mã"] == "TCB").all()
