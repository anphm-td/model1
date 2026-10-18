"""Dashboard Streamlit — giao diện người dùng tiếng Việt, 4 tab.

Chạy: ``streamlit run app/dashboard.py`` (từ thư mục gốc dự án).

4 tab (thanh điều hướng st.radio horizontal — cho phép đổi tab bằng lập trình
khi người dùng bấm "mở chi tiết" một mã):
- Tab 1 "Thị trường": bảng mã HOSE từ HoseSymbolDirectory (cache
  data/symbols.parquet), tìm kiếm theo mã/tên, nút "+ Thêm vào danh mục" TỪNG
  DÒNG (phân trang 20 dòng/trang); khi nguồn danh sách mã lỗi/trống cho phép
  thêm mã thủ công.
- Tab 2 "Danh mục": watchlist (data/watchlist.json qua src.watchlist), nút xoá
  từng mã, nút "mở chi tiết" đặt st.session_state["selected_ticker"] và chuyển
  sang tab Phân tích & dự báo.
- Tab 3 "Phân tích & dự báo": candlestick + RSI(14) + volume (plotly) từ
  data_loader.load_history (cache parquet, không ffill ngày nghỉ); nút
  "Huấn luyện & dự báo" gọi src.pipeline.run_pipeline(ticker) — CHỈ khi bấm
  nút, KHÔNG BAO GIỜ chạy pipeline khi load trang; bảng so sánh 4 model (giá,
  hướng, xác suất tăng) đọc từ SQLite qua src.forecast; biểu đồ backtest PNG +
  SHAP đọc từ reports/{ticker}/.
- Tab 4 "Lịch sử dự đoán": bảng predictions↔outcomes (src.db) lọc theo
  mã/model/thời gian + win rate & MAE tích lũy theo model; nút đối chiếu dự
  báo chưa settle với giá thực tế (src.db.settle_predictions).

Nguyên tắc:
- Chỉ GỌI tầng src/ (data_loader, watchlist, db, forecast, dnse_client,
  pipeline, features) — dashboard không tự tính feature/model, không đụng SQL
  trực tiếp ngoài đếm phục vụ lọc.
- Mọi lỗi bắt được đều hiển thị tiếng Việt rõ ràng (không nuốt im lặng).
- st.session_state["selected_ticker"] là mã đang chọn dùng chung toàn app;
  st.session_state["nav"] là tab đang mở.

Bổ sung gần đây:
- Banner TOÀN CỤC "DỰ BÁO PHIÊN MAI" dưới header, trước nav — forecast_summary
  được gọi ĐÚNG 1 LẦN ở đầu main flow cho ticker đang chọn, tab Phân tích &
  dự báo TÁI DÙNG kết quả đó (src.forecast.tomorrow_consensus gộp đồng thuận).
- Widget TradingView miễn phí (không cần key) qua st.components.v1.html:
  Advanced Chart + Technical Analysis (tab 3), Ticker Tape + Market Overview
  (tab 1) — cấu hình app.tradingview trong config.yaml; tắt → plotly fallback.
- Tab Danh mục có mục "Tài khoản DNSE" (src.dnse_client, CHỈ ĐỌC) chỉ hiện khi
  app.dnse.enabled=true VÀ token khác rỗng.
"""
from __future__ import annotations

import html as html_lib
import json
import logging
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from plotly.subplots import make_subplots

# --- Bootstrap đường dẫn: `streamlit run app/dashboard.py` đặt CWD/script dir
# là app/, còn package `src` nằm ở thư mục gốc dự án ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import db  # noqa: E402
from src.data_loader import load_history  # noqa: E402
from src.data_sources.cafef_source import HoseSymbolDirectory  # noqa: E402
from src.dnse_client import DnseClient, DnseError  # noqa: E402
from src.evaluation import InsufficientHistoryError  # noqa: E402
from src.features.price_features import PriceFeatures  # noqa: E402
from src.forecast import forecast_summary, tomorrow_consensus  # noqa: E402
from src.pipeline import PipelineError, load_config, run_pipeline  # noqa: E402
from src.watchlist import Watchlist, WatchlistError  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hằng số hiển thị
# ---------------------------------------------------------------------------

#: Thứ tự + tên 4 tab (tiếng Việt).
TABS: tuple[str, ...] = ("Thị trường", "Danh mục", "Phân tích & dự báo", "Lịch sử dự đoán")

#: Tên 4 họ model theo quy ước src/pipeline.py (thứ tự cố định cho báo cáo).
MODEL_VN: dict[str, str] = {
    "naive": "Naive (persistence)",
    "arima": "ARIMA",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}

#: Nhãn tiếng Việt cho hướng 3 lớp (quy ước {up, down, neutral} của src/db).
DIRECTION_VN: dict[str, str] = {"up": "▲ Tăng", "down": "▼ Giảm", "neutral": "— Đi ngang"}

#: Số dòng mỗi trang ở tab Thị trường (mỗi dòng có nút riêng → phải phân trang).
MARKET_PAGE_SIZE: int = 20

#: Số feature lớn nhất hiển thị trên biểu đồ SHAP.
SHAP_TOP_N: int = 15


# ---------------------------------------------------------------------------
# Tầng đọc dữ liệu (cache Streamlit — KHÔNG chạy pipeline ở đây)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Đang đọc config.yaml…")
def load_app_config() -> dict[str, Any]:
    """Đọc config.yaml qua src.pipeline.load_config (cache cho cả session)."""
    return load_config()


def _resolve(path_text: str) -> Path:
    """Đường dẫn tương đối trong config → tuyệt đối theo gốc dự án."""
    path = Path(path_text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def watchlist() -> Watchlist:
    """Watchlist theo đường dẫn config.yaml: app.watchlist_path."""
    cfg = load_app_config()
    raw = str(cfg.get("app", {}).get("watchlist_path", "data/watchlist.json"))
    return Watchlist(_resolve(raw))


def db_path() -> Path:
    """Đường dẫn SQLite theo config.yaml: database.path."""
    cfg = load_app_config()
    raw = str(cfg.get("database", {}).get("path", "data/app.db"))
    return _resolve(raw)


def open_db() -> sqlite3.Connection:
    """Mở kết nối DB và bảo đảm schema tồn tại (an toàn khi DB còn trống)."""
    conn = sqlite3.connect(str(db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    db.init_db(conn)
    return conn


@st.cache_data(ttl=3600, show_spinner="Đang tải dữ liệu giá (cache parquet + cập nhật nguồn)…")
def load_price(ticker: str) -> tuple[pd.DataFrame, pd.Series]:
    """Giá OHLCV ngày + RSI(14) cho ``ticker``.

    Gọi ``src.data_loader.load_history`` (cache parquet data/raw + processed,
    cập nhật tăng dần, KHÔNG ffill ngày nghỉ). RSI lấy từ
    ``src.features.price_features.PriceFeatures`` (causal — tái dùng tầng src,
    dashboard không tự tính chỉ báo).

    Returns:
        (ohlcv, rsi_14) — ohlcv khung hợp đồng chuẩn của dự án; rsi_14 là
        Series cùng index (NaN ở phần warm-up).
    """
    frame = load_history(ticker)
    rsi = pd.Series(dtype="float64", index=frame.index, name="rsi_14")
    if not frame.empty:
        rsi = PriceFeatures().provide(frame)["rsi_14"]
    return frame, rsi


@st.cache_data(ttl=3600, show_spinner="Đang tải danh sách mã HOSE…")
def load_symbols() -> tuple[pd.DataFrame, str | None, str | None]:
    """Danh sách mã HOSE qua ``HoseSymbolDirectory`` (cache data/symbols.parquet).

    Returns:
        (frame, source, message) — frame cột [symbol] (+ name/industry nếu
        nguồn có thật); message khác None nghĩa là đang dùng cache cũ hoặc
        nguồn lỗi (hiển thị cảnh báo trung thực cho người dùng).
    """
    result = HoseSymbolDirectory().get()
    return result.frame, result.source, result.message


@st.cache_data(show_spinner=False)
def read_report_csv(kind: str, ticker: str) -> pd.DataFrame | None:
    """Đọc artifact CSV từ reports/{ticker}/: ``kind`` ∈ {"metrics", "shap_h1", ...}.

    Returns:
        DataFrame hoặc None khi file chưa tồn tại (chưa từng huấn luyện).
    """
    if kind == "metrics":
        path = PROJECT_ROOT / "reports" / ticker / "metrics.csv"
    elif kind.startswith("shap_h"):
        path = PROJECT_ROOT / "reports" / ticker / f"{kind}.csv"
    else:  # pragma: no cover — chỉ dùng nội bộ với kind hợp lệ
        raise ValueError(f"kind không hỗ trợ: {kind!r}")
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 — artifact hỏng hiển thị cảnh báo, không chết
        logger.warning("Đọc %s lỗi: %s", path, exc)
        return None


@st.cache_data(show_spinner=False)
def read_backtest_png(ticker: str, horizon: int) -> bytes | None:
    """Nội dung reports/{ticker}/backtest_h{h}.png (None nếu chưa có)."""
    path = PROJECT_ROOT / "reports" / ticker / f"backtest_h{horizon}.png"
    if not path.exists():
        return None
    return path.read_bytes()


# ---------------------------------------------------------------------------
# Tiện ích hiển thị
# ---------------------------------------------------------------------------


def fmt_price(value: Any) -> str:
    """Giá nghìn đồng → chuỗi 2 số lẻ (None/NaN → '—')."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not pd.notna(number):
        return "—"
    return f"{number:,.2f}"


def fmt_pct(value: Any, digits: int = 1, signed: bool = False) -> str:
    """Tỉ lệ (0.012) → chuỗi % ('1.2%' / '+1.2%'); None/NaN → '—'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not pd.notna(number):
        return "—"
    sign = "+" if signed and number > 0 else ""
    return f"{sign}{number * 100:.{digits}f}%"


def fmt_direction(value: Any) -> str:
    """Hướng 3 lớp DB → nhãn tiếng Việt (None → '—')."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return DIRECTION_VN.get(str(value), str(value))


# ---------------------------------------------------------------------------
# Banner "DỰ BÁO PHIÊN MAI" (TOÀN CỤC) + widget TradingView (miễn phí)
# ---------------------------------------------------------------------------


def escape_html(value: Any) -> str:
    """Escape giá trị chèn vào HTML (ticker/tên từ watchlist, người dùng nhập)."""
    return html_lib.escape(str(value), quote=True)


def tradingview_settings(cfg: dict[str, Any]) -> tuple[bool, str]:
    """(enabled, exchange) từ config ``app.tradingview`` (mặc định: bật, HOSE)."""
    tv = (cfg or {}).get("app", {}).get("tradingview", {}) or {}
    enabled = bool(tv.get("enabled", True))
    exchange = str(tv.get("exchange", "HOSE")).strip().upper() or "HOSE"
    return enabled, exchange


def tv_symbol(ticker: str, exchange: str) -> str:
    """'HOSE:TCB' — chỉ giữ ký tự hợp lệ cho symbol TradingView (chống chèn JS)."""
    clean = lambda s: re.sub(r"[^A-Z0-9._-]", "", str(s).strip().upper())  # noqa: E731
    return f"{clean(exchange)}:{clean(ticker)}"


def render_tv_advanced_chart(ticker: str, exchange: str, *, height: int = 620) -> None:
    """TradingView Advanced Chart (tv.js) — nến + RSI(14) + Volume, khung ngày.

    Dùng ``new TradingView.widget`` đúng theo API tv.js miễn phí (không cần key);
    PHẢI đặt chiều cao tường minh cho container vì iframe của
    st.components.v1.html không tự giãn theo widget.
    """
    widget_config = {
        "container_id": "tv_advanced_chart",
        "symbol": tv_symbol(ticker, exchange),
        "interval": "D",
        "theme": "light",
        "studies": ["RSI@tv-basicstudies", "Volume@tv-basicstudies"],
        "autosize": True,
        "height": height,
        "locale": "vi_VN",
        "timezone": "Asia/Ho_Chi_Minh",
    }
    code = (
        f'<div id="tv_advanced_chart" style="height:{height}px;width:100%"></div>\n'
        '<script src="https://s3.tradingview.com/tv.js"></script>\n'
        f'<script type="text/javascript">\nnew TradingView.widget({json.dumps(widget_config)});\n</script>'
    )
    components.html(code, height=height, scrolling=False)


def render_tv_embed(payload: dict[str, Any], embed_name: str, *, height: int) -> None:
    """Widget dạng embed chính thức của TradingView (ticker-tape, market-overview,
    technical-analysis) — script ``embed-widget-{name}.js`` với JSON cấu hình."""
    code = (
        '<div class="tradingview-widget-container" style="height:{h}px;width:100%">\n'
        '  <div class="tradingview-widget-container__widget" style="height:{h}px;width:100%"></div>\n'
        "</div>\n"
        '<script src="https://s3.tradingview.com/external-embedding/embed-widget-{name}.js" async>\n'
        "{json}\n"
        "</script>"
    ).format(h=height, name=embed_name, json=json.dumps(payload, indent=2))
    components.html(code, height=height, scrolling=False)


def render_tv_technical_analysis(ticker: str, exchange: str, *, height: int = 400) -> None:
    """Widget Technical Analysis (gauge oscillators + moving averages) của TradingView."""
    render_tv_embed(
        {
            "symbol": tv_symbol(ticker, exchange),
            "interval": "1D",
            "width": "100%",
            "height": height,
            "isTransparent": False,
            "theme": "light",
            "locale": "vi_VN",
            "showIntervalTabs": True,
        },
        "technical-analysis",
        height=height,
    )


def render_tv_ticker_tape(entries: list[tuple[str, str]], exchange: str, *, height: int = 46) -> None:
    """Ticker Tape mảnh — VN-Index + danh mục đang theo dõi."""
    render_tv_embed(
        {
            "symbols": [
                {"proName": tv_symbol(ticker, exchange), "title": title}
                for ticker, title in entries
                if str(ticker).strip()
            ],
            "showSymbolLogo": False,
            "colorTheme": "light",
            "isTransparent": False,
            "displayMode": "adaptive",
            "locale": "vi_VN",
        },
        "ticker-tape",
        height=height,
    )


def render_tv_market_overview(watchlist_tickers: list[str], exchange: str, *, height: int = 400) -> None:
    """Widget Market Overview — tab 'VN-Index' + tab 'Danh mục' (nếu có)."""
    tabs: list[dict[str, Any]] = [
        {"title": "VN-Index", "symbols": [{"s": tv_symbol("VNINDEX", exchange), "d": "VN-Index"}]}
    ]
    if watchlist_tickers:
        tabs.append(
            {
                "title": "Danh mục",
                "symbols": [{"s": tv_symbol(t, exchange), "d": str(t)} for t in watchlist_tickers],
            }
        )
    render_tv_embed(
        {
            "tabs": tabs,
            "colorTheme": "light",
            "locale": "vi_VN",
            "width": "100%",
            "height": height,
            "isTransparent": False,
            "showSymbolLogo": False,
        },
        "market-overview",
        height=height,
    )


def format_tomorrow_cell(consensus: dict[str, Any]) -> str:
    """Ô 'Dự báo ngày mai' cho bảng DNSE: giá · P(tăng) · đồng thuận báo tăng."""
    if not consensus.get("has_forecast"):
        return "—"
    price_text = fmt_price(consensus.get("price_mean"))  # None → "—"
    p_up = consensus.get("p_up_mean")
    p_text = f"{p_up:.0%}" if p_up is not None else fmt_direction(None)
    n_up = int(consensus.get("n_up", 0))
    n_models = int(consensus.get("n_models", 0))
    return f"{price_text} nghìn đ · P(tăng) {p_text} · {n_up}/{n_models} tăng"


def render_forecast_banner(summary: dict[str, Any]) -> None:
    """Banner TOÀN CỤC "DỰ BÁO PHIÊN MAI" — một dòng card dưới header, trước nav.

    Dùng ``tomorrow_consensus`` trên ``summary["predictions"]`` (horizon=1);
    ``summary`` là kết quả forecast_summary gọi đúng 1 lần ở đầu main flow.
    None → "—" như fmt_price/fmt_direction; chưa huấn luyện/chưa có dự báo h=1
    → cảnh báo "hãy Huấn luyện & dự báo".
    """
    ticker = escape_html(summary.get("ticker", ""))
    run_info = summary.get("run") or {}
    note = ""
    if summary.get("has_run") and run_info:
        note += (
            f"Nguồn: run #{run_info.get('id')} · train lúc "
            f"{str(run_info.get('trained_at') or '—')} · dữ liệu đến "
            f"{str(run_info.get('data_until') or '—')} · "
        )
    note += "tham khảo, không phải khuyến nghị."
    note_html = escape_html(note)

    consensus = tomorrow_consensus(summary.get("predictions") or [])
    if not summary.get("has_run") or not consensus.get("has_forecast"):
        st.markdown(
            "<div style=\"padding:0.6rem 1rem;border-radius:0.6rem;border:1px solid #ffe082;"
            "background:#fff8e1;\">"
            f"<b>⚠️ Chưa có dự báo cho {ticker} — hãy Huấn luyện &amp; dự báo</b><br>"
            f"<small>{note_html}</small></div>",
            unsafe_allow_html=True,
        )
        return

    price_mean = consensus.get("price_mean")
    price_html = (
        f"<b>{price_mean:.2f}</b> nghìn đ (≈ {round(price_mean * 1000):,} VND)"
        if price_mean is not None
        else fmt_price(None)
    )
    n_up = int(consensus.get("n_up", 0))
    n_down = int(consensus.get("n_down", 0))
    n_neutral = int(consensus.get("n_neutral", 0))
    n_models = int(consensus.get("n_models", 0))
    direction_html = (
        f"{n_up} tăng / {n_down} giảm / {n_neutral} ngang"
        if n_up + n_down + n_neutral
        else fmt_direction(None)
    )
    p_up = consensus.get("p_up_mean")
    p_up_html = f"P(tăng) <b>{p_up:.0%}</b>" if p_up is not None else "P(tăng) —"
    target = escape_html(str(consensus.get("target_date") or "—"))

    line = (
        f"🔮 <b>DỰ BÁO PHIÊN MAI {target} — {ticker}</b>"
        f" · giá {price_html}"
        f" · hướng: {direction_html}"
        f" · {p_up_html}"
        f" · đồng thuận {n_up}/{n_models} model báo tăng"
    )
    st.markdown(
        "<div style=\"padding:0.6rem 1rem;border-radius:0.6rem;border:1px solid #c5cae9;"
        "background:linear-gradient(90deg,#e8eaf6 0%,#f8f9fa 100%);\">"
        f"{line}<br><small>{note_html}</small></div>",
        unsafe_allow_html=True,
    )


def render_plotly_price_chart(
    shown: pd.DataFrame, rsi_shown: pd.Series, ticker: str
) -> None:
    """Fallback plotly nội bộ (khi app.tradingview.enabled=false): candlestick +
    RSI(14) + volume từ dữ liệu app — nguồn giá của chính pipeline."""

    ohlc_ok = shown[["open", "high", "low"]].notna().all(axis=1)
    candles = shown.loc[ohlc_ok]
    close_only = shown.loc[~ohlc_ok]

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.62, 0.19, 0.19),
        vertical_spacing=0.03,
        subplot_titles=(f"{ticker} — giá (nghìn đồng)", "RSI(14)", "Khối lượng"),
    )
    fig.add_trace(
        go.Candlestick(
            x=candles.index,
            open=candles["open"], high=candles["high"], low=candles["low"], close=candles["close"],
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
            name="Nến OHLC",
        ),
        row=1, col=1,
    )
    if not close_only.empty:
        # Lịch sử cũ từ nguồn HOSE chính thức chỉ có close + volume — vẽ đường close,
        # trung thực thay vì bịa nến (KHÔNG ffill/bịa OHLC).
        fig.add_trace(
            go.Scatter(
                x=close_only.index, y=close_only["close"], mode="lines",
                line={"color": "#9e9e9e", "width": 1.2}, name="close (chỉ có close+volume)",
            ),
            row=1, col=1,
        )
    fig.update_layout(xaxis_rangeslider_visible=False)

    fig.add_trace(
        go.Scatter(
            x=rsi_shown.index, y=rsi_shown, mode="lines",
            line={"color": "#7e57c2", "width": 1.6}, name="RSI(14)",
        ),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dot", line_color="#ef5350", row=2, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="#26a69a", row=2, col=1)

    volume_colors = [
        "#26a69a" if i == 0 or shown["close"].iloc[i] >= shown["close"].iloc[i - 1] else "#ef5350"
        for i in range(len(shown))
    ]
    fig.add_trace(
        go.Bar(x=shown.index, y=shown["volume"], marker_color=volume_colors, name="Khối lượng", showlegend=False),
        row=3, col=1,
    )
    fig.update_yaxes(title_text="RSI", row=2, col=1)
    fig.update_yaxes(title_text="Volume", row=3, col=1)
    fig.update_layout(
        height=760,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")


def selected_ticker() -> str:
    """Mã đang chọn (st.session_state['selected_ticker'], mặc định theo config)."""
    cfg = load_app_config()
    default = str(cfg.get("default_ticker", "TCB")).strip().upper()
    current = str(st.session_state.get("selected_ticker", "") or "").strip().upper()
    return current or default


def open_ticker_detail(ticker: str) -> None:
    """Đặt mã đang chọn + chuyển sang tab Phân tích & dự báo (dùng session_state)."""
    st.session_state["selected_ticker"] = str(ticker).strip().upper()
    st.session_state["nav"] = "Phân tích & dự báo"
    st.rerun()


def db_tickers() -> list[str]:
    """Các mã đã có trong bảng predictions/runs (phục vụ bộ lọc lịch sử)."""
    conn = open_db()
    try:
        rows = conn.execute(
            "SELECT DISTINCT ticker FROM predictions "
            "UNION SELECT DISTINCT ticker FROM runs ORDER BY ticker"
        ).fetchall()
        return [str(row["ticker"]) for row in rows]
    finally:
        conn.close()


def known_tickers() -> list[str]:
    """Gộp: mã đang chọn + watchlist + mã có trong DB — cho selectbox chọn mã."""
    tickers: set[str] = {selected_ticker()}
    try:
        tickers.update(watchlist().list_tickers())
    except WatchlistError:
        pass  # watchlist hỏng → tab Danh mục đã báo lỗi riêng
    tickers.update(db_tickers())
    return sorted(tickers)


# ---------------------------------------------------------------------------
# Khung trang: header + thanh điều hướng 4 tab
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Dự báo cổ phiếu HOSE",
    page_icon="📈",
    layout="wide",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")

cfg = load_app_config()
app_title = str(cfg.get("project", {}).get("title", "Dự báo cổ phiếu sàn HOSE"))

header_left, header_right = st.columns([2.6, 1])
with header_left:
    st.title(f"📈 {app_title}")
with header_right:
    st.markdown(
        f"Mã đang chọn: **{selected_ticker()}**",
        help="Đổi bằng nút “mở chi tiết” ở tab Danh mục hoặc selectbox trong tab Phân tích & dự báo.",
    )
    st.caption(f"SQLite: {db_path().name} · watchlist: data/watchlist.json")

# --- Banner "DỰ BÁO PHIÊN MAI" (TOÀN CỤC, dưới header — trước nav) ---
# forecast_summary được gọi ĐÚNG 1 LẦN ở đầu main flow cho mã đang chọn; tab
# "Phân tích & dự báo" TÁI DÙNG biến `summary` này (chỉ gọi lại nếu mã vừa đổi
# trong run đó qua selectbox — xem phần tab 3).
conn = open_db()
try:
    summary = forecast_summary(conn, selected_ticker())
finally:
    conn.close()
render_forecast_banner(summary)

if "nav" not in st.session_state:
    st.session_state["nav"] = TABS[0]
# Không dùng `key` cho thanh điều hướng: gán st.session_state[key] SAU khi widget
# đã instantiate (vd. nút "mở chi tiết" ở tab Danh mục) sẽ raise StreamlitAPIException.
# Pattern không-key + index đồng bộ: session_state["nav"] là nguồn sự thật, lựa chọn
# của người dùng được sync ngược lại trong cùng run. Dùng st.radio(horizontal) thay
# st.pills vì AppTest (streamlit.testing) replay nhầm giá trị single-mode của pills.
nav = st.radio(
    "Tab",
    list(TABS),
    index=list(TABS).index(st.session_state["nav"]),
    horizontal=True,
    label_visibility="collapsed",
)
if nav in TABS and nav != st.session_state["nav"]:
    st.session_state["nav"] = nav
current_tab = st.session_state["nav"]
st.divider()

# ===========================================================================
# TAB 1 — THỊ TRƯỜNG
# ===========================================================================
if current_tab == "Thị trường":
    st.subheader("Thị trường — mã cổ phiếu sàn HOSE")

    # --- Ticker Tape mảnh (TradingView): VN-Index + danh mục đang theo dõi ---
    tv_enabled, tv_exchange = tradingview_settings(cfg)
    if tv_enabled:
        try:
            watch_tickers_tv = watchlist().list_tickers()
        except WatchlistError:
            watch_tickers_tv = []
        render_tv_ticker_tape(
            [("VNINDEX", "VN-Index"), *[(t, str(t)) for t in watch_tickers_tv]],
            tv_exchange,
        )
        st.caption("Băng giá trực tiếp từ TradingView — VN-Index và các mã trong danh mục.")

    symbols_frame, symbols_source, symbols_message = load_symbols()
    if symbols_message:
        st.warning(f"Nguồn danh sách mã: {symbols_message}")
    has_name = "name" in symbols_frame.columns
    st.caption(
        f"{len(symbols_frame)} mã · nguồn: {symbols_source or 'không có'}"
        + ("" if has_name else " · nguồn hiện chỉ cung cấp MÃ (không có tên công ty) — tìm theo mã")
    )

    # --- Tìm kiếm theo mã/tên ---
    query = st.text_input(
        "Tìm kiếm theo mã hoặc tên",
        value="",
        placeholder="VD: TCB hoặc Techcom",
        help="So khớp không phân biệt hoa/thường trên mã" + (", tên công ty" if has_name else "") + ".",
    )
    frame = symbols_frame.copy()
    if query.strip():
        q = query.strip()
        mask = frame["symbol"].str.contains(q, case=False, na=False, regex=False)
        if has_name:
            mask = mask | frame["name"].fillna("").str.contains(q, case=False, na=False, regex=False)
        frame = frame.loc[mask].reset_index(drop=True)

    try:
        in_watchlist = set(watchlist().list_tickers())
    except WatchlistError as exc:
        st.error(f"Watchlist không đọc được: {exc}")
        in_watchlist = set()

    if frame.empty:
        st.info("Không có mã nào khớp tìm kiếm." if query.strip() else "Danh sách mã rỗng.")
    else:
        # --- Phân trang (mỗi dòng có nút riêng nên không render > 20 dòng) ---
        if "market_page" not in st.session_state:
            st.session_state["market_page"] = 0
        if st.session_state.get("market_query") != query:
            st.session_state["market_query"] = query
            st.session_state["market_page"] = 0
        n_pages = max(1, -(-len(frame) // MARKET_PAGE_SIZE))
        st.session_state["market_page"] = min(int(st.session_state["market_page"]), n_pages - 1)
        page = int(st.session_state["market_page"])

        nav_prev, nav_info, nav_next = st.columns([1, 2, 1])
        if nav_prev.button("◀ Trang trước", disabled=page == 0, width="stretch"):
            st.session_state["market_page"] = page - 1
            st.rerun()
        nav_info.markdown(f"**Trang {page + 1}/{n_pages}** — {len(frame)} mã khớp")
        if nav_next.button("Trang sau ▶", disabled=page >= n_pages - 1, width="stretch"):
            st.session_state["market_page"] = page + 1
            st.rerun()

        page_frame = frame.iloc[page * MARKET_PAGE_SIZE : (page + 1) * MARKET_PAGE_SIZE]

        head = st.columns([0.9, 3.1, 2.2])
        head[0].markdown("**Mã**")
        head[1].markdown("**Tên công ty**" if has_name else "**Danh mục**")
        head[2].markdown("**Thêm vào danh mục**")
        st.divider()

        for row in page_frame.itertuples(index=False):
            symbol = str(row.symbol).upper()
            cols = st.columns([0.9, 3.1, 2.2])
            cols[0].markdown(f"**{symbol}**")
            if has_name:
                name = getattr(row, "name", None)
                cols[1].write(str(name) if name and str(name) != "nan" else "—")
            if symbol in in_watchlist:
                cols[2].button(
                    "✓ Đã có trong danh mục",
                    key=f"has_{symbol}_{page}",
                    disabled=True,
                    width="stretch",
                )
            elif cols[2].button(
                "➕ Thêm vào danh mục",
                key=f"add_{symbol}_{page}",
                type="primary",
                width="stretch",
            ):
                try:
                    added = watchlist().add(symbol)
                except (WatchlistError, ValueError) as exc:
                    st.error(f"Không thêm được {symbol}: {exc}")
                else:
                    if added:
                        st.toast(f"Đã thêm {symbol} vào danh mục", icon="✅")
                    else:
                        st.toast(f"{symbol} đã có trong danh mục", icon="ℹ️")
                    st.rerun()
            st.divider()

    # --- Thêm thủ công (khi nguồn danh sách mã lỗi/trống) ---
    with st.expander("➕ Thêm mã thủ công"):
        manual = st.text_input(
            "Mã cổ phiếu", value="", placeholder="VD: TCB", max_chars=10
        ).strip().upper()
        if st.button("Thêm mã này", key="add_manual", type="primary"):
            if not manual:
                st.warning("Nhập mã cổ phiếu trước khi thêm.")
            else:
                try:
                    added = watchlist().add(manual)
                except (WatchlistError, ValueError) as exc:
                    st.error(f"Không thêm được {manual}: {exc}")
                else:
                    if added:
                        st.toast(f"Đã thêm {manual} vào danh mục", icon="✅")
                        st.rerun()
                    else:
                        st.info(f"{manual} đã có trong danh mục.")

    # --- Market Overview (TradingView): VN-Index + watchlist ---
    if tv_enabled:
        st.divider()
        st.markdown("### Bối cảnh thị trường")
        render_tv_market_overview(watch_tickers_tv, tv_exchange)
        st.caption(
            f"TradingView Market Overview — {tv_symbol('VNINDEX', tv_exchange)} và danh mục "
            "đang theo dõi (widget miễn phí, không cần key; dữ liệu ngoài app)."
        )

# ===========================================================================
# TAB 2 — DANH MỤC (WATCHLIST)
# ===========================================================================
elif current_tab == "Danh mục":
    st.subheader("Danh mục theo dõi")
    st.caption("Mã lưu tại data/watchlist.json — bấm “📈 Chi tiết” để mở phân tích & dự báo cho mã đó.")

    try:
        entries = watchlist().entries()
    except WatchlistError as exc:
        st.error(f"Watchlist không đọc được (kiểm tra tệp JSON): {exc}")
        entries = []

    if not entries:
        st.info("Danh mục trống — chuyển tab “Thị trường” để thêm mã, hoặc thêm mã thủ công ở đó.")
    else:
        summary_rows: list[dict[str, Any]] = []
        for entry in entries:
            ticker = str(entry["ticker"])
            close_value: float | None = None
            close_date: str | None = None
            trained_at: str | None = None
            # Giá cuối đọc từ cache processed (CHỈ file local — không gọi mạng từng dòng)
            processed = PROJECT_ROOT / "data" / "processed" / f"{ticker}.parquet"
            if processed.exists():
                try:
                    local = pd.read_parquet(processed)
                    if not local.empty:
                        close_value = float(local["close"].iloc[-1])
                        idx = local.index[-1]
                        close_date = pd.Timestamp(idx).strftime("%Y-%m-%d")
                except Exception as exc:  # noqa: BLE001 — cache hỏng → để trống ô giá
                    logger.warning("Đọc cache processed cho %s lỗi: %s", ticker, exc)
            conn = open_db()
            try:
                runs = db.list_runs(conn, ticker)
                trained_at = str(runs[0]["trained_at"])[:16] if runs else None
            finally:
                conn.close()
            summary_rows.append(
                {
                    "Mã": ticker,
                    "Giá đóng cửa (nghìn đ)": close_value,
                    "Phiên cuối": close_date,
                    "Huấn luyện gần nhất": trained_at or "chưa huấn luyện",
                }
            )

        st.dataframe(
            pd.DataFrame(summary_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Giá đóng cửa (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
            },
        )

        st.divider()
        for entry in entries:
            ticker = str(entry["ticker"])
            c_sym, c_added, c_detail, c_remove = st.columns([1.2, 2.4, 1.4, 1.4])
            c_sym.markdown(f"**{ticker}**")
            c_added.caption(f"Thêm lúc {str(entry.get('added_at', ''))[:16]}")
            if c_detail.button("📈 Chi tiết", key=f"detail_{ticker}", width="stretch"):
                open_ticker_detail(ticker)
            if c_remove.button("🗑 Xoá", key=f"remove_{ticker}", width="stretch"):
                try:
                    removed = watchlist().remove(ticker)
                except (WatchlistError, ValueError) as exc:
                    st.error(f"Không xoá được {ticker}: {exc}")
                else:
                    if removed and st.session_state.get("selected_ticker") == ticker:
                        st.session_state.pop("selected_ticker", None)  # về mã mặc định
                    st.toast(f"Đã xoá {ticker} khỏi danh mục", icon="🗑️")
                    st.rerun()
            st.divider()

    # --- Tài khoản DNSE (CHỈ ĐỌC) — chỉ render khi bật trong config VÀ có token ---
    # disabled/không token → KHÔNG render gì; lỗi API → st.warning, không crash.
    dnse_cfg = (cfg.get("app", {}) or {}).get("dnse", {}) or {}
    dnse_token = str(dnse_cfg.get("jwt_token", "") or "").strip()
    dnse_account = str(dnse_cfg.get("account_no", "") or "").strip()
    if bool(dnse_cfg.get("enabled", False)) and dnse_token and dnse_account:
        st.divider()
        st.markdown("### 🏦 Tài khoản DNSE — vị thế & dự báo")
        st.caption(
            f"Tiểu khoản {escape_html(dnse_account)} · DNSE LightSpeed API CHỈ ĐỌC "
            "(không đặt/sửa/hủy lệnh) · giá đã quy đổi VND thô → nghìn đồng."
        )
        dnse_client = DnseClient(dnse_token, dnse_account)
        try:
            positions = dnse_client.get_positions()
        except DnseError as exc:
            st.warning(f"Không lấy được vị thế từ DNSE: {exc}")
        except Exception as exc:  # noqa: BLE001 — lỗi không lường trước vẫn không được chết tab
            logger.warning("DNSE get_positions lỗi: %s: %s", type(exc).__name__, exc)
            st.warning(f"Không lấy được vị thế từ DNSE: {type(exc).__name__}: {exc}")
        else:
            if not positions:
                st.info("Tiểu khoản này hiện KHÔNG có deal nắm giữ nào (status OPEN).")
            else:
                conn = open_db()
                try:
                    dnse_rows: list[dict[str, Any]] = []
                    for pos in positions:
                        symbol = str(pos.get("symbol") or "").upper()
                        forecast_cell = "—"
                        if symbol:
                            try:
                                pos_summary = forecast_summary(conn, symbol)
                                forecast_cell = format_tomorrow_cell(
                                    tomorrow_consensus(pos_summary.get("predictions") or [])
                                )
                            except Exception as exc:  # noqa: BLE001 — lỗi 1 mã không chết cả bảng
                                logger.warning("Đọc dự báo cho %s lỗi: %s", symbol, exc)
                                forecast_cell = "— lỗi đọc dự báo"
                        dnse_rows.append(
                            {
                                "Mã": symbol or "—",
                                "KL nắm giữ": pos.get("quantity"),
                                "Giá vốn TB (nghìn đ)": pos.get("avg_cost_price"),
                                "Giá TT (nghìn đ)": pos.get("market_price"),
                                "Dự báo ngày mai": forecast_cell,
                            }
                        )
                finally:
                    conn.close()
                st.dataframe(
                    pd.DataFrame(dnse_rows),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Giá vốn TB (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
                        "Giá TT (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
                    },
                )
                st.caption(
                    "KL = khối lượng đang nắm giữ (tích lũy − đã đóng, KHÔNG nhân 1000). "
                    "“Dự báo ngày mai” = đồng thuận horizon=1 của app cho từng mã — giá nghìn đ, "
                    "P(tăng) và số model báo tăng."
                )

# ===========================================================================
# TAB 3 — PHÂN TÍCH & DỰ BÁO
# ===========================================================================
elif current_tab == "Phân tích & dự báo":
    ticker = selected_ticker()
    st.subheader(f"Phân tích & dự báo — {ticker}")

    # --- Chọn mã + khung xem ---
    options = known_tickers()
    if ticker not in options:
        options = sorted({ticker, *options})
    col_ticker, col_window, col_train = st.columns([1.2, 1.6, 2.0])
    choice = col_ticker.selectbox(
        "Mã phân tích",
        options,
        index=options.index(ticker),
    )
    if choice != ticker:
        st.session_state["selected_ticker"] = choice
        ticker = choice
    window_label = col_window.selectbox(
        "Khung xem (phiên)",
        ("90 phiên gần nhất", "180 phiên gần nhất", "250 phiên gần nhất",
         "500 phiên gần nhất", "Toàn bộ lịch sử"),
        index=2,
    )
    window: int | None = int(window_label.split()[0]) if "Toàn bộ" not in window_label else None

    # --- Nút Huấn luyện & dự báo: pipeline CHỈ chạy trong nhánh này ---
    last_run_caption = ""
    conn = open_db()
    try:
        runs_for_caption = db.list_runs(conn, ticker)
        last_run_caption = (
            f"Lần train gần nhất: {str(runs_for_caption[0]['trained_at'])[:16]} (run #{runs_for_caption[0]['id']})"
            if runs_for_caption
            else "Mã này CHƯA được huấn luyện — bấm nút để chạy pipeline."
        )
    finally:
        conn.close()
    col_train.markdown(last_run_caption)
    if col_train.button("🚀 Huấn luyện & dự báo", key="train_btn", type="primary", width="stretch"):
        with st.status(
            f"Đang huấn luyện & dự báo cho {ticker} (walk-forward 4 model × các horizon)…",
            expanded=True,
        ) as status:
            st.write("Bước 1/6 — đối chiếu các dự báo cũ với giá thực tế (settle)…")
            st.write("Bước 2/6 — tải/cập nhật dữ liệu giá (cache parquet, không ffill ngày nghỉ)…")
            st.write("Bước 3/6 — sinh feature causal + target theo horizons…")
            st.write("Bước 4/6 — walk-forward: naive, ARIMA, LightGBM, LSTM (purge + embargo)…")
            st.write("Bước 5/6 — retrain toàn khung + dự báo phiên kế tiếp…")
            st.write("Bước 6/6 — ghi DB + reports/{ticker}/…")
            try:
                result = run_pipeline(ticker, cfg)
            except (PipelineError, InsufficientHistoryError) as exc:
                status.update(label="Huấn luyện thất bại", state="error", expanded=True)
                st.error(f"Pipeline dừng với lỗi có chủ đích: {exc}")
            except Exception as exc:  # noqa: BLE001 — hiển thị đầy đủ, không nuốt
                status.update(label="Huấn luyện thất bại (lỗi không lường trước)", state="error", expanded=True)
                st.error(f"Pipeline dừng với lỗi: {type(exc).__name__}: {exc}")
                with st.expander("Chi tiết lỗi (debug)"):
                    st.exception(exc)
            else:
                status.update(
                    label=(
                        f"Hoàn tất — run #{result.run_id}: {result.n_sessions} phiên "
                        f"(đến {result.data_until}), settle {result.settled}, re-point {result.repointed}"
                    ),
                    state="complete",
                    expanded=False,
                )
                st.session_state["last_run_result"] = result.to_dict()
                load_price.clear()  # pipeline có thể đã cập nhật cache parquet

    # --- Giá + chỉ báo (chỉ đọc cache — không phải pipeline) ---
    price_frame, rsi_series = load_price(ticker)
    if price_frame.empty:
        st.error(
            f"Không tải được dữ liệu giá cho {ticker} (nguồn lỗi hoặc mã không tồn tại). "
            "Kiểm tra kết nối/mã rồi thử lại."
        )
        st.stop()

    shown = price_frame.tail(window) if window else price_frame
    rsi_shown = rsi_series.reindex(shown.index)

    last_close = float(price_frame["close"].iloc[-1])
    prev_close = float(price_frame["close"].iloc[-2]) if len(price_frame) >= 2 else last_close
    change_1d = last_close / prev_close - 1.0 if prev_close > 0 else 0.0
    last_rsi = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else None
    last_volume = float(price_frame["volume"].iloc[-1])

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Giá đóng cửa (nghìn đ)", f"{last_close:,.2f}", f"{change_1d:+.2%} so với 1 phiên")
    kpi2.metric("RSI(14)", f"{last_rsi:.1f}" if last_rsi is not None else "—")
    kpi3.metric("Khối lượng (triệu cp)", f"{last_volume / 1e6:,.2f}")
    kpi4.metric("Số phiên trong khung xem", f"{len(shown):,}")

    # --- Biểu đồ giá: TradingView Advanced Chart (bật) hoặc plotly nội bộ (fallback) ---
    tv_enabled, tv_exchange = tradingview_settings(cfg)
    if tv_enabled:
        render_tv_advanced_chart(ticker, tv_exchange, height=620)
        st.caption(
            f"Biểu đồ {tv_symbol(ticker, tv_exchange)} từ TradingView — khung ngày, theme sáng, "
            "kèm RSI(14) + Volume (studies cơ bản). Widget miễn phí, không cần key; dữ liệu "
            "hiển thị là nguồn bên ngoài TradingView, không phải dữ liệu pipeline của app."
        )
        st.divider()
        st.markdown("#### Technical Analysis — gauge tín hiệu (TradingView)")
        render_tv_technical_analysis(ticker, tv_exchange, height=400)
        st.caption(
            "Đánh giá kỹ thuật tổng hợp (oscillators + đường trung bình) của TradingView cho khung "
            "ngày — chỉ tham khảo, không phải tín hiệu giao dịch của app."
        )
    else:
        render_plotly_price_chart(shown, rsi_shown, ticker)
        st.caption(
            "Nền tảng dữ liệu không ffill ngày nghỉ. Nến chỉ vẽ được vùng gần đây có đủ OHLC từ nguồn chính; "
            "lịch sử cũ hơn (nguồn HOSE chính thức) chỉ có close + volume → hiển thị đường close, không bịa nến. "
            "(app.tradingview.enabled=false → dùng biểu đồ plotly nội bộ.)"
        )

    # --- Kết quả dự báo: đọc từ DB (src.forecast) ---
    st.divider()
    st.markdown("### Kết quả dự báo")
    # Banner toàn cục ở đầu main flow đã gọi forecast_summary cho mã đang chọn;
    # ở đây TÁI DÙNG `summary`. Chỉ gọi lại trong đúng run mà mã vừa ĐỔI qua
    # selectbox (banner còn mang mã cũ) — bình thường toàn trang chỉ gọi 1 lần.
    if summary.get("ticker") != ticker:
        conn = open_db()
        try:
            summary = forecast_summary(conn, ticker)
        finally:
            conn.close()

    if not summary["has_run"]:
        st.info(summary["message"])
    else:
        run_info = summary["run"]
        st.caption(
            f"Run #{run_info['id']} · train lúc {run_info['trained_at']} · dữ liệu đến {run_info['data_until']}"
        )
        last_result = st.session_state.get("last_run_result")
        if isinstance(last_result, dict) and last_result.get("ticker") == ticker:
            st.caption(
                "Lần chạy gần nhất trong phiên này: settle {settled} dự báo cũ, "
                "re-point {repointed} dự báo lệch ngày lễ.".format(
                    settled=last_result.get("settled", 0), repointed=last_result.get("repointed", 0)
                )
            )

        # --- Bảng so sánh 4 model: giá / hướng / xác suất tăng ---
        st.markdown("#### Dự báo phiên kế tiếp — so sánh 4 model")
        prediction_rows = [
            {
                "Mô hình": MODEL_VN.get(str(row.get("model")), str(row.get("model"))),
                "Horizon (phiên)": int(row.get("horizon", 0)),
                "Phiên mục tiêu (ước lượng)": str(row.get("target_date", ""))[:10],
                "Giá dự báo (nghìn đ)": row.get("predicted_price"),
                "Hướng (theo giá dự báo)": fmt_direction(row.get("predicted_direction")),
                "Xác suất tăng (%)": (
                    float(row.get("probability_up")) * 100.0
                    if row.get("probability_up") is not None and pd.notna(row.get("probability_up"))
                    else None
                ),
                "Close tại t (nghìn đ)": row.get("actual_price") if "actual_price" in row else None,
            }
            for row in summary["predictions"]
        ]
        if prediction_rows:
            # Close tại t: lấy từ forecast (close_t) nếu có — predictions ghi qua DB
            # không mang close_t; bảng dùng predicted_price đối chiếu close hiện tại.
            forecast_df = pd.DataFrame(prediction_rows)
            conn = open_db()
            try:
                runs_now = db.list_runs(conn, ticker)
                close_t = None
                if runs_now and runs_now[0]["metrics_json"]:
                    import json as _json

                    metrics = _json.loads(runs_now[0]["metrics_json"])
                    forecasts = metrics.get("forecasts", [])
                    if forecasts:
                        close_t = forecasts[0].get("close_t")
            finally:
                conn.close()
            if close_t:
                forecast_df["Close tại t (nghìn đ)"] = close_t
            else:
                forecast_df = forecast_df.drop(columns=["Close tại t (nghìn đ)"])
            st.dataframe(
                forecast_df,
                hide_index=True,
                width="stretch",
                column_config={
                    "Giá dự báo (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
                    "Close tại t (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
                    "Xác suất tăng (%)": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                "Hướng = phân loại lợi nhuận ước tính của giá dự báo so với close hiện tại theo vùng neutral "
                f"±{cfg.get('neutral_band_pct', 0.3)}% (cùng quy ước 3 lớp với src/db). Xác suất tăng là đầu ra "
                "phía phân loại của model — hai phía có thể không nhất quán, hiển thị trung thực. "
                "Phiên mục tiêu là ƯỚC LƯỢNG ngày làm việc kế tiếp (lịch nghỉ lễ HOSE chưa biết trước)."
            )
        else:
            st.caption("Run có metrics nhưng chưa ghi dự báo phiên kế tiếp nào.")

        # --- Walk-forward: skill vs persistence & edge vs base rate (reports/) ---
        st.markdown("#### Walk-forward — so với baseline bắt buộc")
        metrics_csv = read_report_csv("metrics", ticker)
        if metrics_csv is None or metrics_csv.empty:
            st.warning("Chưa có reports/{ticker}/metrics.csv — bảng metrics sẽ xuất hiện sau lần huấn luyện đầu.")
        else:
            horizon_options = sorted({int(h) for h in metrics_csv["horizon"].tolist()})
            h_choice = st.radio(
                "Horizon (phiên)", horizon_options,
                format_func=lambda h: f"h = {h} phiên",
                horizontal=True,
            )
            h_frame = metrics_csv.loc[metrics_csv["horizon"] == h_choice].copy()
            metric_table = pd.DataFrame(
                {
                    "Mô hình": [
                        MODEL_VN.get(str(m), str(m)) for m in h_frame["model"].tolist()
                    ],
                    "Số fold": h_frame.get("n_folds"),
                    "MAE": h_frame.get("reg_mae"),
                    "MAE persistence": h_frame.get("reg_persistence_mae"),
                    # metrics.csv đã lưu skill/edge theo ĐIỂM % — giữ nguyên đơn vị,
                    # NumberColumn printf format KHÔNG tự nhân 100.
                    "Skill vs persistence (%)": h_frame.get("reg_skill_vs_persistence_pct"),
                    "Độ chính xác hướng (%)": h_frame.get("cls_accuracy") * 100.0
                    if "cls_accuracy" in h_frame
                    else None,
                    "Base rate — lớp đa số (%)": h_frame.get("cls_base_rate_majority") * 100.0
                    if "cls_base_rate_majority" in h_frame
                    else None,
                    "Edge vs base rate (điểm %)": h_frame.get("cls_edge_vs_base_rate") * 100.0
                    if "cls_edge_vs_base_rate" in h_frame
                    else None,
                    "ROC-AUC": h_frame.get("cls_roc_auc"),
                    "Lỗi": h_frame.get("error"),
                }
            )
            st.dataframe(
                metric_table,
                hide_index=True,
                width="stretch",
                column_config={
                    "MAE": st.column_config.NumberColumn(format="%.4f"),
                    "MAE persistence": st.column_config.NumberColumn(format="%.4f"),
                    "Skill vs persistence (%)": st.column_config.NumberColumn(
                        format="%.2f%%", help="Dương = model có MAE NHỎ hơn persistence trên cùng fold."
                    ),
                    "Độ chính xác hướng (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Base rate — lớp đa số (%)": st.column_config.NumberColumn(format="%.1f%%"),
                    "Edge vs base rate (điểm %)": st.column_config.NumberColumn(
                        format="%.2f%%", help="Dương = độ chính xác hướng cao hơn lớp đa số trên cùng fold."
                    ),
                    "ROC-AUC": st.column_config.NumberColumn(format="%.3f"),
                },
            )
            skill_values = [
                float(v)
                for v in h_frame.get("reg_skill_vs_persistence_pct", []).tolist()
                if pd.notna(v)
            ] if "reg_skill_vs_persistence_pct" in h_frame else []
            if skill_values and max(skill_values) <= 0:
                st.warning(
                    "Không model nào thắng persistence ở horizon này (skill đều ≤ 0) — "
                    "kết quả chỉ nên tham khảo, KHÔNG nên tin dự báo giá của model."
                )
            st.caption(
                "Số liệu là aggregate walk-forward (expanding window, retrain mỗi "
                f"{cfg.get('walk_forward', {}).get('retrain_every', 21)} phiên, embargo "
                f"{cfg.get('walk_forward', {}).get('embargo', 5)} phiên) — so với persistence (giá) và "
                "majority-class (hướng) trên CÙNG các fold, không phải in-sample."
            )

            # --- Biểu đồ backtest vs thực tế (PNG do pipeline xuất) ---
            st.markdown("#### Backtest walk-forward vs thực tế")
            png = read_backtest_png(ticker, h_choice)
            if png is None:
                st.caption(f"Chưa có biểu đồ backtest_h{h_choice}.png cho {ticker}.")
            else:
                st.image(png, caption=f"{ticker} — walk-forward h={h_choice} ( LightGBM chính vs persistence)")

            # --- SHAP: độ quan trọng feature của LightGBM (reports/) ---
            st.markdown("#### SHAP — feature quan trọng nhất (LightGBM)")
            shap_csv = read_report_csv(f"shap_h{h_choice}", ticker)
            if shap_csv is None or shap_csv.empty:
                st.caption(
                    f"Chưa có shap_importance_h{h_choice}.csv (thường do các fold chỉ có một lớp — "
                    "LightGBM classifier không fit được, được pipeline ghi rõ trong log)."
                )
            else:
                top = (
                    shap_csv[["feature", "mean_abs_shap"]]
                    .dropna()
                    .sort_values("mean_abs_shap", ascending=False)
                    .head(SHAP_TOP_N)
                    .iloc[::-1]
                )
                shap_fig = go.Figure(
                    go.Bar(
                        x=top["mean_abs_shap"].astype(float),
                        y=top["feature"].astype(str),
                        orientation="h",
                        marker_color="#5c6bc0",
                    )
                )
                shap_fig.update_layout(
                    height=max(280, 26 * len(top) + 90),
                    xaxis_title="|SHAP| trung bình theo fold",
                    yaxis_title=None,
                    margin={"l": 10, "r": 10, "t": 20, "b": 10},
                )
                st.plotly_chart(shap_fig, width="stretch")
                st.caption(f"Top {len(top)}/{len(shap_csv)} feature — trung bình |SHAP| qua các fold walk-forward.")

# ===========================================================================
# TAB 4 — LỊCH SỬ DỰ ĐOÁN
# ===========================================================================
else:
    st.subheader("Lịch sử dự đoán ↔ kết quả thực tế")
    st.caption(
        "Bảng predictions ⟕ outcomes trong SQLite (src.db) — mỗi dự báo được đối chiếu khi phiên mục tiêu "
        "đã có giá thực tế (so close tại t với close cách đúng `horizon` PHIÊN, vùng neutral ±"
        f"{cfg.get('neutral_band_pct', 0.3)}%)."
    )

    all_tickers = db_tickers()
    if not all_tickers:
        try:
            all_tickers = watchlist().list_tickers()
        except WatchlistError:
            all_tickers = []
    if not all_tickers:
        st.info("Chưa có dự báo nào trong DB — chạy “Huấn luyện & dự báo” ở tab Phân tích & dự báo trước.")
        st.stop()

    f_ticker, f_model, f_from, f_to = st.columns([1.2, 1.4, 1.2, 1.2])
    ticker_options = ["Tất cả mã", *sorted(set(all_tickers) | {selected_ticker()})]
    ticker_filter = f_ticker.selectbox(
        "Mã",
        ticker_options,
        index=ticker_options.index(selected_ticker()) if selected_ticker() in ticker_options else 0,
    )
    conn = open_db()
    try:
        model_options = ["Tất cả model", *sorted({str(r["model"]) for r in conn.execute("SELECT DISTINCT model FROM predictions")})]
    finally:
        conn.close()
    model_filter = f_model.selectbox("Mô hình", model_options)
    from_date = f_from.date_input("Từ ngày (phiên mục tiêu)", value=None)
    to_date = f_to.date_input("Đến ngày (phiên mục tiêu)", value=None)

    # --- Nút đối chiếu (settle) — không đụng pipeline, chỉ so giá trong cache/tải tăng dần ---
    if st.button("🔄 Đối chiếu dự báo chưa settle với giá thực tế", key="settle_btn"):
        with st.spinner("Đang tải giá và đối chiếu…"):
            conn = open_db()
            try:
                settled_count = db.settle_predictions(
                    conn,
                    neutral_band=float(cfg.get("neutral_band_pct", 0.3)) / 100.0,
                )
            except Exception as exc:  # noqa: BLE001 — nguồn giá lỗi phải hiển thị rõ
                st.error(f"Đối chiếu thất bại: {type(exc).__name__}: {exc}")
            else:
                st.toast(f"Đã đối chiếu {settled_count} dự báo", icon="✅")
                st.rerun()
            finally:
                conn.close()

    conn = open_db()
    try:
        rows = db.list_predictions(
            conn,
            ticker=None if ticker_filter == "Tất cả mã" else ticker_filter,
        )
        history = pd.DataFrame([dict(r) for r in rows])
    finally:
        conn.close()

    if history.empty:
        st.info("Chưa có dự báo nào khớp bộ lọc.")
        st.stop()

    history["target_date"] = history["target_date"].astype(str).str.slice(0, 10)
    # ``db.list_predictions`` không trả cột abs_error (chỉ actual_price/hit) —
    # tính từ cặp (predicted_price, actual_price) có mặt cả hai, cùng quy ước
    # với src/forecast.settlement_stats.
    history["_abs_error"] = history.apply(
        lambda r: abs(float(r["predicted_price"]) - float(r["actual_price"]))
        if pd.notna(r["predicted_price"]) and pd.notna(r["actual_price"])
        else None,
        axis=1,
    )
    if model_filter != "Tất cả model":
        history = history.loc[history["model"] == model_filter]
    if from_date is not None:
        history = history.loc[history["target_date"] >= from_date.isoformat()]
    if to_date is not None:
        history = history.loc[history["target_date"] <= to_date.isoformat()]
    history = history.sort_values(["target_date", "id"], ascending=[False, False])

    display = pd.DataFrame(
        {
            "Phiên mục tiêu": history["target_date"],
            "Mã": history["ticker"],
            "Mô hình": history["model"].map(lambda m: MODEL_VN.get(str(m), str(m))),
            "Horizon": history["horizon"].map(lambda h: f"{int(h)} phiên"),
            "Giá dự báo (nghìn đ)": history["predicted_price"],
            "Hướng dự báo": history["predicted_direction"].map(fmt_direction),
            "P(tăng)": history["probability_up"],
            "Giá thực tế (nghìn đ)": history["actual_price"],
            "Hướng thực tế": history["actual_direction"].map(fmt_direction),
            "P(tăng) (%)": pd.to_numeric(history["probability_up"], errors="coerce") * 100.0,
            "Đúng hướng?": history["hit"].map(
                lambda h: {1: "✓ Đúng", 0: "✗ Sai"}.get(int(h), "— chưa đối chiếu")
                if pd.notna(h) else "— chưa đối chiếu"
            ),
            "Sai số |Δ| (nghìn đ)": history["_abs_error"],
            "Ghi lúc": history["created_at"].astype(str).str.slice(0, 16),
            "Đối chiếu lúc": history["settled_at"].astype(str).str.slice(0, 16)
            if "settled_at" in history else None,
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Giá dự báo (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
            "Giá thực tế (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
            "P(tăng) (%)": st.column_config.NumberColumn(format="%.1f"),
            "Sai số |Δ| (nghìn đ)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(f"{len(display)} dự báo khớp bộ lọc · hit = hướng dự báo trùng hướng thực tế (3 lớp).")

    # --- Win rate & MAE tích lũy theo model (trên phần khớp bộ lọc) ---
    st.markdown("### Win rate & MAE theo model")
    settled = history.loc[history["actual_price"].notna()].copy()
    if settled.empty:
        st.info("Chưa có dự báo nào đã đối chiếu trong bộ lọc này.")
    else:
        agg_rows: list[dict[str, Any]] = []
        for model_key, group in settled.groupby("model", sort=True):
            with_direction = group.loc[group["hit"].notna()]
            hits = int((with_direction["hit"] == 1).sum())
            errors = group.loc[group["_abs_error"].notna(), "_abs_error"].astype(float)
            agg_rows.append(
                {
                    "Mô hình": MODEL_VN.get(str(model_key), str(model_key)),
                    "Số dự báo đã đối chiếu": int(len(group)),
                    "Đúng hướng": hits,
                    "Win rate (%)": (hits / len(with_direction)) * 100.0
                    if len(with_direction)
                    else None,
                    "MAE trung bình (nghìn đ)": float(errors.mean()) if len(errors) else None,
                    "MAE lớn nhất (nghìn đ)": float(errors.max()) if len(errors) else None,
                }
            )
        st.dataframe(
            pd.DataFrame(agg_rows),
            hide_index=True,
            width="stretch",
            column_config={
                "Win rate (%)": st.column_config.NumberColumn(
                    format="%.1f%%",
                    help="Tỉ lệ dự báo hướng ĐÚNG trên các dự báo đã đối chiếu CÓ hướng.",
                ),
                "MAE trung bình (nghìn đ)": st.column_config.NumberColumn(format="%.3f"),
                "MAE lớn nhất (nghìn đ)": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption(
            "Win rate đối chiếu với base rate lớp đa số (xem tab Phân tích & dự báo) — không so với 50% ngẫu nhiên, "
            "vì giá có drift. MAE tính trên các dòng có cả giá dự báo lẫn giá thực tế."
        )

        # MAE tích lũy (trung bình luỹ kế sai số tuyệt đối) theo model theo thời gian
        curve_source = settled.loc[settled["_abs_error"].notna()].sort_values("target_date")
        if not curve_source.empty:
            curve_fig = go.Figure()
            for model_key, group in curve_source.groupby("model", sort=True):
                cumulative = group["_abs_error"].astype(float).expanding().mean()
                curve_fig.add_trace(
                    go.Scatter(
                        x=pd.to_datetime(group["target_date"]),
                        y=cumulative,
                        mode="lines",
                        name=MODEL_VN.get(str(model_key), str(model_key)),
                        line={"width": 1.8},
                    )
                )
            curve_fig.update_layout(
                height=340,
                xaxis_title="Phiên mục tiêu",
                yaxis_title="MAE tích lũy (nghìn đồng)",
                legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
                margin={"l": 10, "r": 10, "t": 20, "b": 10},
            )
            st.plotly_chart(curve_fig, width="stretch")
            st.caption("MAE tích lũy = trung bình luỹ kế sai số tuyệt đối theo thời gian — đường dốc lên "
                       "nghĩa là model đang dự báo tệ dần ở giai đoạn gần.")
