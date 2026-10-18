"""Dự báo — đọc lớp dự báo MỚI NHẤT từ SQLite cho dashboard (tab "Kết quả").

Module này CHỈ ĐỌC dữ liệu mà ``src/pipeline.run_pipeline`` đã ghi (bảng runs
+ predictions + outcomes qua ``src/db``) — không tải model, không tính feature,
không gọi mạng. Mọi số "skill vs persistence" (MAE) và "edge vs base rate"
(hướng) hiển thị trên dashboard đến từ ``metrics_json`` của run mới nhất,
chính là kết quả walk-forward đã chạy trong pipeline.

API cho app/dashboard.py:
- ``latest_run(conn, ticker)``            → dòng runs mới nhất (hoặc None).
- ``latest_predictions(conn, ticker)``    → DataFrame dự báo của run mới nhất,
  kèm kết quả thực tế (outcome) nếu đã settle.
- ``forecast_summary(conn, ticker)``      → dict tổng hợp cho tab Kết quả:
  thông tin run, metrics per horizon/model, dự báo mới nhất từng model,
  thống kê đối chiếu (số đã settle, tỉ lệ hit).
- ``tomorrow_consensus(predictions)``     → hàm thuần: đồng thuận dự báo
  PHIÊN MAI (horizon=1) từ list predictions — KHÔNG đụng DB/mạng.
- ``settlement_stats(conn, ticker)``      → thống kê đối chiếu dự báo.

Ticker không có run nào → ``forecast_summary`` trả dict với ``has_run=False``
(không raise — dashboard hiển thị trạng thái "chưa huấn luyện"); lỗi SQLite
thì để propagate (lỗi hạ tầng phải thấy rõ).
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from typing import Any

import pandas as pd

from . import db

logger = logging.getLogger(__name__)

__all__ = [
    "forecast_summary",
    "latest_predictions",
    "latest_run",
    "settlement_stats",
    "tomorrow_consensus",
]


def latest_run(conn: sqlite3.Connection, ticker: str) -> sqlite3.Row | None:
    """Dòng ``runs`` mới nhất của ``ticker`` (theo trained_at, rồi id); None nếu chưa có."""
    runs = db.list_runs(conn, ticker)
    return runs[0] if runs else None


def latest_predictions(conn: sqlite3.Connection, ticker: str) -> pd.DataFrame:
    """Dự báo của run MỚI NHẤT cho ``ticker`` (kèm outcome nếu đã settle).

    Cột: các cột bảng predictions (model, target_date, horizon, predicted_price,
    predicted_direction, probability_up, created_at) + cột outcome
    (actual_price, actual_direction, hit) + trained_at/data_until của run.
    DataFrame rỗng (đúng schema) khi chưa có run nào.
    """
    run = latest_run(conn, ticker)
    empty = pd.DataFrame(
        columns=[
            "id", "run_id", "ticker", "model", "target_date", "horizon",
            "predicted_price", "predicted_direction", "probability_up", "created_at",
            "actual_price", "actual_direction", "hit", "trained_at", "data_until",
        ]
    )
    if run is None:
        return empty
    rows = db.list_predictions(conn, run_id=int(run["id"]))
    frame = pd.DataFrame([dict(row) for row in rows])
    if frame.empty:
        frame = empty.copy()
    frame["trained_at"] = run["trained_at"]
    frame["data_until"] = run["data_until"]
    return frame


def settlement_stats(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Thống kê đối chiếu dự báo của ``ticker`` với giá thực tế.

    - ``n_predictions``: tổng số dự báo đã ghi (mọi run);
    - ``n_settled`` / ``n_unsettled``: đã/chưa có kết quả thực tế;
    - ``n_hits`` / ``hit_rate``: số/tỉ lệ dự báo hướng ĐÚNG trong các dự báo đã
      settle CÓ hướng (predicted_direction không NULL — các dòng model không
      cho dự báo giá/hướng được loại khỏi hit_rate, không bịa 0);
    - ``mean_abs_error``: |dự báo giá − giá thực| trung bình của các dòng đã
      settle có predicted_price (None khi không có dòng nào).
    """
    rows = db.list_predictions(conn, ticker=ticker)
    n_predictions = len(rows)
    n_settled = sum(1 for row in rows if row["actual_price"] is not None)
    with_direction = [
        row for row in rows
        if row["actual_price"] is not None and row["predicted_direction"] is not None
        and row["hit"] is not None
    ]
    n_hits = sum(1 for row in with_direction if int(row["hit"]) == 1)
    # ``db.list_predictions`` không trả cột abs_error — tính từ cặp
    # (predicted_price, actual_price) có mặt cả hai.
    abs_errors = [
        abs(float(row["predicted_price"]) - float(row["actual_price"]))
        for row in rows
        if row["predicted_price"] is not None and row["actual_price"] is not None
    ]
    return {
        "ticker": str(ticker).upper(),
        "n_predictions": n_predictions,
        "n_settled": n_settled,
        "n_unsettled": n_predictions - n_settled,
        "n_hits": n_hits,
        "hit_rate": (n_hits / len(with_direction)) if with_direction else None,
        "mean_abs_error": (sum(abs_errors) / len(abs_errors)) if abs_errors else None,
    }


def _row_get(row: Any, key: str, default: Any = None) -> Any:
    """Đọc ``row[key]`` cho cả Mapping (dict) lẫn sqlite3.Row; thiếu → default."""
    getter = getattr(row, "get", None)
    if getter is not None:
        return getter(key, default)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        return default


def _is_missing(value: Any) -> bool:
    """None/NaN/inf coi như "không có số liệu" (predictions đi qua pandas nên
    ô trống có thể thành NaN thay vì None)."""
    if value is None:
        return True
    if isinstance(value, float) and not math.isfinite(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _finite_or_none(value: Any) -> float | None:
    """Số hữu hạn hoặc None — bỏ None/NaN/inf và giá trị không ép được sang float."""
    if _is_missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def tomorrow_consensus(predictions: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Đồng thuận dự báo PHIÊN MAI (horizon=1) — hàm thuần, không đọc DB/mạng.

    Đầu vào: list predictions như ``forecast_summary(...)["predictions"]`` —
    mỗi phần tử một dict (hoặc sqlite3.Row) có ``horizon``, ``target_date``,
    ``predicted_price``, ``predicted_direction``, ``probability_up``. Giá trị
    CÓ THỂ None/NaN khi model không cho dự báo giá/xác suất.

    Trả:
    - list rỗng hoặc không có dòng horizon=1 → ``{"has_forecast": False}``;
    - ngược lại: ``has_forecast=True``, ``target_date`` (phiên mục tiêu của các
      dòng horizon=1, None nếu thiếu), ``n_models`` (số dòng horizon=1 — mỗi
      dòng là một model), ``price_mean/price_min/price_max`` (bỏ None/NaN;
      None khi không model nào có giá), ``p_up_mean`` (bỏ None/NaN; None khi
      không có), ``n_up/n_down/n_neutral`` (số model báo từng hướng — hướng
      None không được đếm).
    """
    rows = [
        row for row in (predictions or [])
        if (horizon := _finite_or_none(_row_get(row, "horizon"))) is not None and int(horizon) == 1
    ]
    if not rows:
        return {"has_forecast": False}

    target_date: str | None = None
    for row in rows:
        raw = _row_get(row, "target_date")
        if not _is_missing(raw) and str(raw).strip():
            target_date = str(raw).strip()[:10]
            break

    prices = [
        price for price in (_finite_or_none(_row_get(row, "predicted_price")) for row in rows)
        if price is not None
    ]
    probabilities = [
        prob for prob in (_finite_or_none(_row_get(row, "probability_up")) for row in rows)
        if prob is not None
    ]
    directions = [str(_row_get(row, "predicted_direction")).strip().lower() for row in rows]
    return {
        "has_forecast": True,
        "target_date": target_date,
        "n_models": len(rows),
        "price_mean": (sum(prices) / len(prices)) if prices else None,
        "price_min": min(prices) if prices else None,
        "price_max": max(prices) if prices else None,
        "p_up_mean": (sum(probabilities) / len(probabilities)) if probabilities else None,
        "n_up": directions.count("up"),
        "n_down": directions.count("down"),
        "n_neutral": directions.count("neutral"),
    }


def forecast_summary(conn: sqlite3.Connection, ticker: str) -> dict[str, Any]:
    """Tổng hợp cho tab "Kết quả" của dashboard — đọc-only, không đụng model.

    Trả dict:
    - ``has_run``: False + ``message`` khi ticker chưa từng được huấn luyện;
    - ``run``: {id, trained_at, data_until, model_dir};
    - ``metrics``: metrics_json đã parse (per_horizon → model → aggregate:
      skill vs persistence, edge vs base rate, ...);
    - ``predictions``: list[dict] dự báo mới nhất từng model/horizon (+ outcome
      nếu đã settle) — đã sort theo (horizon, model);
    - ``settlement``: dict từ ``settlement_stats``.
    """
    ticker = str(ticker).strip().upper()
    run = latest_run(conn, ticker)
    if run is None:
        return {
            "ticker": ticker,
            "has_run": False,
            "message": (
                f"Mã {ticker} chưa được huấn luyện — mở tab \"Huấn luyện & dự báo\" "
                "và chạy pipeline cho mã này."
            ),
            "run": None,
            "metrics": None,
            "predictions": [],
            "settlement": settlement_stats(conn, ticker),
        }
    metrics_json = run["metrics_json"]
    try:
        metrics = json.loads(metrics_json) if metrics_json else None
    except json.JSONDecodeError as exc:
        logger.error("metrics_json của run #%d hỏng (%s) — hiển thị metrics rỗng", run["id"], exc)
        metrics = None

    predictions_frame = latest_predictions(conn, ticker)
    predictions = sorted(
        predictions_frame.to_dict(orient="records"),
        key=lambda row: (int(row.get("horizon", 0)), str(row.get("model", ""))),
    )
    return {
        "ticker": ticker,
        "has_run": True,
        "run": {
            "id": int(run["id"]),
            "trained_at": run["trained_at"],
            "data_until": run["data_until"],
            "model_dir": run["model_dir"],
        },
        "metrics": metrics,
        "predictions": predictions,
        "settlement": settlement_stats(conn, ticker),
    }
