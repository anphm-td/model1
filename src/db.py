"""Lưu trữ — SQLite data/app.db cho lịch sử chạy, dự báo và kết quả thực tế.

Schema (khớp yêu cầu dự án):
- runs(id, ticker, trained_at, data_until, metrics_json, model_dir)
      một lần "Huấn luyện & dự báo": metrics_json là JSON (skill vs persistence,
      edge vs base rate, ...).
- predictions(id, run_id→runs.id, ticker, model, target_date, horizon,
      predicted_price, predicted_direction, probability_up, created_at)
      predicted_direction ∈ {'up','down','neutral'} — cùng quy ước 3 lớp với
      src/labeling.py (vùng neutral ±neutral_band).
- outcomes(prediction_id→predictions.id, actual_price, actual_direction,
      abs_error, hit, settled_at)
      hit = 1 nếu predicted_direction == actual_direction; NULL nếu không có
      hướng dự báo. settled_at = thời điểm đối chiếu.

settle_predictions(): đối chiếu các prediction CHƯA settle mà target_date đã có
giá thực tế — idempotent, chỉ settle khi target_date có đúng trong chuỗi giá.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Đường dẫn DB mặc định (khớp config.yaml: database.path).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_DB_PATH = DATA_DIR / "app.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    trained_at TEXT NOT NULL,
    data_until TEXT,
    metrics_json TEXT,
    model_dir TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES runs(id),
    ticker TEXT NOT NULL,
    model TEXT NOT NULL,
    target_date TEXT NOT NULL,
    horizon INTEGER NOT NULL,
    predicted_price REAL,
    predicted_direction TEXT
        CHECK (predicted_direction IN ('up', 'down', 'neutral') OR predicted_direction IS NULL),
    probability_up REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcomes (
    prediction_id INTEGER PRIMARY KEY REFERENCES predictions(id),
    actual_price REAL,
    actual_direction TEXT
        CHECK (actual_direction IN ('up', 'down', 'neutral') OR actual_direction IS NULL),
    abs_error REAL,
    hit INTEGER CHECK (hit IN (0, 1) OR hit IS NULL),
    settled_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_ticker ON runs(ticker, trained_at);
CREATE INDEX IF NOT EXISTS idx_predictions_ticker ON predictions(ticker, target_date);
"""


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def connect(db_path: Path | str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Mở kết nối SQLite (row_factory=Row, bật foreign keys)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Tạo schema nếu chưa có — chạy lại an toàn (idempotent)."""
    conn.executescript(SCHEMA)
    conn.commit()


def classify_direction(return_value: float, neutral_band: float) -> str:
    """Phân loại hướng theo vùng neutral ±neutral_band (quy ước chung của dự án)."""
    if return_value > neutral_band:
        return "up"
    if return_value < -neutral_band:
        return "down"
    return "neutral"


# ---------------------------------------------------------------------------
# Ghi / tra cứu
# ---------------------------------------------------------------------------


def record_run(
    conn: sqlite3.Connection,
    ticker: str,
    *,
    data_until: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    model_dir: str | None = None,
    trained_at: str | None = None,
) -> int:
    """Ghi một lần huấn luyện; ``metrics`` được lưu thành metrics_json (JSON UTF-8)."""
    cursor = conn.execute(
        "INSERT INTO runs (ticker, trained_at, data_until, metrics_json, model_dir) VALUES (?, ?, ?, ?, ?)",
        (
            str(ticker).upper(),
            trained_at or _now_iso(),
            data_until,
            json.dumps(metrics, ensure_ascii=False, sort_keys=True) if metrics is not None else None,
            model_dir,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def record_prediction(
    conn: sqlite3.Connection,
    run_id: int,
    ticker: str,
    model: str,
    target_date: str,
    horizon: int,
    *,
    predicted_price: float | None = None,
    predicted_direction: str | None = None,
    probability_up: float | None = None,
    created_at: str | None = None,
) -> int:
    """Ghi một dự báo cho ``target_date`` với ``horizon`` phiên (1 hoặc 5)."""
    cursor = conn.execute(
        (
            "INSERT INTO predictions (run_id, ticker, model, target_date, horizon, "
            "predicted_price, predicted_direction, probability_up, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        ),
        (
            int(run_id),
            str(ticker).upper(),
            model,
            str(target_date),
            int(horizon),
            predicted_price,
            predicted_direction,
            probability_up,
            created_at or _now_iso(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def list_runs(conn: sqlite3.Connection, ticker: str | None = None) -> list[sqlite3.Row]:
    """Lịch sử chạy (mới trước), lọc theo ticker nếu có."""
    if ticker:
        rows = conn.execute(
            "SELECT * FROM runs WHERE ticker = ? ORDER BY trained_at DESC, id DESC",
            (ticker.upper(),),
        )
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY trained_at DESC, id DESC")
    return list(rows)


def list_predictions(
    conn: sqlite3.Connection,
    run_id: int | None = None,
    ticker: str | None = None,
    settled: bool | None = None,
) -> list[sqlite3.Row]:
    """Tra prediction; ``settled`` True/False lọc đã/chưa đối chiếu, None = tất cả."""
    sql = (
        "SELECT p.*, o.actual_price AS actual_price, o.actual_direction AS actual_direction, "
        "o.hit AS hit FROM predictions p LEFT JOIN outcomes o ON o.prediction_id = p.id WHERE 1=1"
    )
    params: list[Any] = []
    if run_id is not None:
        sql += " AND p.run_id = ?"
        params.append(int(run_id))
    if ticker:
        sql += " AND p.ticker = ?"
        params.append(ticker.upper())
    if settled is True:
        sql += " AND o.prediction_id IS NOT NULL"
    elif settled is False:
        sql += " AND o.prediction_id IS NULL"
    sql += " ORDER BY p.target_date, p.id"
    return list(conn.execute(sql, params))


def unsettled_predictions(conn: sqlite3.Connection, ticker: str | None = None) -> list[sqlite3.Row]:
    """Các prediction chưa có dòng outcome nào."""
    sql = (
        "SELECT p.* FROM predictions p LEFT JOIN outcomes o ON o.prediction_id = p.id "
        "WHERE o.prediction_id IS NULL"
    )
    params: tuple[Any, ...] = ()
    if ticker:
        sql += " AND p.ticker = ?"
        params = (ticker.upper(),)
    sql += " ORDER BY p.target_date, p.id"
    return list(conn.execute(sql, params))


# ---------------------------------------------------------------------------
# Đối chiếu dự báo với giá thực tế
# ---------------------------------------------------------------------------


def settle_predictions(
    conn: sqlite3.Connection,
    price_provider: Callable[[str], pd.DataFrame] | None = None,
    neutral_band: float = 0.003,
) -> int:
    """Đối chiếu prediction chưa settle mà ``target_date`` đã có giá thực tế.

    Tham số:
    - ``price_provider(ticker)`` → DataFrame có cột ``close``, index là ngày phiên
      (đúng định dạng data_loader.load_history trả về). Mặc định dùng
      data_loader.load_history (tải tăng dần, có cache).
    - ``neutral_band``: vùng neutral ±band cho lợi nhuận thực tế — phải khớp
      quy ước khi ghi predicted_direction (config.yaml: neutral_band_pct 0.3%).

    Quy ước hướng thực tế: so close[target_date] với close cách đúng ``horizon``
    PHIÊN giao dịch trước đó (không phải ngày lịch), phân loại theo ±neutral_band
    ('up'/'down'/'neutral'). abs_error = |predicted_price − actual_price| khi có
    predicted_price. Chỉ settle khi target_date có ĐÚNG trong chuỗi giá; prediction
    còn lại giữ nguyên. Idempotent (chạy lại settle 0 dòng mới).

    Trả về số prediction đã settle trong lần gọi này.
    """
    if price_provider is None:
        from .data_loader import load_history

        price_provider = load_history

    rows = unsettled_predictions(conn)
    if not rows:
        return 0

    frames: dict[str, pd.DataFrame] = {}
    settled_at = _now_iso()
    settled_count = 0
    for row in rows:
        ticker = row["ticker"]
        if ticker not in frames:
            try:
                frame = price_provider(ticker)
            except Exception as exc:  # noqa: BLE001 — nguồn giá lỗi không được giết settle
                logger.warning("settle: không lấy được giá cho %s: %s", ticker, exc)
                frame = None
            if frame is None or len(frame) == 0:
                frame = pd.DataFrame()
            frames[ticker] = frame
        frame = frames[ticker]
        if frame.empty:
            continue

        target = pd.Timestamp(row["target_date"])
        if target not in frame.index:
            continue  # chưa có phiên giá đúng target_date (tương lai / ngày nghỉ)
        position = int(frame.index.get_loc(target))
        horizon = int(row["horizon"])
        if position < horizon:
            logger.warning(
                "settle: %s target %s không đủ %d phiên nền — bỏ qua",
                ticker, row["target_date"], horizon,
            )
            continue

        close = frame["close"].astype(float)
        actual_price = float(close.iloc[position])
        base_price = float(close.iloc[position - horizon])
        if pd.isna(actual_price) or pd.isna(base_price) or base_price <= 0:
            continue
        actual_direction = classify_direction(actual_price / base_price - 1.0, neutral_band)

        predicted_price = row["predicted_price"]
        abs_error = None if predicted_price is None else abs(float(predicted_price) - actual_price)
        predicted_direction = row["predicted_direction"]
        hit = None if predicted_direction is None else int(predicted_direction == actual_direction)

        cursor = conn.execute(
            (
                "INSERT OR IGNORE INTO outcomes "
                "(prediction_id, actual_price, actual_direction, abs_error, hit, settled_at) "
                "VALUES (?, ?, ?, ?, ?, ?)"
            ),
            (row["id"], actual_price, actual_direction, abs_error, hit, settled_at),
        )
        if cursor.rowcount:
            settled_count += 1
    conn.commit()
    return settled_count


__all__ = [
    "DEFAULT_DB_PATH",
    "SCHEMA",
    "classify_direction",
    "connect",
    "init_db",
    "list_predictions",
    "list_runs",
    "record_prediction",
    "record_run",
    "settle_predictions",
    "unsettled_predictions",
]
