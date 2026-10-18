#!/usr/bin/env python
"""Job chạy HÀNG ĐÊM — entry point tự động hoá cho app dự báo cổ phiếu sàn HOSE.

Với MỖI mã trong danh sách (data/watchlist.json — đọc qua ``src.watchlist``;
watchlist không tồn tại/trống → ``default_ticker`` trong config.yaml), gọi
``src.pipeline.run_pipeline(ticker, cfg)`` làm TRỌN quy trình: settle dự báo
chờ → tải giá tăng dần → walk-forward → full retrain → dự báo h=1 và h=5
phiên kế tiếp → ghi DB theo ``database.path`` trong config.yaml (mặc định
data/app.db). Lỗi từng mã (PipelineError, InsufficientHistoryError, lỗi bất
ngờ) KHÔNG chặn các mã còn lại — ghi log rồi chạy tiếp.

Chế độ:
- Mặc định (full)  : huấn luyện + dự báo cho từng mã (như trên).
- ``--settle-only``: chỉ ``db.settle_predictions`` với price_provider
  ``data_loader.load_history`` rồi in kết quả — KHÔNG huấn luyện (kiểm nhanh).
- ``--config``     : dùng file config khác (mặc định config.yaml gốc dự án).

Nguồn giá lỗi khi còn cache: ``load_history`` trả cache CŨ kèm WARNING
(không chết) — job vẫn huấn luyện trên dữ liệu cũ. Để monitoring phân biệt,
mỗi mã in thêm ``DATA_UNTIL=<phiên dữ liệu>`` và ``DATA_WARNINGS=<số cảnh báo
nguồn giá>``, cảnh báo gốc được ghi lại đầy đủ vào log.

Danh sách mã: data/watchlist.json (theo config: app.watchlist_path) nếu có mã;
ngược lại ``default_ticker`` của config.yaml — KHÔNG hardcode mã nào. File
watchlist HỎNG (kể cả JSON hợp lệ nhưng entry sai dạng) hoặc config.yaml sai
cú pháp YAML → exit 2 với thông điệp rõ (job đêm phải fail to, không âm thầm
chạy nhầm danh sách).

Log: ra stdout VÀ file data/logs/daily_YYYY-MM-DD.log (append, UTF-8, tự tạo
thư mục nếu thiếu).
SQLite: kết nối của job đặt busy_timeout 30s + journal WAL — dashboard/
--settle-only mở cùng lúc không gây "database is locked".

Dòng máy đọc được (stdout, dùng cho monitoring/cron):
    DURATION_SECONDS=<tổng giây>   # luôn in cuối cùng (kể cả khi lỗi cấu hình)
    SETTLED=<số dòng vừa settle>   # mỗi mã đã chạy (full) / một dòng (settle-only)
    NEW_RUN=<run_id>               # mỗi mã đã chạy (chỉ chế độ full)
    DATA_UNTIL=<phiên dữ liệu>     # mỗi mã đã chạy (full)
    DATA_WARNINGS=<n>              # số cảnh báo nguồn giá trong lúc chạy mã đó

Tổng kết tiếng Việt cho người đọc: data_until, các dự báo mới theo
model/horizon (phiên mục tiêu, giá, hướng, xác suất) và đồng thuận phiên mai
qua ``src.forecast.tomorrow_consensus``.

Mã thoát: 0 mọi mã OK; 1 có mã lỗi (hoặc runner gặp lỗi bất ngờ); 2 lỗi cấu
hình/môi trường (config.yaml sai — kể cả neutral_band_pct sai kiểu/≤0,
watchlist hỏng, thiếu default_ticker, thiếu thư viện import, không tạo được
log, DB không mở được).

Cách chạy (từ BẤT KỲ cwd nào — script tự bootstrap sys.path theo __file__):
    C:\\ProgramData\\anaconda3\\python.exe scripts\\daily_run.py                # full
    C:\\ProgramData\\anaconda3\\python.exe scripts\\daily_run.py --settle-only  # kiểm nhanh
"""
from __future__ import annotations

import argparse
import logging
import math
import sqlite3
import sys
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

# Cho phép chạy trực tiếp ``python scripts/daily_run.py`` từ bất kỳ cwd nào —
# gốc dự án tính từ __file__ (scripts/daily_run.py → …/model1).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

#: Chờ khoá SQLite tối đa (ms) trước khi báo "database is locked".
SQLITE_BUSY_TIMEOUT_MS: int = 30_000


def _configure_streams() -> None:
    """Windows console: buộc UTF-8 cho stdout/stderr, ký tự lỗi thay bằng '?'."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


# Import src SAU bootstrap; thiếu thư viện (torch/lightgbm/... của interpreter
# sai) → thoát sạch exit 2 với hướng dẫn, KHÔNG traceback chết giữa job.
try:
    import yaml  # noqa: E402  (dependency của src.pipeline — cần YAMLError ở main)

    from src import db, forecast, pipeline  # noqa: E402
    from src.data_loader import load_history  # noqa: E402
    from src.watchlist import Watchlist, WatchlistError  # noqa: E402
except ImportError as exc:
    _configure_streams()
    print(f"[daily_run] LỖI MÔI TRƯỜNG: thiếu thư viện khi import src: {exc}", file=sys.stderr)
    print("[daily_run] Job đêm phải chạy bằng interpreter ĐỦ thư viện của dự án, ví dụ:", file=sys.stderr)
    print(
        r"[daily_run]   C:\ProgramData\anaconda3\python.exe "
        r"E:\PROJECT\model1\scripts\daily_run.py",
        file=sys.stderr,
    )
    sys.exit(2)

logger = logging.getLogger("daily_run")

#: Logger ghi CHỈ vào file log (propagate=False) — _emit() dùng để in ra stdout
#: bằng print() và ghi song song vào file đúng MỘT lần mỗi nơi.
_file_only = logging.getLogger("daily_run.file_only")

#: Cắm đã cấu hình log chưa (main() gọi lại không nhân đôi handler).
_logging_ready = False


# ---------------------------------------------------------------------------
# Log / emit
# ---------------------------------------------------------------------------


def _setup_logging(log_path: Path) -> None:
    """Log ra stdout VÀ file (append, UTF-8) — mọi logger của src đi qua root."""
    global _logging_ready
    if _logging_ready:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(stdout_handler)
    root.addHandler(file_handler)
    _file_only.setLevel(logging.INFO)
    _file_only.propagate = False
    _file_only.addHandler(file_handler)
    _logging_ready = True


def _emit(line: str) -> None:
    """In MỘT dòng ra stdout (flush ngay) và ghi vào file log đúng một lần."""
    print(line, flush=True)
    _file_only.info(line)


# ---------------------------------------------------------------------------
# Đường dẫn / cấu hình
# ---------------------------------------------------------------------------


def _resolve_project_path(raw: Any, default: str) -> Path:
    """Đường dẫn tương đối trong config → tuyệt đối theo gốc dự án (quy ước
    app/dashboard.py); rỗng/None → ``default``."""
    text = str(raw).strip() if raw is not None and str(raw).strip() else default
    path = Path(text)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _watchlist(cfg: Mapping[str, Any]) -> Watchlist:
    """Watchlist theo config.yaml (app.watchlist_path, mặc định data/watchlist.json)."""
    app_cfg = cfg.get("app") or {}
    return Watchlist(_resolve_project_path(app_cfg.get("watchlist_path"), "data/watchlist.json"))


def _db_path(cfg: Mapping[str, Any]) -> Path:
    """Đường dẫn SQLite theo config.yaml (database.path, mặc định data/app.db)."""
    db_cfg = cfg.get("database") or {}
    return _resolve_project_path(db_cfg.get("path"), "data/app.db")


def _neutral_band_fraction(cfg: Mapping[str, Any]) -> float:
    """Vùng neutral ±neutral_band_pct (%) → tỉ lệ (0.3 → 0.003) — KIỂM TRA SÁT.

    Ủy thác ``pipeline._neutral_band_fraction`` (cùng quy tắc với full mode:
    phải > 0) và bọc thêm lỗi KIỂU (``null`` → TypeError, ``"abc"`` → ValueError)
    thành PipelineError. Sai band là lỗi cấu hình NGHIÊM TRỌNG với
    ``--settle-only``: ``db.settle_predictions`` sẽ phân loại hướng thực tế sai
    (``db.classify_direction``) và outcomes ghi MỘT LẦN (INSERT OR IGNORE,
    idempotent) — không bao giờ được đối chiếu lại. Gọi ở pha cấu hình của
    main() cho CẢ hai chế độ → sai là exit 2 sạch trước khi đụng DB/huấn luyện.
    """
    try:
        return pipeline._neutral_band_fraction(cfg)
    except (TypeError, ValueError) as exc:
        raise pipeline.PipelineError(
            f"neutral_band_pct phải là số > 0 (%), nhận {cfg.get('neutral_band_pct')!r}"
        ) from exc


def _connect_tuned(db_path: Path) -> sqlite3.Connection:
    """Kết nối SQLite của job: busy_timeout dài + WAL.

    ``db.connect`` mặc định không cấu hình gì (timeout 5s, journal rollback) —
    dashboard hoặc --settle-only mở cùng lúc dễ gây "database is locked". WAL
    cho phép reader song song với writer của job; busy_timeout 30s chịu giao
    dịch ngắn. WAL là thuộc tính persistent của file DB — đặt một lần là đủ.
    """
    conn = db.connect(db_path)
    try:
        conn.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.Error as exc:  # ví dụ DB nằm trên FS không hỗ trợ WAL
        logger.warning("Không đặt được PRAGMA WAL/busy_timeout (chạy tiếp mặc định): %s", exc)
    return conn


def _resolve_tickers(cfg: Mapping[str, Any]) -> list[str]:
    """Danh sách mã chạy hôm nay: watchlist (nếu có mã) → default_ticker config.

    KHÔNG hardcode mã nào: watchlist trống/không có file → bắt buộc config phải
    có ``default_ticker``, thiếu → PipelineError (exit 2). File watchlist hỏng
    — kể cả JSON hợp lệ nhưng entry sai dạng (thiếu 'ticker', phần tử không
    phải dict) — được quy về PipelineError (exit 2), không chết bằng traceback.
    """
    try:
        tickers = _watchlist(cfg).list_tickers()
    except (WatchlistError, KeyError, TypeError, ValueError) as exc:
        raise pipeline.PipelineError(
            f"Watchlist không đọc được (file hỏng hoặc entry sai dạng): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    deduped = list(dict.fromkeys(tickers))  # file tay có thể chứa mã trùng
    if len(deduped) != len(tickers):
        logger.warning("Watchlist có mã trùng — đã bỏ trùng, giữ thứ tự: %s", deduped)
    if deduped:
        return deduped
    default = str(cfg.get("default_ticker", "")).strip().upper()
    if not default:
        raise pipeline.PipelineError(
            "Không xác định được danh sách mã: watchlist trống/không có và "
            "config.yaml thiếu default_ticker."
        )
    return [default]


# ---------------------------------------------------------------------------
# Gom cảnh báo nguồn giá (cache cũ do nguồn chết — job vẫn chạy, phải báo rõ)
# ---------------------------------------------------------------------------


class _SourceWarningCollector(logging.Handler):
    """Gom cảnh báo của logger nguồn dữ liệu trong lúc chạy MỘT mã.

    Nguồn giá lỗi khi còn cache → ``load_history`` vẫn trả cache CŨ kèm WARNING
    (data_loader.py: "giữ nguyên cache hiện có") nên job không chết nhưng đang
    train trên dữ liệu cũ. Bộ đếm này đưa tín hiệu đó ra dòng máy đọc
    ``DATA_WARNINGS=<n>`` để monitoring phân biệt chạy thành công với nguồn chết.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _collect_source_warnings() -> tuple[_SourceWarningCollector, logging.Handler]:
    """(collector, logger nguồn dữ liệu) — logger lấy từ module của load_history,
    không hardcode tên logger."""
    collector = _SourceWarningCollector()
    return collector, logging.getLogger(load_history.__module__)


def _report_source_warnings(ticker: str, messages: list[str], limit: int = 5) -> None:
    """Đưa cảnh báo nguồn giá ra dòng máy đọc + log chi tiết cho người đọc."""
    _emit(f"DATA_WARNINGS={len(messages)}")
    for message in messages[:limit]:
        logger.warning("[NGUỒN GIÁ]%s %s", f" [{ticker}]" if ticker else "", message)
    if len(messages) > limit:
        logger.warning(
            "[NGUỒN GIÁ] %s: còn %d cảnh báo khác — xem log nguồn data_loader.",
            ticker, len(messages) - limit,
        )


# ---------------------------------------------------------------------------
# Tổng kết tiếng Việt cho người đọc
# ---------------------------------------------------------------------------


def _fmt_price(value: Any) -> str:
    """Giá dự báo → 3 số lẻ; None/NaN/không ép được số → 'n/a'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.3f}" if math.isfinite(number) else "n/a"


def _fmt_prob(value: Any) -> str:
    """Xác suất P(LỚN) → 2 số lẻ; None/NaN → 'n/a'."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return f"{number:.2f}" if math.isfinite(number) else "n/a"


def _fmt_text(value: Any) -> str:
    """Hướng/phiên mục tiêu → chuỗi; None/rỗng → 'n/a'."""
    return str(value) if value not in (None, "") else "n/a"


def _print_ticker_summary(result: pipeline.PipelineResult) -> None:
    """Tổng kết một mã: data_until, dự báo mới theo model/horizon, đồng thuận phiên mai."""
    _emit(f"--- TỔNG KẾT {result.ticker} (run #{result.run_id}) ---")
    _emit(f"Dữ liệu đến phiên: {result.data_until} ({result.n_sessions} phiên)")
    if not result.forecasts:
        _emit("Không có dự báo mới nào (mọi model lỗi hoặc không đủ dữ liệu — xem log).")
    else:
        _emit("Dự báo mới (phiên mục tiêu | giá dự báo | hướng | P(LỚN)):")
        for row in result.forecasts:
            _emit(
                f"    h={row['horizon']} {row['model']:<9}: "
                f"{_fmt_text(row['target_date'])} | {_fmt_price(row['predicted_price'])} | "
                f"{_fmt_text(row['predicted_direction'])} | {_fmt_prob(row['probability_up'])}"
            )
    consensus = forecast.tomorrow_consensus(list(result.forecasts))
    if not consensus.get("has_forecast"):
        _emit("Đồng thuận phiên mai: không tính được (không có dự báo h=1).")
        return
    _emit(
        f"Đồng thuận phiên mai ({consensus['n_models']} model, h=1): "
        f"giá TB {_fmt_price(consensus['price_mean'])} "
        f"({_fmt_price(consensus['price_min'])}–{_fmt_price(consensus['price_max'])}) | "
        f"P(LỚN) TB {_fmt_prob(consensus['p_up_mean'])} | "
        f"up {consensus['n_up']} / down {consensus['n_down']} / "
        f"neutral {consensus['n_neutral']} | phiên {_fmt_text(consensus.get('target_date'))}"
    )


# ---------------------------------------------------------------------------
# Hai chế độ chạy
# ---------------------------------------------------------------------------


def _run_full(cfg: Mapping[str, Any], tickers: list[str]) -> int:
    """Chế độ mặc định: run_pipeline TRỌN quy trình cho từng mã, lỗi thì bỏ qua.

    Mỗi mã đều được gọi ``run_pipeline`` (không bỏ qua mã nào); MỘT kết nối
    SQLite theo ``database.path`` dùng chung (busy_timeout + WAL) và truyền
    thẳng vào run_pipeline — dashboard/--settle-only mở cùng lúc không gây
    "database is locked".
    Trả mã thoát: 0 mọi mã OK; 1 có mã lỗi; 2 DB không mở được.
    """
    conn: sqlite3.Connection | None = None
    failures: list[str] = []
    try:
        try:
            conn = _connect_tuned(_db_path(cfg))
        except sqlite3.Error as exc:
            logger.error("Không mở được DB %s: %s", _db_path(cfg), exc)
            return 2
        for ticker in tickers:
            logger.info("=== Chạy hàng đêm: %s ===", ticker)
            collector, source_logger = _collect_source_warnings()
            source_logger.addHandler(collector)
            try:
                result = pipeline.run_pipeline(ticker, cfg, conn=conn)
            except Exception as exc:  # noqa: BLE001 — PipelineError/InsufficientHistoryError/
                # lỗi bất ngờ: job đêm phải chạy TIẾP các mã còn lại, không chết vì một mã.
                logger.error("[FAIL] %s: %s: %s", ticker, type(exc).__name__, exc)
                failures.append(ticker)
                try:
                    conn.rollback()  # dọn transaction dở nếu run_pipeline chết giữa chừng
                except sqlite3.Error as rollback_exc:
                    logger.warning("Rollback %s lỗi: %s", ticker, rollback_exc)
                continue
            finally:
                source_logger.removeHandler(collector)
            logger.info("[OK] %s: run #%d, settle %d dòng", ticker, result.run_id, result.settled)
            _emit(f"SETTLED={result.settled}")
            _emit(f"NEW_RUN={result.run_id}")
            _emit(f"DATA_UNTIL={result.data_until}")
            _report_source_warnings(ticker, collector.messages)
            _print_ticker_summary(result)
        if failures:
            logger.error(
                "Kết thúc với LỖI ở %d/%d mã: %s",
                len(failures), len(tickers), ", ".join(failures),
            )
            return 1
        logger.info("Hoàn thành: %d mã OK.", len(tickers))
        return 0
    finally:
        if conn is not None:
            conn.close()


def _print_pending(conn: sqlite3.Connection) -> None:
    """Kết quả kiểm nhanh: các dự báo CHƯA settle còn lại, nhóm theo mã."""
    rows = db.unsettled_predictions(conn)
    if not rows:
        _emit("Chưa settle còn lại: 0 — mọi dự báo đã đối chiếu xong.")
        return
    pending: dict[str, list[str]] = {}
    for row in rows:
        pending.setdefault(str(row["ticker"]), []).append(
            f"{str(row['target_date'])[:10]} (h={row['horizon']})"
        )
    _emit(f"Chưa settle còn lại: {len(rows)} dòng, theo mã:")
    for ticker in sorted(pending):
        examples = ", ".join(pending[ticker][:3])
        more = "…" if len(pending[ticker]) > 3 else ""
        _emit(f"    {ticker}: {len(pending[ticker])} chờ — {examples}{more}")


def _run_settle_only(cfg: Mapping[str, Any], neutral_band: float) -> int:
    """Chế độ --settle-only: chỉ đối chiếu dự báo chờ với giá, KHÔNG huấn luyện.

    price_provider là ``data_loader.load_history`` (cache parquet, tải tăng dần)
    — đúng tham số mặc định mà run_pipeline dùng khi settle. ``neutral_band``
    đã được kiểm ở pha cấu hình (main — cùng quy tắc > 0 với full mode): outcomes
    ghi MỘT LẦN (INSERT OR IGNORE) nên tuyệt đối không settle với band sai rồi
    để dòng đó nằm sai hướng vĩnh viễn. Nguồn giá chết khi còn cache →
    load_history trả cache cũ kèm WARNING, được đếm vào DATA_WARNINGS để
    monitoring phân biệt "chưa có phiên mới" với "nguồn chết".
    Trả mã thoát: 0 OK; 2 lỗi hạ tầng (DB không mở được / SQLite lỗi).
    """
    db_path = _db_path(cfg)
    logger.info(
        "[SETTLE-ONLY] Kiểm nhanh: chỉ db.settle_predictions (không huấn luyện) — DB: %s",
        db_path,
    )
    try:
        conn = _connect_tuned(db_path)
    except sqlite3.Error as exc:
        logger.error("Không mở được DB %s: %s", db_path, exc)
        return 2
    try:
        db.init_db(conn)  # idempotent — DB/file trống vẫn chạy được
        collector, source_logger = _collect_source_warnings()
        source_logger.addHandler(collector)
        try:
            settled = db.settle_predictions(conn, load_history, neutral_band=neutral_band)
        finally:
            source_logger.removeHandler(collector)
        _emit(f"SETTLED={settled}")
        _report_source_warnings("", collector.messages)
        _print_pending(conn)
    except sqlite3.Error as exc:
        logger.error("Lỗi SQLite khi settle: %s", exc)
        return 2
    finally:
        conn.close()
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Job hàng đêm — xem docstring module cho chế độ, dòng máy đọc, mã thoát."""
    start = time.perf_counter()
    _configure_streams()
    parser = argparse.ArgumentParser(
        prog="python scripts/daily_run.py",
        description=(
            "Job HÀNG ĐÊM: settle dự báo chờ → huấn luyện & dự báo (walk-forward + "
            "retrain full) cho từng mã trong watchlist/mặc định → ghi DB → tổng kết. "
            "--settle-only chỉ đối chiếu dự báo chờ (không huấn luyện)."
        ),
    )
    parser.add_argument(
        "--settle-only", action="store_true",
        help="Chỉ db.settle_predictions với giá load_history — KHÔNG huấn luyện (kiểm nhanh)",
    )
    parser.add_argument("--config", default=None, help="Đường dẫn config.yaml khác mặc định")
    args = parser.parse_args(argv)

    log_path = PROJECT_ROOT / "data" / "logs" / f"daily_{datetime.now():%Y-%m-%d}.log"
    try:
        _setup_logging(log_path)
    except OSError as exc:
        # Không tạo được log file → lỗi môi trường: thoát sạch 2, vẫn in DURATION.
        print(f"[daily_run] Không tạo được file log {log_path}: {exc}", file=sys.stderr)
        print(f"DURATION_SECONDS={time.perf_counter() - start:.1f}")
        return 2
    logger.info("=== daily_run bắt đầu (log: %s) ===", log_path)

    exit_code = 2
    try:
        try:
            cfg = pipeline.load_config(args.config)
            tickers = _resolve_tickers(cfg)
            # Vùng neutral kiểm NGAY ở pha cấu hình, cho CẢ hai chế độ: sai kiểu
            # (null/"abc") hoặc ≤ 0 là lỗi cấu hình → exit 2 sạch TRƯỚC khi đụng
            # DB. Nếu không, --settle-only sẽ settle im lặng với band sai
            # (outcomes INSERT OR IGNORE không bao giờ được đối chiếu lại) còn
            # full sẽ chết từng mã với exit 1 — sai contract "2 = lỗi cấu hình".
            neutral_band = _neutral_band_fraction(cfg)
        except (pipeline.PipelineError, WatchlistError, ValueError, yaml.YAMLError) as exc:
            # yaml.YAMLError KHÔNG con của ValueError (YAMLError→Exception) — phải
            # bắt riêng để config.yaml sai cú pháp là exit 2 sạch, không traceback.
            logger.error("Lỗi cấu hình — dừng trước khi chạy: %s: %s", type(exc).__name__, exc)
            return 2
        logger.info("Danh sách mã hôm nay: %s", ", ".join(tickers))
        exit_code = (
            _run_settle_only(cfg, neutral_band) if args.settle_only else _run_full(cfg, tickers)
        )
        return exit_code
    except Exception:  # noqa: BLE001 — lỗi bất ngờ của runner: log đủ traceback vào
        # file log (stderr không được giữ lại trong job đêm) rồi thoát 1.
        logger.exception("daily_run gặp lỗi bất ngờ — thoát 1")
        exit_code = 1
        return exit_code
    finally:
        _emit(f"DURATION_SECONDS={time.perf_counter() - start:.1f}")
        logger.info("=== daily_run kết thúc (exit %s) ===", exit_code)


if __name__ == "__main__":
    sys.exit(main())
