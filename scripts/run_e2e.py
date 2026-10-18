#!/usr/bin/env python
"""E2E thật với cấu hình rút gọn — kiểm chứng trọn đường pipeline nhanh.

Đi ĐỦ đường thật của ``src.pipeline.run_pipeline``: settle dự báo cũ → tải/cập
nhật dữ liệu giá (CAFEF + fallback, cache parquet) → feature → labeling →
walk-forward CẢ 4 model (naive/ARIMA/LightGBM/LSTM) → retrain full → dự báo
phiên kế tiếp → ghi runs/predictions vào data/app.db → lưu model
models/{ticker}/ → xuất reports/{ticker}/ (metrics.csv, backtest PNG, SHAP).

Chỉ RÚT GỌN hai điểm (còn lại đúng config.yaml):
- LSTM ``max_epochs = 3`` (config thật: 10);
- chỉ chạy ``max_folds = 4`` fold walk-forward ĐẦU TIÊN (khung được cắt đúng
  đủ phiên; retrain full + dự báo live vẫn dùng đúng khung đó).

Cách chạy:
    python scripts/run_e2e.py            # mã mặc định (config: default_ticker)
    python scripts/run_e2e.py TCB [VCB…] # chỉ định mã (tham số vị trí)
    python scripts/run_e2e.py --ticker VCB          # tương đương dòng trên
    python scripts/run_e2e.py --ticker VCB --ticker TCB  # --ticker lặp được
    python scripts/run_e2e.py --watchlist

Mã thoát: 0 khi mọi mã chạy tới cùng; 1 khi có mã lỗi (lỗi in rõ từng mã).
"""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from collections.abc import Mapping
from pathlib import Path

# Cho phép chạy trực tiếp ``python scripts/run_e2e.py`` (không qua package).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import pipeline  # noqa: E402
from src.db import DEFAULT_DB_PATH  # noqa: E402
from src.watchlist import Watchlist, WatchlistError  # noqa: E402

logger = logging.getLogger("run_e2e")

#: Số fold walk-forward đầu tiên được chạy (rút gọn).
E2E_MAX_FOLDS: int = 4

#: Số epoch LSTM cho e2e (rút gọn; config thật 10 — models/lstm_model.py).
E2E_LSTM_EPOCHS: int = 3


def reduced_config(config: Mapping[str, object] | None) -> dict[str, object]:
    """Bản sao config với phần rút gọn e2e — KHÔNG sửa config.yaml gốc.

    Chỉ đụng ``models.lstm.max_epochs`` (ghi đè nếu đã có, thêm nếu chưa).
    Số fold được rút gọn bằng tham số ``max_folds`` của run_pipeline, không
    phải bằng đổi cấu hình walk-forward (initial_train_size/retrain/embargo
    giữ nguyên hình học thật).
    """
    reduced = copy.deepcopy(dict(config)) if config is not None else {}
    models = reduced.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError("config.models phải là mapping")
    lstm = dict(models.get("lstm") or {})
    lstm["max_epochs"] = E2E_LSTM_EPOCHS
    models["lstm"] = lstm
    return reduced


def _resolve_tickers(args: argparse.Namespace, config: Mapping[str, object]) -> list[str]:
    """Mã cần chạy: argv → watchlist → mã mặc định trong config."""
    if args.tickers:
        return [str(tk).strip().upper() for tk in args.tickers]
    try:
        watchlist_tickers = Watchlist().list_tickers()
    except WatchlistError as exc:
        raise ValueError(f"Watchlist không đọc được: {exc}") from exc
    if watchlist_tickers:
        return watchlist_tickers
    return [str(config.get("default_ticker", "TCB")).strip().upper()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/run_e2e.py",
        description=(
            "E2E thật rút gọn: LSTM 3 epoch, 4 fold walk-forward đầu — mọi bước "
            "khác (settle, dữ liệu mạng, DB, model, reports) đúng như pipeline thật."
        ),
    )
    parser.add_argument("tickers", nargs="*", help="Mã HOSE (bỏ trống → watchlist/mặc định)")
    parser.add_argument(
        "--ticker", action="append", default=[], metavar="MÃ",
        help="Mã HOSE — tương đương tham số vị trí, lặp được cho nhiều mã "
        "(vd. --ticker VCB --ticker TCB)",
    )
    parser.add_argument("--watchlist", action="store_true", help="Chạy cả watchlist")
    parser.add_argument("--config", default=None, help="Đường dẫn config.yaml khác mặc định")
    parser.add_argument("--refresh", action="store_true", help="Tải lại toàn bộ lịch sử giá")
    args = parser.parse_args(argv)
    # Gộp tham số vị trí + --ticker (lặp được), hoá chuẩn IN HOA, bỏ trùng giữ thứ tự.
    args.tickers = list(
        dict.fromkeys(
            str(tk).strip().upper() for tk in [*args.tickers, *args.ticker] if str(tk).strip()
        )
    )

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    try:
        config = pipeline.load_config(args.config)
        if args.watchlist:
            args.tickers = []
        tickers = _resolve_tickers(args, config)
        config = reduced_config(config)
    except (ValueError, pipeline.PipelineError) as exc:
        logger.error("%s", exc)
        return 2

    logger.info(
        "E2E rút gọn: LSTM max_epochs=%d, max_folds=%d — mã: %s",
        E2E_LSTM_EPOCHS,
        E2E_MAX_FOLDS,
        ", ".join(tickers),
    )
    failures: list[str] = []
    for ticker in tickers:
        try:
            result = pipeline.run_pipeline(
                ticker,
                config,
                refresh=args.refresh,
                max_folds=E2E_MAX_FOLDS,
            )
        except Exception as exc:  # noqa: BLE001 — mỗi mã tự chịu lỗi, báo rõ rồi chạy mã kế
            logger.error("[FAIL] %s: %s: %s", ticker, type(exc).__name__, exc)
            failures.append(ticker)
            continue
        pipeline._print_summary(result)

    if failures:
        logger.error("E2E kết thúc với LỖI ở %d/%d mã: %s", len(failures), len(tickers), failures)
        return 1
    logger.info("E2E hoàn thành: %d mã OK (DB: %s)", len(tickers), DEFAULT_DB_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
