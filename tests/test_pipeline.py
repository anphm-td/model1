"""Kiểm thử src/pipeline + src/forecast + scripts/run_e2e — e2e thu nhỏ, OFFLINE.

Chiến lược (khớp conftest học thuyết của dự án, tự chứa trong file này):
- Khung giá OHLCV giả lập cố định (seed, ~430 phiên, random walk log-normal,
  open/high/low NaN ở 60 phiên đầu để mô phỏng lịch sử chỉ có close/volume của
  nguồn HOSE chính thức) — KHÔNG BAO GIỜ gọi mạng: ``load_history`` /
  ``load_vnindex`` bị monkeypatch tại ``src.pipeline`` (điểm pipeline gọi).
- Config thu nhỏ: initial_train_size 300, retrain 30, embargo 5, LSTM 1 epoch
  (window 20, hidden 8), LightGBM 60 cây, ARIMA lưới 2×2 — chạy đủ 4 model ×
  horizons [1, 5] trong vài chục giây.
- Một lần ``run_pipeline`` DUY NHẤT (fixture module-scoped) chia sẻ cho mọi
  test khẳng định: DB (runs/predictions/metrics_json), artifacts (metrics.csv,
  PNG, SHAP, model files), bất biến baseline (naive skill == 0), số dòng scored
  (các dòng "live" chưa có kết quả KHÔNG vào metrics — chứng minh chống leakage
  ở mức e2e), settle/re-point và API đọc của src/forecast.

Các bất biến chống leakage chi tiết (purge/embargo, scaler fit theo fold,
cửa sổ LSTM, feature causal) đã được chứng minh trong tests/test_evaluation.py,
tests/test_features.py, tests/test_model_*.py — ở đây chỉ kiểm tra đường lắp
ghép không phá các bảo đảm đó.
"""
from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import db, forecast, pipeline  # noqa: E402
from src.evaluation import InsufficientHistoryError, make_folds, WalkForwardConfig  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Hình học fixture ---
N_SESSIONS = 430
INIT_TRAIN = 300
RETRAIN_EVERY = 30
EMBARGO = 5
HORIZONS = (1, 5)

# Số dòng scored kỳ vọng (tổng dòng test trừ các dòng "live" cuối khung):
# test rows = n − (init + embargo) = 125; live = h dòng cuối chưa có kết quả.
N_TEST_ROWS = {1: N_SESSIONS - (INIT_TRAIN + EMBARGO) - 1, 5: N_SESSIONS - (INIT_TRAIN + EMBARGO) - 5}

SEED = 11


def make_ohlcv(
    n: int = N_SESSIONS,
    seed: int = SEED,
    s0: float = 50.0,
    mu: float = 0.0003,
    sig: float = 0.012,
    ohlc_nan_head: int = 60,
) -> pd.DataFrame:
    """OHLCV giả lập: random walk log-normal; open/high/low NaN ở đầu chuỗi."""
    index = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(s0 * np.exp(np.cumsum(rng.normal(mu, sig, n))), index=index)
    frame = pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype("float64"),
        },
        index=index,
    )
    if ohlc_nan_head > 0:
        frame.iloc[:ohlc_nan_head, frame.columns.get_indexer(["open", "high", "low"])] = np.nan
    return frame


def make_config(**overrides: object) -> dict[str, object]:
    """Config thu nhỏ khớp hình học fixture (đủ các mục pipeline đọc)."""
    config: dict[str, object] = {
        "default_ticker": "TCB",
        "horizons": list(HORIZONS),
        "neutral_band_pct": 0.3,
        "data": {"start_date": "2024-01-01"},
        "features": {
            "feature_groups": {"price": True, "market_context": True, "news_sentiment": False}
        },
        "walk_forward": {
            "initial_train_size": INIT_TRAIN,
            "retrain_every": RETRAIN_EVERY,
            "embargo": EMBARGO,
            "shuffle": False,
        },
        "models": {
            "lstm": {"window": 20, "hidden_size": 8, "max_epochs": 1, "batch_size": 64},
            "lightgbm": {"n_estimators": 60},
            "arima": {"max_p": 1, "max_q": 1},
        },
    }
    config.update(overrides)
    return config


class _FakeSources:
    """Nguồn giá giả: dict ticker → frame; thay ``load_history``/``load_vnindex``."""

    def __init__(self, ticker_frame: pd.DataFrame, index_frame: pd.DataFrame) -> None:
        self.frames = {"TCB": ticker_frame, "VNINDEX": index_frame}

    def load_history(self, ticker: str, **kwargs: object) -> pd.DataFrame:
        frame = self.frames.get(str(ticker).upper())
        return frame if frame is not None else pd.DataFrame()

    def load_vnindex(self, **kwargs: object) -> pd.DataFrame:
        return self.frames["VNINDEX"]


@pytest.fixture(scope="module")
def pipeline_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Chạy run_pipeline DUY NHẤT cho cả module test — trả môi trường dùng chung.

    Patch ``load_history``/``load_vnindex`` thủ công (monkeypatch là
    function-scoped, không dùng được trong fixture module-scoped) và luôn
    khôi phục trong finally.
    """
    tmp_path = tmp_path_factory.mktemp("pipeline_e2e")
    ticker_frame = make_ohlcv()
    index_frame = make_ohlcv(seed=SEED + 1, s0=1200.0, mu=0.0002, sig=0.008, ohlc_nan_head=0)
    sources = _FakeSources(ticker_frame, index_frame)
    original_load_history = pipeline.load_history
    original_load_vnindex = pipeline.load_vnindex
    pipeline.load_history = sources.load_history  # type: ignore[assignment]
    pipeline.load_vnindex = sources.load_vnindex  # type: ignore[assignment]
    try:
        reports_dir = tmp_path / "reports"
        models_dir = tmp_path / "models"
        db_path = tmp_path / "app.db"
        result = pipeline.run_pipeline(
            "TCB",
            make_config(),
            db_path=db_path,
            reports_dir=reports_dir,
            models_dir=models_dir,
        )
    finally:
        pipeline.load_history = original_load_history  # type: ignore[assignment]
        pipeline.load_vnindex = original_load_vnindex  # type: ignore[assignment]
    return {
        "result": result,
        "db_path": db_path,
        "reports_dir": reports_dir,
        "models_dir": models_dir,
        "ticker_frame": ticker_frame,
        "index_frame": index_frame,
        "sources": sources,
    }


# ---------------------------------------------------------------------------
# E2E: DB, artifacts, bất invariant baseline
# ---------------------------------------------------------------------------


class TestRunPipelineE2E:
    def test_result_fields(self, pipeline_run: dict[str, object]) -> None:
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        assert result.ticker == "TCB"
        assert result.run_id >= 1
        assert result.horizons == HORIZONS
        assert result.n_sessions == N_SESSIONS
        assert result.data_until == "2025-08-25"  # 430 ngày làm việc từ 2024-01-02
        assert set(result.metrics.keys()) == {"1", "5"}
        for horizon in HORIZONS:
            assert set(result.metrics[str(horizon)].keys()) == set(pipeline.MODEL_FAMILIES)
        assert len(result.forecasts) == len(HORIZONS) * len(pipeline.MODEL_FAMILIES)

    def test_db_rows_and_metrics_json(self, pipeline_run: dict[str, object]) -> None:
        conn = db.connect(pipeline_run["db_path"])  # type: ignore[arg-type]
        try:
            runs = db.list_runs(conn, "TCB")
            assert len(runs) == 1
            run = runs[0]
            assert run["data_until"] == "2025-08-25"
            assert run["model_dir"] == str(Path(pipeline_run["models_dir"]) / "TCB")  # type: ignore[arg-type]
            metrics = json.loads(run["metrics_json"])
            assert set(metrics["per_horizon"]["1"]) == set(pipeline.MODEL_FAMILIES)
            assert metrics["walk_forward"]["embargo"] == EMBARGO

            predictions = db.list_predictions(conn, run_id=int(run["id"]))
            assert len(predictions) == len(HORIZONS) * len(pipeline.MODEL_FAMILIES)
            by_model_h = {(row["model"], row["horizon"]): row for row in predictions}
            assert set(by_model_h) == {
                (family, h) for family in pipeline.MODEL_FAMILIES for h in HORIZONS
            }
            for row in predictions:
                assert row["predicted_price"] is None or row["predicted_price"] > 0
                assert row["predicted_direction"] in {"up", "down", "neutral"}
                assert row["probability_up"] is None or 0.0 <= row["probability_up"] <= 1.0
        finally:
            conn.close()

    def test_naive_baseline_invariants(self, pipeline_run: dict[str, object]) -> None:
        """Persistence phải "hoà" chính nó: skill == 0; dự báo giá == close_t."""
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        for horizon in HORIZONS:
            naive = result.metrics[str(horizon)]["naive"]
            assert naive["reg_skill_vs_persistence_pct"] == pytest.approx(0.0, abs=1e-9)
            assert naive["reg_mae"] == pytest.approx(naive["reg_persistence_mae"], rel=1e-9)
            # scored rows: naive điền CẢ hai phía trên mọi dòng có kết quả
            assert naive["reg_n_rows"] == N_TEST_ROWS[horizon]
            assert naive["cls_n_rows"] == N_TEST_ROWS[horizon]
            assert naive["n_test_rows"] == N_TEST_ROWS[horizon] + horizon  # + dòng live
        forecast_rows = [row for row in result.forecasts if row["model"] == "naive"]
        close_t = float(result.forecasts[0]["close_t"])
        for row in forecast_rows:
            assert row["predicted_price"] == pytest.approx(close_t, rel=1e-12)
            assert row["predicted_direction"] == "neutral"  # persistence: không tín hiệu
            assert row["probability_up"] == pytest.approx(0.5)

    def test_arima_probability_is_sign_binary(self, pipeline_run: dict[str, object]) -> None:
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        for row in result.forecasts:
            if row["model"] == "arima":
                assert row["probability_up"] in (0.0, 1.0)

    def test_artifacts_exist(self, pipeline_run: dict[str, object]) -> None:
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        metrics_csv = Path(result.artifacts["metrics_csv"])
        assert metrics_csv.exists()
        table = pd.read_csv(metrics_csv)
        assert len(table) == len(HORIZONS) * len(pipeline.MODEL_FAMILIES)
        assert set(table["model"]) == set(pipeline.MODEL_FAMILIES)
        assert sorted(table["horizon"].unique()) == sorted(HORIZONS)
        naive_row = table[(table["model"] == "naive") & (table["horizon"] == 1)].iloc[0]
        assert float(naive_row["reg_skill_vs_persistence_pct"]) == pytest.approx(0.0, abs=1e-9)

        charts = [Path(path) for path in result.artifacts["charts"]]
        assert [path.name for path in charts] == ["backtest_h1.png", "backtest_h5.png"]
        for path in charts:
            assert path.exists() and path.stat().st_size > 10_000

        shap_files = [Path(path) for path in result.artifacts["shap"]]
        assert [path.name for path in shap_files] == [
            "shap_importance_h1.csv",
            "shap_importance_h5.csv",
        ]
        for path in shap_files:
            shap_table = pd.read_csv(path)
            assert {"feature", "mean_abs_shap"} <= set(shap_table.columns)
            assert (shap_table["mean_abs_shap"] >= 0).all()

    def test_model_files_saved(self, pipeline_run: dict[str, object]) -> None:
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        model_dir = Path(result.artifacts["model_dir"])
        assert model_dir == Path(pipeline_run["models_dir"]) / "TCB"  # type: ignore[arg-type]
        expected_files = {"naive", "arima", "lightgbm_reg", "lightgbm_cls", "lstm"}
        saved = {path.stem for path in model_dir.glob("*.pkl")}
        assert saved == expected_files
        metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["ticker"] == "TCB"
        assert metadata["horizons"] == list(HORIZONS)
        assert set(metadata["files"]) == expected_files
        assert len(metadata["feature_columns"]) > 0
        assert not any(col.startswith("target_") for col in metadata["feature_columns"])

    def test_live_rows_excluded_from_metrics(self, pipeline_run: dict[str, object]) -> None:
        """Dòng "live" (chưa có kết quả thực tế) không được chấm điểm — e2e check.

        Cùng khung 4 model: chỉ model không điền dự báo mới có n_rows nhỏ hơn;
        naive điền đầy đủ ⇒ scored đúng = n_test − h. Nghịch lý "5 ngày điểm cao
        hơn 1 ngày" kiểu leakage phải không thể xảy ra do sai số dòng.
        """
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        for horizon in HORIZONS:
            for family in pipeline.MODEL_FAMILIES:
                aggregate = result.metrics[str(horizon)][family]
                if family == "naive":
                    assert aggregate["reg_n_rows"] == N_TEST_ROWS[horizon]
                else:
                    assert aggregate["reg_n_rows"] <= N_TEST_ROWS[horizon]


# ---------------------------------------------------------------------------
# Cấu hình/lỗi: rút gọn fold, thiếu dữ liệu, dữ liệu rỗng
# ---------------------------------------------------------------------------


class TestRunPipelineConfig:
    def test_max_folds_truncates_frame_and_folds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _FakeSources(make_ohlcv(), make_ohlcv(seed=12, s0=1200.0, ohlc_nan_head=0))
        monkeypatch.setattr(pipeline, "load_history", sources.load_history)
        monkeypatch.setattr(pipeline, "load_vnindex", sources.load_vnindex)
        tiny = make_config(
            models={
                "lstm": {"window": 20, "hidden_size": 4, "max_epochs": 1},
                "lightgbm": {"n_estimators": 5},
                "arima": {"max_p": 0, "max_q": 0},
            }
        )
        result = pipeline.run_pipeline(
            "TCB",
            tiny,
            db_path=tmp_path / "app.db",
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
            max_folds=2,
        )
        assert result.n_sessions == INIT_TRAIN + EMBARGO + 2 * RETRAIN_EVERY
        for horizon in HORIZONS:
            for family in pipeline.MODEL_FAMILIES:
                assert result.metrics[str(horizon)][family]["n_folds"] == 2

        # Hình học: giới hạn phiên cho đúng 2 fold đầu (đúng công thức engine)
        limit = pipeline._max_folds_limit(
            WalkForwardConfig(initial_train_size=INIT_TRAIN, retrain_every=RETRAIN_EVERY,
                              embargo=EMBARGO, horizon=1),
            2,
        )
        folds = make_folds(
            limit,
            WalkForwardConfig(initial_train_size=INIT_TRAIN, retrain_every=RETRAIN_EVERY,
                              embargo=EMBARGO, horizon=1),
        )
        assert len(folds) == 2

    def test_insufficient_history_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sources = _FakeSources(make_ohlcv(), make_ohlcv(seed=13, s0=1200.0, ohlc_nan_head=0))
        monkeypatch.setattr(pipeline, "load_history", sources.load_history)
        monkeypatch.setattr(pipeline, "load_vnindex", sources.load_vnindex)
        bad = make_config(
            walk_forward={
                "initial_train_size": 4000, "retrain_every": 30, "embargo": 5, "shuffle": False
            }
        )
        with pytest.raises(InsufficientHistoryError):
            pipeline.run_pipeline(
                "TCB", bad,
                db_path=tmp_path / "app.db",
                reports_dir=tmp_path / "reports",
                models_dir=tmp_path / "models",
            )

    def test_empty_price_data_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        sources = _FakeSources(pd.DataFrame(), pd.DataFrame())
        monkeypatch.setattr(pipeline, "load_history", sources.load_history)
        monkeypatch.setattr(pipeline, "load_vnindex", sources.load_vnindex)
        with pytest.raises(pipeline.PipelineError, match="TCB"):
            pipeline.run_pipeline(
                "TCB", make_config(),
                db_path=tmp_path / "app.db",
                reports_dir=tmp_path / "reports",
                models_dir=tmp_path / "models",
            )

    def test_shuffle_config_rejected(self, tmp_path: Path) -> None:
        """shuffle: true trong config phải bị từ chối (engine không có chế độ shuffle)."""
        bad = make_config(
            walk_forward={
                "initial_train_size": INIT_TRAIN, "retrain_every": RETRAIN_EVERY,
                "embargo": EMBARGO, "shuffle": True,
            }
        )
        with pytest.raises(ValueError, match="shuffle"):
            # Không cần dữ liệu: WalkForwardConfig.from_mapping bị kiểm tra trước walk-forward
            pipeline.WalkForwardConfig.from_mapping(bad, horizon=1)  # type: ignore[attr-defined]

    def test_single_model_failure_does_not_kill_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Một model lỗi ở walk-forward → pipeline vẫn chạy tới cùng, lỗi ghi RÕ.

        ARIMA bị ép lỗi (min_train_rows vô lý) — naive/lightgbm/lstm vẫn có
        metrics + dự báo; metrics_json của ARIMA chỉ có khoá ``error``.
        """
        sources = _FakeSources(make_ohlcv(), make_ohlcv(seed=14, s0=1200.0, ohlc_nan_head=0))
        monkeypatch.setattr(pipeline, "load_history", sources.load_history)
        monkeypatch.setattr(pipeline, "load_vnindex", sources.load_vnindex)
        tiny = make_config(
            models={
                "lstm": {"window": 20, "hidden_size": 4, "max_epochs": 1},
                "lightgbm": {"n_estimators": 5},
                "arima": {"min_train_rows": 10**9},
            }
        )
        result = pipeline.run_pipeline(
            "TCB", tiny,
            db_path=tmp_path / "app.db",
            reports_dir=tmp_path / "reports",
            models_dir=tmp_path / "models",
        )
        for horizon in HORIZONS:
            assert "error" in result.metrics[str(horizon)]["arima"]
            for family in ("naive", "lightgbm", "lstm"):
                assert "error" not in result.metrics[str(horizon)][family]
        assert {row["model"] for row in result.forecasts} == {"naive", "lightgbm", "lstm"}
        # metrics.csv có dòng ARIMA với cột error (dòng khác để trống)
        table = pd.read_csv(result.artifacts["metrics_csv"])
        arima_rows = table[table["model"] == "arima"]
        assert arima_rows["error"].notna().all()

    def test_drop_sparse_features(self) -> None:
        """Feature coverage thấp bị bỏ (LSTM cần cửa sổ hữu hạn), log rõ, không đụng close/target."""
        index = pd.bdate_range("2024-01-02", periods=100)
        frame = pd.DataFrame(
            {
                "close": np.linspace(10, 20, 100),
                "f_dense": np.arange(100, dtype="float64"),
                "f_sparse": np.where(np.arange(100) >= 95, 1.0, np.nan),  # 5% hữu hạn
                "f_half": np.where(np.arange(100) % 2 == 0, 1.0, np.nan),  # 50% hữu hạn
                "target_ret_1d": np.arange(100, dtype="float64"),  # không được đụng
            },
            index=index,
        )
        reduced, dropped = pipeline._drop_sparse_features(frame, 0.5)
        assert dropped == ["f_sparse"]
        assert "f_sparse" not in reduced.columns
        assert "f_half" in reduced.columns  # 50% == ngưỡng → giữ (chỉ bỏ < ngưỡng)
        assert "f_dense" in reduced.columns
        assert "close" in reduced.columns
        assert "target_ret_1d" in reduced.columns  # target không bị đụng

        # min_valid_ratio None/0 → tắt, giữ nguyên mọi cột
        untouched, dropped_off = pipeline._drop_sparse_features(frame, None)
        assert dropped_off == [] and list(untouched.columns) == list(frame.columns)

        # Ngưỡng sai → lỗi rõ
        with pytest.raises(pipeline.PipelineError, match="min_valid_ratio"):
            pipeline._drop_sparse_features(frame, 1.5)


# ---------------------------------------------------------------------------
# Settle + re-point target_date (đối chiếu dự báo với giá thực tế)
# ---------------------------------------------------------------------------


class TestSettlement:
    def test_settle_after_new_data_arrives(
        self, pipeline_run: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        result: pipeline.PipelineResult = pipeline_run["result"]  # type: ignore[assignment]
        ticker_frame: pd.DataFrame = pipeline_run["ticker_frame"].copy()  # type: ignore[union-attr]
        # Giá mới 8 phiên, tăng ~+2%/phiên đầu → target h=1/h=5 đều "up"
        ext_index = pd.bdate_range(ticker_frame.index[-1] + pd.Timedelta(days=1), periods=8)
        last_close = float(ticker_frame["close"].iloc[-1])
        extended = pd.concat(
            [
                ticker_frame,
                pd.DataFrame(
                    {
                        "open": last_close,
                        "high": last_close * 1.05,
                        "low": last_close * 0.99,
                        "close": last_close * np.array([1.02, 1.025, 1.03, 1.028, 1.031, 1.033, 1.03, 1.034]),
                        "volume": 2_000_000.0,
                    },
                    index=ext_index,
                ),
            ]
        )
        conn = db.connect(pipeline_run["db_path"])  # type: ignore[arg-type]
        try:
            repointed = pipeline.align_target_dates(conn, "TCB", extended)
            assert repointed == 0  # không có ngày lễ trong fixture → ước lượng đúng sẵn
            settled = db.settle_predictions(conn, lambda tk: extended, neutral_band=0.003)
            # >= số prediction của run 1 (test khác có thể đã thêm prediction phụ)
            assert settled >= len(HORIZONS) * len(pipeline.MODEL_FAMILIES)
            # Idempotent: chạy lại settle → 0 dòng mới
            assert db.settle_predictions(conn, lambda tk: extended, neutral_band=0.003) == 0

            rows = db.list_predictions(conn, run_id=int(result.run_id), ticker="TCB")
            assert all(row["actual_price"] is not None for row in rows)
            naive_h1 = next(
                row for row in rows if row["model"] == "naive" and row["horizon"] == 1
            )
            assert naive_h1["hit"] == 0  # naive: hướng neutral vs thực tế "up"
            # outcome khớp dữ liệu: actual = close tại target_date; abs_error của
            # naive = |close_t − actual| (persistence)
            close_t = float(ticker_frame["close"].iloc[-1])
            actual_price = float(extended.loc[pd.Timestamp(naive_h1["target_date"]), "close"])
            assert naive_h1["actual_price"] == pytest.approx(actual_price)
            assert abs(float(naive_h1["predicted_price"]) - actual_price) == pytest.approx(
                abs(close_t - actual_price)
            )
            lightgbm_h1 = next(
                row for row in rows if row["model"] == "lightgbm" and row["horizon"] == 1
            )
            assert lightgbm_h1["hit"] == 1  # lightgbm dự báo "up", thực tế "up"
        finally:
            conn.close()

    def test_align_target_dates_fixes_holiday_guess(
        self, pipeline_run: dict[str, object]
    ) -> None:
        """Dự báo ghi nhầm target vào ngày nghỉ → được chỉnh về phiên THỰC.

        Mô phỏng: prediction h=1 với target_date "2030-01-01" (không phải phiên);
        khi dữ liệu phiên kế tiếp về, align phải chuyển target về phiên thực tại
        vị trí data_until + 1 (phiên đầu của phần giá mở rộng).
        """
        ticker_frame: pd.DataFrame = pipeline_run["ticker_frame"]  # type: ignore[union-attr]
        ext_index = pd.bdate_range(ticker_frame.index[-1] + pd.Timedelta(days=1), periods=3)
        last_close = float(ticker_frame["close"].iloc[-1])
        extended = pd.concat(
            [
                ticker_frame,
                pd.DataFrame(
                    {
                        "open": last_close,
                        "high": last_close * 1.01,
                        "low": last_close * 0.99,
                        "close": last_close * 1.02,
                        "volume": 2_000_000.0,
                    },
                    index=ext_index,
                ),
            ]
        )
        conn = db.connect(pipeline_run["db_path"])  # type: ignore[arg-type]
        try:
            run = db.list_runs(conn, "TCB")[0]
            prediction_id = db.record_prediction(
                conn, int(run["id"]), "TCB", "naive", "2030-01-01", 1,
                predicted_price=last_close, predicted_direction="up", probability_up=0.5,
            )
            repointed = pipeline.align_target_dates(conn, "TCB", extended)
            assert repointed == 1
            row = next(
                row for row in db.list_predictions(conn, run_id=int(run["id"]))
                if row["id"] == prediction_id
            )
            assert row["target_date"] == ext_index[0].strftime("%Y-%m-%d")
            settled = db.settle_predictions(conn, lambda tk: extended, neutral_band=0.003)
            assert settled >= 1
            outcomes = conn.execute(
                "SELECT actual_price, hit FROM outcomes WHERE prediction_id = ?", (prediction_id,)
            ).fetchall()
            assert len(outcomes) == 1
            assert outcomes[0]["actual_price"] == pytest.approx(last_close * 1.02)
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# src/forecast — API đọc cho dashboard
# ---------------------------------------------------------------------------


class TestForecastReadApi:
    def test_forecast_summary_with_run(
        self, pipeline_run: dict[str, object], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Test phải độc lập thứ tự với TestSettlement: settle là idempotent —
        # chạy lại align+settle ở đây đảm bảo outcomes có mặt (đã settle thì bỏ qua).
        sources: _FakeSources = pipeline_run["sources"]  # type: ignore[assignment]
        monkeypatch.setattr(pipeline, "load_history", sources.load_history)
        monkeypatch.setattr(pipeline, "load_vnindex", sources.load_vnindex)
        ticker_frame: pd.DataFrame = pipeline_run["ticker_frame"]  # type: ignore[union-attr]
        ext_index = pd.bdate_range(ticker_frame.index[-1] + pd.Timedelta(days=1), periods=8)
        last_close = float(ticker_frame["close"].iloc[-1])
        extended = pd.concat(
            [
                ticker_frame,
                pd.DataFrame(
                    {
                        "open": last_close,
                        "high": last_close * 1.05,
                        "low": last_close * 0.99,
                        "close": last_close * 1.02,
                        "volume": 2_000_000.0,
                    },
                    index=ext_index,
                ),
            ]
        )
        conn = db.connect(pipeline_run["db_path"])  # type: ignore[arg-type]
        try:
            pipeline.align_target_dates(conn, "TCB", extended)
            db.settle_predictions(conn, lambda tk: extended, neutral_band=0.003)

            summary = forecast.forecast_summary(conn, "TCB")
            assert summary["has_run"] is True
            assert summary["run"]["id"] >= 1
            assert set(summary["metrics"]["per_horizon"]["1"]) == set(pipeline.MODEL_FAMILIES)
            # >= 8: test align (chạy trước theo thứ tự file) có thể đã thêm 1 prediction
            assert len(summary["predictions"]) >= len(HORIZONS) * len(pipeline.MODEL_FAMILIES)
            assert {row["model"] for row in summary["predictions"]} == set(pipeline.MODEL_FAMILIES)
            settlement = summary["settlement"]
            assert settlement["n_settled"] == settlement["n_predictions"]
            assert settlement["mean_abs_error"] is not None and settlement["mean_abs_error"] > 0

            predictions_frame = forecast.latest_predictions(conn, "TCB")
            assert {"model", "horizon", "predicted_price", "predicted_direction",
                    "probability_up", "actual_price", "hit"} <= set(predictions_frame.columns)
            assert forecast.latest_run(conn, "TCB") is not None
        finally:
            conn.close()

    def test_forecast_summary_without_run(self, tmp_path: Path) -> None:
        conn = db.connect(tmp_path / "empty.db")
        try:
            db.init_db(conn)
            summary = forecast.forecast_summary(conn, "VCB")
            assert summary["has_run"] is False
            assert "chưa được huấn luyện" in summary["message"]
            assert summary["predictions"] == []
            assert forecast.latest_predictions(conn, "VCB").empty
            assert forecast.settlement_stats(conn, "VCB")["n_predictions"] == 0
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# CLI (python -m src.pipeline) — argparse + điều phối, không chạy model thật
# ---------------------------------------------------------------------------


class TestCli:
    @staticmethod
    def _fake_result(ticker: str) -> pipeline.PipelineResult:
        return pipeline.PipelineResult(
            ticker=ticker,
            run_id=1,
            n_sessions=10,
            data_until="2025-08-25",
            horizons=(1,),
            settled=0,
            repointed=0,
            metrics={"1": {"naive": {"reg_mae": 1.0, "reg_skill_vs_persistence_pct": 0.0,
                                     "cls_accuracy": 0.5, "cls_edge_vs_base_rate": 0.0}}},
            forecasts=({"model": "naive", "horizon": 1, "target_date": "2025-08-26",
                        "predicted_price": 10.0, "predicted_direction": "neutral",
                        "probability_up": 0.5, "close_t": 10.0},),
            artifacts={"metrics_csv": "x"},
        )

    def test_main_runs_given_tickers(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        calls: list[tuple[str, object]] = []

        def fake_run_pipeline(ticker: str, config: object, **kwargs: object) -> pipeline.PipelineResult:
            calls.append((ticker, config))
            return self._fake_result(ticker)

        monkeypatch.setattr(pipeline, "run_pipeline", fake_run_pipeline)
        exit_code = pipeline.main(["TCB", "VCB"])
        assert exit_code == 0
        assert [ticker for ticker, _ in calls] == ["TCB", "VCB"]
        assert "TCB" in capsys.readouterr().out

    def test_main_failure_returns_nonzero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def failing_run_pipeline(ticker: str, config: object, **kwargs: object) -> pipeline.PipelineResult:
            raise pipeline.PipelineError("lỗi giả lập")

        monkeypatch.setattr(pipeline, "run_pipeline", failing_run_pipeline)
        assert pipeline.main(["TCB"]) == 1

    def test_main_missing_config_returns_2(self) -> None:
        assert pipeline.main(["TCB", "--config", "khong-ton-tai.yaml"]) == 2

    def test_main_watchlist_resolution(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeWatchlist:
            def list_tickers(self) -> list[str]:
                return ["AAA", "BBB"]

        monkeypatch.setattr("src.watchlist.Watchlist", FakeWatchlist)
        calls: list[str] = []
        monkeypatch.setattr(
            pipeline, "run_pipeline",
            lambda ticker, config, **kw: (calls.append(ticker), self._fake_result(ticker))[1],
        )
        assert pipeline.main(["--watchlist"]) == 0
        assert calls == ["AAA", "BBB"]

    def test_main_empty_watchlist_uses_default_ticker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Watchlist thật trên tệp tmp rỗng (không đụng data/watchlist.json)
        from src.watchlist import Watchlist as RealWatchlist

        monkeypatch.setattr(
            "src.watchlist.Watchlist",
            lambda: RealWatchlist(tmp_path / "watchlist.json"),
        )
        calls: list[str] = []
        monkeypatch.setattr(
            pipeline, "run_pipeline",
            lambda ticker, config, **kw: (calls.append(ticker), self._fake_result(ticker))[1],
        )
        assert pipeline.main([]) == 0
        assert calls == ["TCB"]  # default_ticker của make_config — config thật là config.yaml


# ---------------------------------------------------------------------------
# scripts/run_e2e.py — phần rút gọn của e2e thật
# ---------------------------------------------------------------------------


def _load_run_e2e() -> object:
    spec = importlib.util.spec_from_file_location(
        "run_e2e", PROJECT_ROOT / "scripts" / "run_e2e.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestRunE2EScript:
    def test_reduced_config_overrides_lstm_only(self) -> None:
        run_e2e = _load_run_e2e()
        original: dict[str, object] = make_config()
        original["models"] = {  # type: ignore[assignment]
            "lstm": {"window": 30, "max_epochs": 10},
            "lightgbm": {"n_estimators": 60},
            "arima": {"max_p": 1, "max_q": 1},
        }
        snapshot = json.dumps(original, sort_keys=True)
        reduced = run_e2e.reduced_config(original)  # type: ignore[attr-defined]
        # Chỉ max_epochs của LSTM bị rút xuống 3; phần còn lại giữ nguyên
        assert reduced["models"]["lstm"]["max_epochs"] == run_e2e.E2E_LSTM_EPOCHS == 3
        assert reduced["models"]["lstm"]["window"] == 30
        assert reduced["models"]["lightgbm"] == {"n_estimators": 60}
        assert reduced["walk_forward"] == original["walk_forward"]
        # Config gốc KHÔNG bị sửa
        assert json.dumps(original, sort_keys=True) == snapshot

    def test_four_folds_geometry_matches_real_config(self) -> None:
        """max_folds=4 trên config thật (750/21/embargo 5) cho đúng 4 fold đầu."""
        real_cfg = WalkForwardConfig(initial_train_size=750, retrain_every=21, embargo=5, horizon=1)
        limit = pipeline._max_folds_limit(real_cfg, 4)
        folds = make_folds(limit, real_cfg)
        assert len(folds) == 4
        assert folds[0].n_test == 21 and folds[-1].n_test == 21
        assert folds[0].gap_rows == 5  # embargo giữ nguyên — không phá chống leakage
