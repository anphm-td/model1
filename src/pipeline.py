"""Pipeline — điều phối trọn vẹn một lần "Huấn luyện & dự báo" cho một mã.

``run_pipeline(ticker)`` là điểm vào duy nhất cho nút "Huấn luyện & dự báo"
của dashboard (app/dashboard.py) và cho CLI (``python -m src.pipeline MÃ``).
Chuỗi bước:

1. ``db.settle_predictions()`` — đối chiếu các dự báo CHƯA settle với giá đã có
   trong cache, trước khi huấn luyện;
2. tải/cập nhật dữ liệu giá 1D (``data_loader.load_history`` — cache parquet
   data/raw + data/processed, cập nhật tăng dần, KHÔNG ffill ngày nghỉ);
3. sinh feature theo nhóm bật trong config (``features.assemble_features``) và
   target theo horizons + vùng neutral (``labeling.build_targets``);
4. walk-forward validation cho CẢ 4 model (naive persistence, ARIMA, LightGBM,
   LSTM) cho từng horizon — engine ``evaluation.walk_forward`` bảo đảm purge +
   embargo, model/scaler fit riêng từng fold, KHÔNG shuffle;
5. retrain trên TOÀN BỘ khung → dự báo phiên kế tiếp (giá + hướng + xác suất)
   cho từng model/horizon;
6. ghi runs/predictions vào SQLite (``src/db``) — metrics_json chứa skill vs
   persistence (MAE) và edge vs base rate (hướng) của từng model/horizon;
7. lưu model đã retrain vào ``models/{ticker}/`` và xuất
   ``reports/{ticker}/metrics.csv`` + biểu đồ ``backtest_h{h}.png`` (+ SHAP
   importance của LightGBM theo fold).

Ngày phiên của dự báo phải trung thực: ``target_date`` ghi lúc dự báo là
ƯỚC LƯỢNG (các ngày làm việc kế tiếp — lịch nghỉ lễ HOSE chưa biết). Sau khi
dữ liệu mới về, ``align_target_dates()`` chỉnh các dự báo chưa settle về phiên
THỰC tại vị trí ``data_until + horizon`` (tránh ngày lễ khiến dự báo lệch ngày
và không bao giờ được đối chiếu), rồi ``settle_predictions()`` chạy lại.

Chống leakage (ưu tiên số 1): pipeline KHÔNG tự tính bất kỳ thống kê nào —
feature đến từ src/features (causal, có test riêng), target từ src/labeling
(dịch đúng chiều), ranh giới train/test từ src/evaluation (purge + embargo có
test riêng). Pipeline chỉ lắp ráp và bảo đảm dữ liệu truyền đúng hợp đồng.

Định hướng (hướng) của dự báo: quy đổi ``predicted_price`` về lợi nhuận đơn
giản so với ``close_t`` rồi phân loại theo vùng neutral ±neutral_band_pct —
cùng quy ước 3 lớp {up, down, neutral} với ``db.classify_direction`` và
``labeling``. ``probability_up`` là phía phân loại riêng của model (khung
"LỚN vs không-LỚN" của engine) — hai phía có thể không nhất quán với nhau,
được báo cáo trung thực chứ không ép khớp.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import sqlite3
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import db
from .data_loader import DATA_DIR as DEFAULT_DATA_DIR
from .data_loader import DEFAULT_START, load_history, load_vnindex
from .evaluation import (
    METRIC_KEYS,
    TARGET_COLUMN_PREFIXES,
    InsufficientHistoryError,
    WalkForwardConfig,
    WalkForwardResult,
    walk_forward,
)
from .features import assemble_features, feature_groups
from .labeling import build_targets
from .models.arima_model import ARIMAModel
from .models.lightgbm_model import (
    LightGBMClassifierModel,
    LightGBMRegressorModel,
    export_shap_importance,
)
from .models.lstm_model import LstmModel
from .models.naive import NaiveModel

logger = logging.getLogger(__name__)

#: Thư mục gốc dự án (src/pipeline.py → …/model1).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

#: config.yaml mặc định ở gốc dự án.
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

#: Bốn họ model bắt buộc của walk-forward (thứ tự cố định cho báo cáo).
MODEL_FAMILIES: tuple[str, ...] = ("naive", "arima", "lightgbm", "lstm")

#: Tên model chính dùng vẽ biểu đồ backtest (regression side của LightGBM).
PRIMARY_CHART_FAMILY: str = "lightgbm"

#: Tham số hợp lệ của từng họ model trong config.yaml (mục ``models``).
_MODEL_OPTION_KEYS: dict[str, tuple[str, ...]] = {
    "naive": ("direction", "up_probability"),
    "arima": (
        "max_p",
        "max_q",
        "max_d",
        "adf_alpha",
        "close_column",
        "seasonal_order",
        "min_train_rows",
    ),
    # lightgbm/lstm tự đọc và tự kiểm tra section của mình
    # (models/lightgbm_model.py, models/lstm_model.py).
}


class PipelineError(RuntimeError):
    """Pipeline không chạy được tới cùng — lỗi dữ liệu/cấu hình với thông điệp rõ."""


# ---------------------------------------------------------------------------
# Kết quả trả về
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResult:
    """Tổng kết một lần ``run_pipeline`` — dashboard/CLI đọc trực tiếp."""

    ticker: str
    run_id: int
    n_sessions: int
    data_until: str
    horizons: tuple[int, ...]
    #: Số prediction được settle (2 lần settle cộng dồn) trong lần chạy này.
    settled: int
    #: Số prediction được chỉnh target_date về phiên thực (lệch ngày lễ).
    repointed: int
    #: {str(horizon): {model: aggregate metrics}} — metrics gộp walk-forward.
    metrics: dict[str, dict[str, dict[str, Any]]]
    #: Các dự báo phiên kế tiếp đã ghi DB (mỗi model × horizon một dòng).
    forecasts: tuple[dict[str, Any], ...]
    #: Đường dẫn artifacts đã ghi: metrics_csv, charts, model_dir, shap, db.
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Biến thành dict JSON-an toàn (cho log/hiển thị)."""
        return _json_safe(asdict(self))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Đọc config.yaml (mặc định ở gốc dự án); trả dict đã parse."""
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise PipelineError(f"Không tìm thấy file cấu hình: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, Mapping):
        raise PipelineError(f"config.yaml phải là mapping YAML, nhận {type(config).__name__}")
    return dict(config)


def _horizons_of(config: Mapping[str, Any]) -> tuple[int, ...]:
    """Danh sách horizons từ config (mặc định [1, 5]) — hợp lệ, duy nhất, >= 1."""
    raw = config.get("horizons", [1, 5])
    if not isinstance(raw, (list, tuple)) or not raw:
        raise PipelineError(f"horizons phải là danh sách số nguyên không rỗng, nhận {raw!r}")
    horizons: list[int] = []
    for item in raw:
        h = int(item)
        if h < 1:
            raise PipelineError(f"horizon phải >= 1, nhận {item!r}")
        if h in horizons:
            raise PipelineError(f"horizons có phần tử trùng: {raw!r}")
        horizons.append(h)
    return tuple(horizons)


def _neutral_band_fraction(config: Mapping[str, Any]) -> float:
    """Vùng neutral ±neutral_band_pct (%) → tỉ lệ (0.3 → 0.003)."""
    pct = float(config.get("neutral_band_pct", 0.3))
    if pct <= 0:
        raise PipelineError(f"neutral_band_pct phải > 0 (%), nhận {pct}")
    return pct / 100.0


def _family_kwargs(config: Mapping[str, Any], family: str) -> dict[str, Any]:
    """Đọc ``config.models.<family>`` cho naive/arima — kiểm tra tên khoá rõ ràng."""
    models = config.get("models")
    section = models.get(family) if isinstance(models, Mapping) else None
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise PipelineError(
            f"config.models.{family} phải là mapping, nhận {type(section).__name__}"
        )
    allowed = _MODEL_OPTION_KEYS.get(family)
    if allowed is None:  # lightgbm/lstm tự kiểm tra trong module của mình
        return dict(section)
    unknown = sorted(str(key) for key in section if str(key) not in allowed)
    if unknown:
        raise PipelineError(
            f"config.models.{family} có tham số không nhận: {unknown} — "
            f"các tham số hợp lệ: {sorted(allowed)}."
        )
    return dict(section)


def _arima_kwargs(config: Mapping[str, Any], gap: int) -> dict[str, Any]:
    """Tham số ARIMAModel từ config; ``embargo`` của engine KHÔNG bị ghi đè.

    ``embargo`` phải bằng khoảng cách fold thực tế ``max(embargo_cfg, horizon)``
    (evaluation.make_folds) — nếu người dùng đặt trong config, bỏ và cảnh báo.
    """
    kwargs = _family_kwargs(config, "arima")
    for key in ("embargo", "horizon"):
        if key in kwargs:
            logger.warning(
                "Bỏ qua config.models.arima.%s=%r — pipeline tự đặt theo hình học "
                "walk-forward (embargo=%d).",
                key,
                kwargs.pop(key),
                gap,
            )
    return kwargs


# ---------------------------------------------------------------------------
# JSON & tiện ích nhỏ
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """Quy đổi giá trị về kiểu JSON được: numpy → python, NaN/inf → None."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def _estimate_target_dates(last_session: pd.Timestamp, horizons: tuple[int, ...]) -> dict[int, str]:
    """ƯỚC LƯỢNG ngày phiên mục tiêu: ``h`` ngày làm việc kế tiếp sau phiên cuối.

    Lịch nghỉ lễ HOSE chưa biết trước → đây là ước lượng tốt nhất lúc ghi dự
    báo; ``align_target_dates`` sẽ chỉnh về phiên thực khi dữ liệu về.
    """
    future = pd.bdate_range(last_session + pd.Timedelta(days=1), periods=max(horizons))
    return {h: future[h - 1].strftime("%Y-%m-%d") for h in horizons}


def align_target_dates(
    conn: sqlite3.Connection,
    ticker: str,
    price_frame: pd.DataFrame,
) -> int:
    """Chỉnh ``target_date`` của dự báo chưa settle về PHIÊN THỰC đã có dữ liệu.

    Dự báo được ghi với target_date là ngày làm việc ước lượng; nếu giữa đường
    có ngày lễ HOSE, phiên mục tiêu thực tế nằm ở vị trí ``phiên data_until +
    horizon`` của chuỗi giá. Khi phiên thực đã có trong ``price_frame`` mà
    target_date ước lượng lệch nhau, cập nhật lại target_date (sổ sách, KHÔNG
    đụng giá/hướng dự báo) để ``db.settle_predictions`` đối chiếu đúng phiên.
    Chỉ đụng prediction CHƯA settle của ``ticker``. Trả về số dòng được chỉnh.
    """
    rows = conn.execute(
        "SELECT p.id AS id, p.target_date AS target_date, p.horizon AS horizon, "
        "r.data_until AS data_until FROM predictions p "
        "JOIN runs r ON r.id = p.run_id "
        "LEFT JOIN outcomes o ON o.prediction_id = p.id "
        "WHERE p.ticker = ? AND o.prediction_id IS NULL",
        (str(ticker).upper(),),
    ).fetchall()
    if not rows or price_frame.empty:
        return 0
    index = price_frame.index
    repointed = 0
    for row in rows:
        data_until = row["data_until"]
        if not data_until:
            continue
        base = pd.Timestamp(data_until)
        if base not in index:
            continue
        position = int(index.get_loc(base))
        target_position = position + int(row["horizon"])
        if target_position >= len(index):
            continue  # phiên mục tiêu chưa có dữ liệu — chưa thể biết ngày thật
        true_target = index[target_position].strftime("%Y-%m-%d")
        if true_target != str(row["target_date"])[:10]:
            conn.execute(
                "UPDATE predictions SET target_date = ? WHERE id = ?",
                (true_target, int(row["id"])),
            )
            logger.info(
                "align_target_dates: prediction #%d target %s → phiên thực %s "
                "(data_until %s + h=%d, lệch ngày nghỉ lễ)",
                row["id"],
                str(row["target_date"])[:10],
                true_target,
                data_until,
                int(row["horizon"]),
            )
            repointed += 1
    conn.commit()
    return repointed


# ---------------------------------------------------------------------------
# Walk-forward cho từng họ model
# ---------------------------------------------------------------------------


def _make_factory(
    family: str,
    horizon: int,
    config: Mapping[str, Any],
    gap: int,
    created: list[Any],
) -> Callable[[], Any]:
    """Factory KHÔNG tham số cho engine — tạo model MỚI mỗi fold (ghi nhận lại
    vào ``created`` để xuất SHAP của LightGBM theo fold)."""

    def factory() -> Any:
        if family == "naive":
            model: Any = NaiveModel(**_family_kwargs(config, "naive"))
        elif family == "arima":
            model = ARIMAModel(horizon=horizon, embargo=gap, **_arima_kwargs(config, gap))
        elif family == "lightgbm":
            model = LightGBMRegressorModel(horizon=horizon, config=dict(config))
        elif family == "lstm":
            model = LstmModel.from_config(dict(config), horizon=horizon)
        else:  # pragma: no cover — MODEL_FAMILIES chặn trước
            raise PipelineError(f"Họ model không hỗ trợ: {family!r}")
        created.append(model)
        return model

    return factory


def _run_family(
    frame: pd.DataFrame,
    family: str,
    horizon: int,
    config: Mapping[str, Any],
    wf_cfg: WalkForwardConfig,
    gap: int,
) -> tuple[dict[str, Any], WalkForwardResult, list[Any]]:
    """Chạy walk-forward cho MỘT họ model; trả (aggregate, result_phía_giá, models).

    - naive/arima/lstm: một lần ``walk_forward`` (model phát cả giá lẫn xác suất).
    - lightgbm: HAI lần chạy — Regressor (phía giá) + Classifier (phía xác
      suất); aggregate gộp: khoá ``reg_*``/đếm từ run regressor, khoá ``cls_*``
      từ run classifier (engine chấm hai phía độc lập theo hợp đồng
      ModelPrediction — mỗi phía trên đúng các dòng model điền dự báo).
    """
    if family != "lightgbm":
        created: list[Any] = []
        result = walk_forward(
            frame, _make_factory(family, horizon, config, gap, created), wf_cfg
        )
        return dict(result.aggregate), result, created

    created_reg: list[Any] = []
    result_reg = walk_forward(
        frame, _make_factory("lightgbm", horizon, config, gap, created_reg), wf_cfg
    )
    result_cls = walk_forward(
        frame,
        lambda: LightGBMClassifierModel(horizon=horizon, config=dict(config)),
        wf_cfg,
    )
    aggregate = {
        key: value for key, value in result_reg.aggregate.items() if not key.startswith("cls_")
    }
    aggregate.update(
        {key: result_cls.aggregate[key] for key in result_cls.aggregate if key.startswith("cls_")}
    )
    return aggregate, result_reg, created_reg


def _drop_sparse_features(
    frame: pd.DataFrame, min_valid_ratio: float | None
) -> tuple[pd.DataFrame, list[str]]:
    """Bỏ feature có tỉ lệ giá trị hữu hạn < ``min_valid_ratio`` (mặc định 0.5).

    Bối cảnh thực tế HOSE: lịch sử cũ từ nguồn chính thức chỉ có close+volume →
    các feature phụ thuộc high/low (vd. stoch_%K/%D) NaN gần như toàn chuỗi.
    LightGBM chịu NaN gốc nhưng LSTM YÊU CẦU cửa sổ window phiên hữu hạn đầy đủ
    — feature coverage thấp làm MỌI cửa sổ chứa NaN và model không còn mẫu train
    (LstmModel raise "không có mẫu train hợp lệ"). Chính sách: không bịa dữ liệu
    (không ffill) — bỏ hẳn cột coverage thấp và LOG RÕ tên cột.

    Trung thực về leakage: việc LOẠI cột chỉ làm MẤT thông tin, không thể đưa
    thông tin tương lai vào dự báo (khác fit thống kê trên toàn chuỗi). Tỉ lệ
    được tính trên toàn khung nhưng quyết định là "cột này có sẵn trong lịch sử
    hay không" — dữ kiện về nguồn dữ liệu, không phải tín hiệu giá.

    Không đụng cột ``close`` và các cột target. ``min_valid_ratio`` đọc từ
    ``config.features.min_valid_ratio`` (None/0 → tắt, giữ nguyên mọi cột).
    """
    if not min_valid_ratio or min_valid_ratio <= 0:
        return frame, []
    if not 0 < min_valid_ratio <= 1:
        raise PipelineError(
            f"features.min_valid_ratio phải trong (0, 1], nhận {min_valid_ratio}"
        )
    feature_columns = [
        col
        for col in frame.columns
        if col != "close" and not str(col).lower().startswith(TARGET_COLUMN_PREFIXES)
    ]
    if not feature_columns:
        return frame, []
    n_rows = len(frame)
    values = frame[feature_columns].to_numpy(dtype="float64", na_value=np.nan)
    valid_ratio = np.isfinite(values).mean(axis=0)
    dropped = [
        col
        for col, ratio in zip(feature_columns, valid_ratio, strict=True)
        if ratio < min_valid_ratio
    ]
    if not dropped:
        return frame, []
    detail = ", ".join(
        f"{col} ({ratio:.1%})"
        for col, ratio in zip(feature_columns, valid_ratio, strict=True)
        if ratio < min_valid_ratio
    )
    logger.warning(
        "Bỏ %d feature có tỉ lệ giá trị hữu hạn < %.0f%%: %s — nguồn dữ liệu không "
        "phủ đủ lịch sử cho các cột này (không ffill/bịa dữ liệu); LightGBM vẫn "
        "chạy được với NaN nhưng LSTM cần cửa sổ đầy đủ.",
        len(dropped),
        min_valid_ratio * 100,
        detail,
    )
    return frame.drop(columns=dropped), dropped


def _max_folds_limit(wf_cfg: WalkForwardConfig, max_folds: int) -> int:
    """Số phiên tối đa để engine chạy ĐÚNG ``max_folds`` fold đầu tiên.

    make_folds: test block k bắt đầu tại ``initial + embargo + k*retrain`` và
    dài ``retrain`` — giữ đúng ``initial + embargo + max_folds*retrain`` phiên
    là được đúng max_folds fold (mỗi fold đủ retrain phiên).
    """
    return wf_cfg.initial_train_size + wf_cfg.embargo + int(max_folds) * wf_cfg.retrain_every


# ---------------------------------------------------------------------------
# Retrain toàn bộ khung + dự báo phiên kế tiếp
# ---------------------------------------------------------------------------


def _strip_target_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Bỏ mọi cột target (tiền tố ``evaluation.TARGET_COLUMN_PREFIXES``) khỏi khung
    đưa vào ``model.predict`` — model KHÔNG được nhìn nhãn, kể cả nhãn NaN của
    các phiên cuối chưa có kết quả."""
    drop = [col for col in frame.columns if str(col).lower().startswith(TARGET_COLUMN_PREFIXES)]
    return frame.drop(columns=drop)


def _full_retrain_forecast(
    frame: pd.DataFrame,
    family: str,
    horizon: int,
    config: Mapping[str, Any],
    gap: int,
) -> tuple[float | None, float | None, dict[str, Any]]:
    """Retrain trên TOÀN BỘ khung và dự báo cho dòng cuối (phiên t mới nhất).

    Trả (predicted_price, probability_up, models_picked). ``predicted_price``
    là dự báo GIÁ close tại phiên t+h (quy đổi từ log-return của model);
    ``None``/``NaN`` khi model không cho dự báo giá hợp lệ (được ghi DB là
    NULL — không bịa). ARIMA retrain full dùng ``embargo=0``: biên train là
    phiên cuối cùng, dự báo h bước kể từ đó (khác walk-forward nơi biên cách
    test ``gap`` phiên — engine giao lá cờ này qua tham số embargo).

    Returns:
        (predicted_price, probability_up, models) — models là dict các model
        đã fit (dùng để pickle vào models/{ticker}/).
    """
    close_t = float(frame["close"].iloc[-1])
    last_row = _strip_target_rows(frame.iloc[[-1]])
    models: dict[str, Any] = {}

    if family == "naive":
        model = NaiveModel(**_family_kwargs(config, "naive")).fit(frame, None)
        prediction = model.predict(last_row)
        models["naive"] = model
        price = float(np.asarray(prediction.predicted_price, dtype="float64")[0])
        prob = float(np.asarray(prediction.probability_up, dtype="float64")[0])
        return price, prob, models

    if family == "arima":
        model = ARIMAModel(horizon=horizon, embargo=0, **_arima_kwargs(config, gap)).fit(frame, None)
        prediction = model.predict(last_row)
        models["arima"] = model
        ret = float(np.asarray(prediction.predicted_return, dtype="float64")[0])
        prob = float(np.asarray(prediction.probability_up, dtype="float64")[0])
        price = close_t * math.exp(ret) if math.isfinite(ret) else None
        return price, prob, models

    if family == "lightgbm":
        reg = LightGBMRegressorModel(horizon=horizon, config=dict(config)).fit(frame, None)
        cls = LightGBMClassifierModel(horizon=horizon, config=dict(config)).fit(frame, None)
        models["lightgbm_reg"] = reg
        models["lightgbm_cls"] = cls
        ret = float(np.asarray(reg.predict(last_row).predicted_return, dtype="float64")[0])
        prob = float(np.asarray(cls.predict(last_row).probability_up, dtype="float64")[0])
        price = close_t * math.exp(ret) if math.isfinite(ret) else None
        return price, prob, models

    if family == "lstm":
        model = LstmModel.from_config(dict(config), horizon=horizon).fit(frame, None)
        prediction = model.predict(last_row)
        models["lstm"] = model
        ret = float(np.asarray(prediction.predicted_return, dtype="float64")[0])
        prob = float(np.asarray(prediction.probability_up, dtype="float64")[0])
        price = close_t * math.exp(ret) if math.isfinite(ret) else None
        return price, prob, models

    raise PipelineError(f"Họ model không hỗ trợ: {family!r}")  # pragma: no cover


def _direction_from_price(
    predicted_price: float | None, close_t: float, neutral_band: float
) -> str | None:
    """Hướng 3 lớp của dự báo giá: lợi nhuận đơn giản implied vs vùng neutral."""
    if predicted_price is None or not math.isfinite(predicted_price) or close_t <= 0:
        return None
    return db.classify_direction(predicted_price / close_t - 1.0, neutral_band)


# ---------------------------------------------------------------------------
# Artifacts: metrics.csv, biểu đồ PNG, model files
# ---------------------------------------------------------------------------


def _export_metrics_csv(
    rows: list[dict[str, Any]], ticker: str, reports_dir: Path
) -> Path:
    """Ghi reports/{ticker}/metrics.csv — một dòng mỗi (horizon, model)."""
    out_dir = reports_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "metrics.csv"
    columns = ["horizon", "model", "n_folds", *METRIC_KEYS, "error"]
    table = pd.DataFrame(rows).reindex(columns=columns)
    table.to_csv(out_path, index=False, encoding="utf-8", float_format="%.6g")
    logger.info("Đã ghi %s (%d dòng)", out_path, len(table))
    return out_path


def _export_shap(
    fold_models: list[Any], ticker: str, horizon: int, reports_dir: Path
) -> Path | None:
    """Xuất SHAP importance của LightGBM theo fold → shap_importance_h{h}.csv.

    ``export_shap_importance`` ghi tên cố định shap_importance.csv — đổi tên
    sau khi ghi để giữ MỘT file cho MỖI horizon. Khi mọi fold đều là model
    hằng (train chỉ một lớp — ValueError của hàm), ghi warning và bỏ qua.
    """
    if not fold_models:
        return None
    out_dir = reports_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        written = export_shap_importance(fold_models, ticker, reports_dir=reports_dir)
    except ValueError as exc:
        logger.warning("Bỏ qua SHAP h=%d cho %s: %s", horizon, ticker, exc)
        return None
    final = out_dir / f"shap_importance_h{horizon}.csv"
    os.replace(written, final)
    logger.info("Đã ghi %s", final)
    return final


def _plot_backtest(
    frame: pd.DataFrame,
    ticker: str,
    horizon: int,
    chart_result: WalkForwardResult,
    aggregates: Mapping[str, Mapping[str, Any]],
    out_path: Path,
) -> Path:
    """Biểu đồ backtest PNG cho MỘT horizon (reports/{ticker}/backtest_h{h}.png).

    - Trên: giá close (toàn bộ, mờ) + trong vùng walk-forward: giá THỰC tại
      phiên mục tiêu vs dự báo model chính vs persistence — cả ba trên CÙNG
      các dòng evaluation (baseline cùng window, bài học StockPredictor).
    - Dưới: cột skill vs persistence (%) và edge vs base rate (điểm % ×100)
      của cả 4 model cho horizon này.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = chart_result.predictions
    scored = predictions[predictions["target_date"].notna()]

    fig, (ax_price, ax_bars) = plt.subplots(
        2, 1, figsize=(11, 7.5), gridspec_kw={"height_ratios": [2.2, 1.0]}
    )
    # --- Panel trên: giá ---
    context_start = max(0, len(frame) - len(predictions) - 120)
    ax_price.plot(
        frame.index[context_start:],
        frame["close"].iloc[context_start:],
        color="#bbbbbb",
        linewidth=1.0,
        label="close (bối cảnh)",
    )
    if len(scored):
        target_x = pd.to_datetime(scored["target_date"])
        ax_price.plot(
            target_x, scored["actual_price"], color="#333333", linewidth=1.4,
            label="thực tế tại t+h",
        )
        ax_price.plot(
            target_x, scored["persistence_price"], color="#2ca02c", linewidth=1.1,
            linestyle="--", label="persistence (close_t)",
        )
        skill_text = ""
        agg_primary = aggregates.get(PRIMARY_CHART_FAMILY, {})
        skill = agg_primary.get("reg_skill_vs_persistence_pct")
        if skill is not None:
            skill_text = f" — skill LightGBM {skill:+.1f}% MAE"
        ax_price.plot(
            target_x, scored["predicted_price"], color="#1f77b4", linewidth=1.4,
            label=f"dự báo {PRIMARY_CHART_FAMILY}{skill_text}",
        )
    ax_price.set_title(
        f"{ticker} — walk-forward h={horizon} phiên (embargo "
        f"{chart_result.config.embargo}, retrain mỗi {chart_result.config.retrain_every} phiên)"
    )
    ax_price.set_ylabel("Giá (nghìn đồng)")
    ax_price.legend(loc="best", fontsize=8)
    ax_price.grid(alpha=0.3)

    # --- Panel dưới: skill & edge của 4 model ---
    bar_skills: list[float] = []
    bar_edges: list[float] = []
    labels: list[str] = []
    for family in MODEL_FAMILIES:
        agg = aggregates.get(family, {})
        skill_value = agg.get("reg_skill_vs_persistence_pct")
        edge_value = agg.get("cls_edge_vs_base_rate")
        bar_skills.append(float(skill_value) if skill_value is not None else 0.0)
        bar_edges.append(float(edge_value) * 100.0 if edge_value is not None else 0.0)
        labels.append(family)
    x = np.arange(len(labels))
    width = 0.38
    ax_bars.bar(x - width / 2, bar_skills, width, color="#1f77b4", label="skill vs persistence (%)")
    ax_bars.bar(x + width / 2, bar_edges, width, color="#ff7f0e", label="edge vs base rate (điểm %×100)")
    ax_bars.axhline(0.0, color="#555555", linewidth=0.8)
    ax_bars.set_xticks(x, labels)
    for bars in ax_bars.containers:
        ax_bars.bar_label(bars, fmt="%.1f", fontsize=7, padding=1)
    ax_bars.legend(loc="best", fontsize=8)
    ax_bars.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Đã ghi %s", out_path)
    return out_path


def _save_models(
    models_by_key: Mapping[str, Any],
    ticker: str,
    models_dir: Path,
    metadata: Mapping[str, Any],
) -> tuple[Path, dict[str, str]]:
    """Pickle các model đã retrain vào models/{ticker}/ + metadata.json."""
    out_dir = models_dir / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    for name, model in models_by_key.items():
        path = out_dir / f"{name}.pkl"
        with path.open("wb") as handle:
            pickle.dump(model, handle)
        saved[name] = path.name
    meta_path = out_dir / "metadata.json"
    meta_path.write_text(
        json.dumps(_json_safe({**metadata, "files": saved}), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Đã lưu %d model vào %s", len(saved), out_dir)
    return out_dir, saved


# ---------------------------------------------------------------------------
# Pipeline chính
# ---------------------------------------------------------------------------


def run_pipeline(
    ticker: str,
    config: Mapping[str, Any] | None = None,
    *,
    conn: sqlite3.Connection | None = None,
    db_path: Path | str = db.DEFAULT_DB_PATH,
    data_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    models_dir: Path | str | None = None,
    refresh: bool = False,
    max_folds: int | None = None,
    price_provider: Callable[[str], pd.DataFrame] | None = None,
) -> PipelineResult:
    """Chạy trọn quy trình "Huấn luyện & dự báo" cho MỘT mã; trả PipelineResult.

    Tham số:
        ticker: mã HOSE (vd. "TCB") — chuẩn hoá CHỮ IN HOA.
        config: dict đã parse config.yaml; None → ``load_config()``.
        conn: kết nối SQLite có sẵn (test); None → mở ``db_path`` và tự đóng.
        db_path: đường dẫn SQLite khi tự mở (mặc định data/app.db).
        data_dir: thư mục cache dữ liệu (mặc định data/ của dự án).
        reports_dir/models_dir: thư mục artifacts (mặc định reports/, models/).
        refresh: bỏ cache và tải lại toàn bộ lịch sử giá.
        max_folds: giới hạn số fold walk-forward (chạy rút gọn — vd. 4 fold
            đầu trong scripts/run_e2e.py); khung được cắt đúng đủ số phiên.
            None → chạy TOÀN BỘ fold theo config.
        price_provider: hàm ``(ticker) → DataFrame giá`` cho settle (test/offline);
            None → dùng ``data_loader.load_history``.

    Lỗi: ``PipelineError`` (dữ liệu rỗng/cấu hình sai), ``InsufficientHistoryError``
    (không đủ phiên cho một fold) — bề mặt lỗi rõ ràng cho dashboard, không nuốt.
    """
    cfg = dict(config) if config is not None else load_config()
    ticker = str(ticker).strip().upper()
    horizons = _horizons_of(cfg)
    neutral_band = _neutral_band_fraction(cfg)
    start_date = str(cfg.get("data", {}).get("start_date", DEFAULT_START) or DEFAULT_START)
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    reports_dir = Path(reports_dir) if reports_dir is not None else PROJECT_ROOT / "reports"
    models_dir = Path(models_dir) if models_dir is not None else PROJECT_ROOT / "models"
    if max_folds is not None and int(max_folds) < 1:
        raise PipelineError(f"max_folds phải >= 1, nhận {max_folds}")
    max_folds = int(max_folds) if max_folds is not None else None

    if price_provider is None:
        provider = lambda tk: load_history(tk, data_dir=data_dir)  # noqa: E731
    else:
        provider = price_provider

    own_conn = conn is None
    connection = conn if conn is not None else db.connect(db_path)
    try:
        db.init_db(connection)

        # --- Bước 1: settle trước (dữ liệu đã có trong cache) ---
        settled = db.settle_predictions(connection, provider, neutral_band=neutral_band)

        # --- Bước 2: tải/cập nhật dữ liệu giá ---
        ohlcv = load_history(ticker, start=start_date, refresh=refresh, data_dir=data_dir)
        if ohlcv is None or ohlcv.empty:
            raise PipelineError(
                f"Không lấy được dữ liệu giá cho {ticker} (nguồn lỗi hoặc mã không "
                "tồn tại) — kiểm tra kết nối/ticker rồi thử lại."
            )
        logger.info(
            "run_pipeline(%s): %d phiên %s → %s",
            ticker, len(ohlcv), ohlcv.index.min().date(), ohlcv.index.max().date(),
        )

        # --- Bước 3: feature + target ---
        groups = feature_groups(cfg)
        index_frame: pd.DataFrame | None = None
        if groups.get("market_context", False):
            index_frame = load_vnindex(start=start_date, refresh=refresh, data_dir=data_dir)
            if index_frame.empty:
                logger.warning(
                    "run_pipeline(%s): VN-Index không tải được — feature bối cảnh "
                    "thành NaN, pipeline tiếp tục.",
                    ticker,
                )
        features = assemble_features(ohlcv, cfg, index_frame=index_frame)
        targets = build_targets(ohlcv, horizons=horizons, neutral_band_pct=neutral_band * 100.0)
        frame = pd.concat([ohlcv[["close"]], features, targets], axis=1)
        frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index(kind="stable")
        min_valid_ratio = cfg.get("features", {}).get("min_valid_ratio", 0.5)
        frame, dropped_features = _drop_sparse_features(frame, min_valid_ratio)

        # --- Rút gọn số fold (e2e) — cắt đủ phiên cho đúng max_folds fold đầu ---
        wf_cfg_h0 = WalkForwardConfig.from_mapping(cfg, horizon=horizons[0])
        if max_folds is not None:
            limit = _max_folds_limit(wf_cfg_h0, max_folds)
            if len(frame) > limit:
                logger.info(
                    "run_pipeline(%s): rút gọn khung %d → %d phiên (max_folds=%d)",
                    ticker, len(frame), limit, max_folds,
                )
                frame = frame.iloc[:limit]

        # --- Settle vòng 2 sau khi chỉnh target_date theo phiên thực ---
        repointed = align_target_dates(connection, ticker, ohlcv)
        if repointed:
            settled += db.settle_predictions(connection, provider, neutral_band=neutral_band)

        # --- Bước 4: walk-forward cho 4 model × từng horizon ---
        last_session = frame.index[-1]
        metrics_by_horizon: dict[str, dict[str, dict[str, Any]]] = {}
        chart_results: dict[str, WalkForwardResult] = {}
        csv_rows: list[dict[str, Any]] = []
        shap_paths: list[Path] = []

        for horizon in horizons:
            wf_cfg = WalkForwardConfig.from_mapping(cfg, horizon=horizon)
            gap = max(wf_cfg.embargo, horizon)
            family_aggregates: dict[str, dict[str, Any]] = {}
            for family in MODEL_FAMILIES:
                try:
                    aggregate, chart_result, fold_models = _run_family(
                        frame, family, horizon, cfg, wf_cfg, gap
                    )
                except (InsufficientHistoryError, PipelineError):
                    # Lỗi hình học/cấu hình (không đủ phiên cho fold, config sai)
                    # đụng MỌI model như nhau → surface, không nuốt.
                    raise
                except Exception as exc:  # noqa: BLE001 — một model lỗi KHÔNG được giết
                    # pipeline: ghi lỗi RÕ vào metrics (không nuốt im lặng), chạy
                    # tiếp các model còn lại. Model lỗi không có dự báo phiên kế.
                    logger.error(
                        "run_pipeline(%s): walk-forward %s h=%d THẤT BẠI: %s: %s",
                        ticker, family, horizon, type(exc).__name__, exc,
                    )
                    family_aggregates[family] = {"error": f"{type(exc).__name__}: {exc}"}
                    csv_rows.append(
                        {"horizon": horizon, "model": family, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    continue
                family_aggregates[family] = aggregate
                csv_rows.append({"horizon": horizon, "model": family, **aggregate})
                if family == PRIMARY_CHART_FAMILY:
                    chart_results[str(horizon)] = chart_result
                    shap_path = _export_shap(fold_models, ticker, horizon, reports_dir)
                    if shap_path is not None:
                        shap_paths.append(shap_path)
                logger.info(
                    "run_pipeline(%s) h=%d %s: MAE %.6f vs persistence %.6f "
                    "(skill %s), acc %s vs base %s (edge %s)",
                    ticker, horizon, family,
                    aggregate.get("reg_mae") if aggregate.get("reg_mae") is not None else float("nan"),
                    aggregate.get("reg_persistence_mae")
                    if aggregate.get("reg_persistence_mae") is not None
                    else float("nan"),
                    f"{aggregate['reg_skill_vs_persistence_pct']:+.2f}%"
                    if aggregate.get("reg_skill_vs_persistence_pct") is not None else "n/a",
                    f"{aggregate['cls_accuracy']:.3f}"
                    if aggregate.get("cls_accuracy") is not None else "n/a",
                    f"{aggregate['cls_base_rate_majority']:.3f}"
                    if aggregate.get("cls_base_rate_majority") is not None else "n/a",
                    f"{aggregate['cls_edge_vs_base_rate']:+.3f}"
                    if aggregate.get("cls_edge_vs_base_rate") is not None else "n/a",
                )
            metrics_by_horizon[str(horizon)] = family_aggregates

        # --- Bước 5: retrain full + dự báo phiên kế tiếp cho từng model/horizon ---
        target_date_estimates = _estimate_target_dates(last_session, horizons)
        forecasts: list[dict[str, Any]] = []
        models_to_save: dict[str, Any] = {}
        naive_saved = False
        for horizon in horizons:
            gap = max(wf_cfg_h0.embargo, horizon)
            close_t = float(frame["close"].iloc[-1])
            for family in MODEL_FAMILIES:
                # Model vừa lỗi ở walk-forward (chỉ có khoá "error") → không có
                # dự báo phiên kế; đã được báo rõ trong metrics.
                if "error" in metrics_by_horizon[str(horizon)].get(family, {}):
                    continue
                try:
                    price, prob, models = _full_retrain_forecast(
                        frame, family, horizon, cfg, gap
                    )
                except Exception as exc:  # noqa: BLE001 — retrain lỗi cũng KHÔNG giết
                    # pipeline: bỏ dự báo của model này, báo rõ qua metrics_json.
                    logger.error(
                        "run_pipeline(%s): retrain full %s h=%d THẤT BẠI: %s: %s",
                        ticker, family, horizon, type(exc).__name__, exc,
                    )
                    metrics_by_horizon[str(horizon)][family]["error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                for model_name, model in models.items():
                    if model_name == "naive":
                        if naive_saved:
                            continue  # naive stateless — lưu đúng một file
                        naive_saved = True
                    models_to_save[model_name] = model
                direction = _direction_from_price(price, close_t, neutral_band)
                prob_value = prob if prob is not None and math.isfinite(prob) else None
                forecast_row = {
                    "model": family,
                    "horizon": horizon,
                    "target_date": target_date_estimates[horizon],
                    "predicted_price": price if price is not None and math.isfinite(price) else None,
                    "predicted_direction": direction,
                    "probability_up": prob_value,
                    "close_t": close_t,
                }
                forecasts.append(forecast_row)

        # --- Bước 6: ghi runs/predictions vào DB ---
        feature_columns = _feature_columns_of(models_to_save)
        metrics_json = {
            "ticker": ticker,
            "horizons": list(horizons),
            "n_sessions": int(len(frame)),
            "neutral_band_pct": float(cfg.get("neutral_band_pct", 0.3)),
            "walk_forward": {
                "initial_train_size": wf_cfg_h0.initial_train_size,
                "retrain_every": wf_cfg_h0.retrain_every,
                "embargo": wf_cfg_h0.embargo,
                "max_folds": max_folds,
            },
            "per_horizon": metrics_by_horizon,
            "forecasts": forecasts,
            "settled_this_run": int(settled),
            "repointed_this_run": int(repointed),
            "dropped_sparse_features": dropped_features,
        }
        run_id = db.record_run(
            connection,
            ticker,
            data_until=last_session.strftime("%Y-%m-%d"),
            metrics=_json_safe(metrics_json),
            model_dir=str(models_dir / ticker),
        )
        for row in forecasts:
            db.record_prediction(
                connection,
                run_id,
                ticker,
                row["model"],
                row["target_date"],
                row["horizon"],
                predicted_price=row["predicted_price"],
                predicted_direction=row["predicted_direction"],
                probability_up=row["probability_up"],
            )

        # --- Bước 7: lưu model + xuất artifacts ---
        model_dir, saved_files = _save_models(
            models_to_save,
            ticker,
            models_dir,
            metadata={
                "ticker": ticker,
                "run_id": run_id,
                "trained_at": pd.Timestamp.now().isoformat(timespec="seconds"),
                "data_until": last_session.strftime("%Y-%m-%d"),
                "n_sessions": int(len(frame)),
                "horizons": list(horizons),
                "feature_columns": feature_columns,
                "note": "Model retrain trên TOÀN BỘ khung — dùng cho dự báo phiên kế tiếp. "
                "Walk-forward KHÔNG lưu model fold (mỗi fold fit lại từ đầu).",
            },
        )
        metrics_csv = _export_metrics_csv(csv_rows, ticker, reports_dir)
        charts: list[Path] = []
        for horizon in horizons:
            chart_result = chart_results.get(str(horizon))
            if chart_result is None:
                continue
            charts.append(
                _plot_backtest(
                    frame,
                    ticker,
                    horizon,
                    chart_result,
                    metrics_by_horizon[str(horizon)],
                    reports_dir / ticker / f"backtest_h{horizon}.png",
                )
            )

        logger.info(
            "run_pipeline(%s): xong — run #%d, %d phiên (đến %s), settle %d, "
            "re-point %d, artifacts: %s",
            ticker, run_id, len(frame), last_session.date(), settled, repointed,
            metrics_csv,
        )
        return PipelineResult(
            ticker=ticker,
            run_id=int(run_id),
            n_sessions=int(len(frame)),
            data_until=last_session.strftime("%Y-%m-%d"),
            horizons=horizons,
            settled=int(settled),
            repointed=int(repointed),
            metrics=metrics_by_horizon,
            forecasts=tuple(forecasts),
            artifacts={
                "metrics_csv": metrics_csv,
                "charts": charts,
                "shap": shap_paths,
                "model_dir": model_dir,
                "model_files": saved_files,
            },
        )
    finally:
        if own_conn:
            connection.close()


def _feature_columns_of(models: Mapping[str, Any]) -> list[str]:
    """Danh sách feature chuẩn từ model retrain (ưu tiên lightgbm_reg → lstm)."""
    for name in ("lightgbm_reg", "lstm", "arima", "naive"):
        model = models.get(name)
        columns = getattr(model, "feature_columns_", None)
        if columns:
            return [str(column) for column in columns]
    return []


# ---------------------------------------------------------------------------
# CLI — python -m src.pipeline MÃ [MÃ...] [--watchlist] [--config PATH] ...
# ---------------------------------------------------------------------------


def _resolve_tickers(args: argparse.Namespace, cfg: Mapping[str, Any]) -> list[str]:
    """Danh sách mã cần chạy: argv → --watchlist → mã mặc định của config."""
    from .watchlist import Watchlist, WatchlistError

    if args.tickers:
        return [str(tk).strip().upper() for tk in args.tickers]
    try:
        watchlist_tickers = Watchlist().list_tickers()
    except WatchlistError as exc:
        raise PipelineError(f"Watchlist không đọc được: {exc}") from exc
    if args.watchlist:
        if not watchlist_tickers:
            raise PipelineError(
                "Watchlist trống — thêm mã qua dashboard (tab Watchlist) hoặc "
                "truyền mã trực tiếp: python -m src.pipeline TCB."
            )
        return watchlist_tickers
    default_ticker = str(cfg.get("default_ticker", "TCB")).strip().upper()
    return [default_ticker]


def main(argv: list[str] | None = None) -> int:
    """CLI huấn luyện: 1 mã, nhiều mã, hoặc cả danh mục watchlist."""
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description=(
            "Huấn luyện & dự báo cho mã HOSE (walk-forward + retrain full + ghi "
            "DB + xuất reports). Không truyền mã → chạy cả watchlist; watchlist "
            "trống → chạy mã mặc định trong config.yaml."
        ),
    )
    parser.add_argument("tickers", nargs="*", help="Mã cổ phiếu HOSE (vd. TCB VCB)")
    parser.add_argument(
        "--watchlist", action="store_true",
        help="Chạy toàn bộ mã trong data/watchlist.json (bỏ qua mã truyền trên argv)",
    )
    parser.add_argument("--config", default=None, help="Đường dẫn config.yaml khác mặc định")
    parser.add_argument("--refresh", action="store_true", help="Tải lại toàn bộ lịch sử giá (bỏ cache)")
    parser.add_argument(
        "--max-folds", type=int, default=None,
        help="Giới hạn số fold walk-forward (chạy nhanh — dùng cho kiểm thử)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    )
    try:
        cfg = load_config(args.config)
        if args.watchlist:
            args.tickers = []
        tickers = _resolve_tickers(args, cfg)
    except (PipelineError, ValueError) as exc:
        logger.error("%s", exc)
        return 2

    failures: list[str] = []
    for ticker in tickers:
        logger.info("=== Huấn luyện & dự báo: %s ===", ticker)
        try:
            result = run_pipeline(
                ticker,
                cfg,
                refresh=args.refresh,
                max_folds=args.max_folds,
            )
        except (PipelineError, InsufficientHistoryError, ValueError) as exc:
            logger.error("[FAIL] %s: %s", ticker, exc)
            failures.append(ticker)
            continue
        _print_summary(result)

    if failures:
        logger.error("Hoàn thành với LỖI ở %d/%d mã: %s", len(failures), len(tickers), failures)
        return 1
    return 0


def _print_summary(result: PipelineResult) -> None:
    """In tổng kết một mã ra console (tiếng Việt, ngắn gọn)."""
    print(
        f"[{result.ticker}] run #{result.run_id} | {result.n_sessions} phiên "
        f"(đến {result.data_until}) | settle {result.settled}, re-point {result.repointed}"
    )
    for horizon in result.horizons:
        for family, aggregate in result.metrics[str(horizon)].items():
            if "error" in aggregate:
                print(f"    h={horizon} {family:<9} LỖI: {aggregate['error']}")
                continue
            skill = aggregate.get("reg_skill_vs_persistence_pct")
            edge = aggregate.get("cls_edge_vs_base_rate")
            mae = aggregate.get("reg_mae")
            acc = aggregate.get("cls_accuracy")
            print(
                f"    h={horizon} {family:<9} MAE {mae if mae is not None else float('nan'):.5f}"
                f" | skill {('%+.1f%%' % skill) if skill is not None else 'n/a':>7}"
                f" | acc {acc if acc is not None else float('nan'):.3f}"
                f" | edge {('%+.1f điểm' % (edge * 100)) if edge is not None else 'n/a'}"
            )
    for row in result.forecasts:
        print(
            f"    DỰ BÁO h={row['horizon']} {row['model']:<9} "
            f"giá {row['predicted_price'] if row['predicted_price'] is not None else 'n/a'} "
            f"| hướng {row['predicted_direction'] or 'n/a'} "
            f"| P(LỚN) {row['probability_up'] if row['probability_up'] is not None else 'n/a'} "
            f"| phiên mục tiêu {row['target_date']}"
        )
    print(f"    Artifacts: {result.artifacts.get('metrics_csv')}")


if __name__ == "__main__":
    sys.exit(main())
