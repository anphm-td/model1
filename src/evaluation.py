"""Đánh giá — walk-forward validation expanding window và so sánh baseline.

Thiết kế (khớp config.yaml: walk_forward — Common 750/21/embargo 5):
- Cửa sổ train MỞ RỘNG (expanding): fold k train trên toàn bộ phiên từ đầu đến
  ``test_start - embargo``, test trên block ``retrain_every`` phiên kế tiếp.
  Test block đầu tiên bắt đầu sau ``initial_train_size + embargo`` phiên.
- PURGE + EMBARGO chống leakage của nhãn (bài học StockIntel: nhãn tại t chứa
  giá đến t+h, nên train phải cách test ít nhất ``horizon`` phiên, không chỉ
  ``embargo``): biên train thực tế là
      ``train_end = min(test_start - embargo, test_start - horizon)``
  — khi ``embargo >= horizon`` (mặc định 5/5) purge rỗng; khi cấu hình đặt
  embargo nhỏ hơn horizon, engine TỰ NỚI khoảng cách lên ``horizon`` và ghi
  số bị cắt vào ``FoldLayout.purged_rows`` (hiện ra ``fold_summary()``, không
  âm thầm). Bất biến được bảo vệ: mọi nhãn trong train chỉ dùng giá < test_start.
- RETRAIN: mỗi fold dùng một model MỚI từ ``model_factory`` — không trạng thái
  nào (kể cả scaler) mang sang fold sau. Scaler/transform là trách nhiệm của
  model, nhưng engine bảo đảm ranh giới dữ liệu: model chỉ nhận đúng các dòng
  train (và val nội bộ) của fold, và ``predict()`` nhận test_df ĐÃ BỎ cột
  target — model không thể nhìn nhãn test kể cả khi muốn.
- KHÔNG SHUFFLE: mọi khung đưa vào model giữ nguyên thứ tự thời gian.

Interface chung cho model (hợp đồng của engine):
- ``fit(train_df, val_df)``: ``train_df``/``val_df`` là khung dòng theo phiên
  (feature + ``close`` + cột ``target_*``); ``val_df`` là phần đuôi của vùng
  train (dùng cho early stopping), ``None`` khi ``val_size = 0``.
- ``predict(test_df)``: khung dòng test KHÔNG có cột target; trả
  ``ModelPrediction`` với ``predicted_price`` (dự báo giá close tại t+h) hoặc
  ``predicted_return`` (log-return t→t+h — engine tự quy đổi sang giá bằng
  ``close_t * exp(ret)``) và/hoặc ``probability_up`` (xác suất LỚN, [0,1]).
  Model chỉ cần điền phần mình làm được; metrics phía không điền sẽ là None.

Baselines (bắt buộc theo config.yaml), tính trên CÙNG các dòng evaluation:
- Persistence (giá): dự báo ``close_{t+h} = close_t``; báo cáo
  ``reg_skill_vs_persistence_pct = 100 * (1 - MAE_model / MAE_persistence)``
  (dương = model thắng persistence; 0 = hoà; âm = thua).
- Majority-class (hướng): luôn chọn lớp đa số của TRAIN (không nhìn test);
  ``cls_base_rate_majority`` là accuracy của baseline đó trên test,
  ``cls_edge_vs_base_rate = accuracy - base_rate`` (điểm phần trăm).
  Baseline là QUYẾT ĐỊNH THEO FOLD ("lớp đa số của train fold này"); khi gộp
  metrics nhiều fold, engine ghép các quyết định fold-honest theo đúng dòng
  test (cột ``baseline_up``) — TUYỆT ĐỐI không gộp nhãn train của MỌI fold
  thành một cửa sổ duy nhất: cửa sổ expanding của fold sau CHỨA test block
  của fold trước, nên nhãn evaluation lọt vào "train" của baseline gộp
  (leakage — baseline có lợi thế thông tin so với model).
  Metrics hướng dùng khung nhị phân "LỚN vs không-LỚN" (nhãn ``target_dir ==
  +1``) vì interface chung của model là ``probability_up``; các dòng neutral
  của labeling được tính vào phía "không-LỚN".

Chống leakage ưu tiên số 1:
- Feature tại t chỉ dùng dữ liệu <= t (bảo đảm bởi src/features, test riêng).
- Target tại t dùng giá đến t+h — engine loại các dòng train mà nhãn chạm vào
  vùng test (purge ở trên) và loại các dòng test CHƯA có kết quả thực tế
  (h phiên cuối khung) khỏi metrics; các dòng đó vẫn nằm trong
  ``predictions`` với ``actual_price`` NaN (dự báo "live", chưa chấm điểm).
- Baseline tính trên cùng window với model — không so MAE của model trên một
  tập dòng với persistence trên tập khác.
"""
from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from .labeling import UP

logger = logging.getLogger(__name__)

#: Tiền tố cột target bị LOẠI khỏi khung đưa vào ``predict()`` (guard chống
#: model đọc nhãn test). Cùng quy ước tiền tố với assemble.NON_FEATURE_PREFIXES.
TARGET_COLUMN_PREFIXES: tuple[str, ...] = ("target_", "fwd_", "label_")

#: Xác suất >= ngưỡng này được coi là dự báo "LỚN" (quy ước nhị phân hoá).
UP_PROBABILITY_THRESHOLD: float = 0.5


class InsufficientHistoryError(ValueError):
    """Khung dữ liệu không đủ phiên để tạo được một fold walk-forward."""


# ---------------------------------------------------------------------------
# Interface chung cho model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelPrediction:
    """Đầu ra bắt buộc của ``WalkForwardModel.predict`` cho một khung test.

    Mỗi trường là mảng 1-D cùng độ dài với khung test, hoặc None nếu model
    không làm phía đó (NaN trong mảng = model bỏ qua dòng đó, dòng bị loại
    khỏi metrics phía tương ứng):
    - ``predicted_price``: dự báo GIÁ close tại t+h (> 0).
    - ``predicted_return``: log-return t→t+h; engine quy đổi sang giá
      ``close_t * exp(return)`` khi ``predicted_price`` không cung cấp.
    - ``probability_up``: xác suất hướng LỚN trong [0, 1].
    """

    predicted_price: np.ndarray | None = None
    predicted_return: np.ndarray | None = None
    probability_up: np.ndarray | None = None


class WalkForwardModel(Protocol):
    """Hợp đồng model mà engine walk-forward điều phối (regression lẫn classification)."""

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> Any:
        """Học trên các dòng train của MỘT fold (``val_df`` nội bộ, có thể None).

        Model PHẢI fit mọi trạng thái học (kể cả scaler/bộ chuẩn hoá) chỉ từ
        ``train_df``/``val_df`` này — không dùng thống kê toàn chuỗi, không
        giữ trạng thái từ fold trước. Không được sửa hai khung truyền vào.
        """

    def predict(self, test_df: pd.DataFrame) -> ModelPrediction:
        """Dự báo cho các dòng test của fold (KHÔNG có cột target).

        Trả ``ModelPrediction`` (xem docstring dataclass) cùng độ dài với
        ``test_df``; thứ tự dòng giữ nguyên.
        """


# ---------------------------------------------------------------------------
# Cấu hình và hình học fold
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WalkForwardConfig:
    """Tham số walk-forward (mặc định khớp config.yaml: 750/21/embargo 5)."""

    initial_train_size: int = 750
    retrain_every: int = 21
    embargo: int = 5
    horizon: int = 1
    #: Số phiên cuối vùng train tách làm validation nội bộ cho ``fit`` (0 = không có).
    val_size: int = 0

    def __post_init__(self) -> None:
        if int(self.initial_train_size) < 1:
            raise ValueError(f"initial_train_size phải >= 1, nhận {self.initial_train_size}")
        if int(self.retrain_every) < 1:
            raise ValueError(f"retrain_every phải >= 1, nhận {self.retrain_every}")
        if int(self.embargo) < 0:
            raise ValueError(f"embargo phải >= 0, nhận {self.embargo}")
        if int(self.horizon) < 1:
            raise ValueError(f"horizon phải >= 1, nhận {self.horizon}")
        if not 0 <= int(self.val_size) < int(self.initial_train_size):
            raise ValueError(
                f"val_size phải trong [0, initial_train_size), nhận {self.val_size} "
                f"(initial_train_size={self.initial_train_size})"
            )

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any], horizon: int = 1) -> "WalkForwardConfig":
        """Đọc mục ``walk_forward`` của config.yaml (engine chạy MỘT horizon/lần).

        ``horizons`` trong yaml là danh sách [1, 5] — tầng pipeline lặp từng
        horizon và gọi ``walk_forward`` riêng cho mỗi horizon; ở đây nhận một
        ``horizon`` duy nhất. ``shuffle: true`` trong config bị TỪ CHỐI: engine
        không có chế độ shuffle (dữ liệu chuỗi thời gian).
        """
        section = config.get("walk_forward", {}) if isinstance(config, Mapping) else {}
        if not isinstance(section, Mapping):
            raise ValueError(f"Mục 'walk_forward' trong config phải là mapping, nhận {type(section).__name__}")
        if section.get("shuffle", False):
            raise ValueError(
                "walk_forward.shuffle=true bị từ chối — walk-forward tuyệt đối không "
                "shuffle dữ liệu chuỗi thời gian (đặt shuffle: false)."
            )
        return cls(
            initial_train_size=int(section.get("initial_train_size", 750)),
            retrain_every=int(section.get("retrain_every", 21)),
            embargo=int(section.get("embargo", 5)),
            horizon=int(horizon),
            val_size=int(section.get("val_size", 0)),
        )


@dataclass(frozen=True)
class FoldLayout:
    """Ranh giới một fold theo VỊ TRÍ dòng (0-based, nửa-khoảng [start, end)).

    Bất biến bảo vệ bởi ``make_folds``:
    - ``train_end <= test_start - embargo`` (khoảng cách embargo theo config);
    - ``train_end <= test_start - horizon`` (purge: mọi nhãn train chỉ dùng
      giá tại vị trí < test_start);
    - ``purged_rows = (test_start - embargo) - train_end`` = số dòng bị cắt
      thêm do purge (0 khi embargo >= horizon).
    """

    fold_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    purged_rows: int
    embargo: int

    @property
    def gap_rows(self) -> int:
        """Số dòng giữa biên train và test (embargo gap thực tế >= config.embargo)."""
        return self.test_start - self.train_end

    @property
    def n_train(self) -> int:
        return self.train_end - self.train_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


def make_folds(n_sessions: int, config: WalkForwardConfig) -> list[FoldLayout]:
    """Danh sách fold expanding-window cho ``n_sessions`` phiên.

    Test block k bắt đầu tại ``initial_train_size + embargo + k*retrain_every``
    và dài ``retrain_every`` phiên (block cuối có thể ngắn hơn); lặp tới hết
    khung. Không đủ phiên cho một test block đầu tiên → ``InsufficientHistoryError``.
    """
    n_sessions = int(n_sessions)
    minimum = config.initial_train_size + config.embargo + 1
    if n_sessions < minimum:
        raise InsufficientHistoryError(
            f"Không đủ lịch sử cho walk-forward: cần >= {minimum} phiên "
            f"(initial_train_size {config.initial_train_size} + embargo "
            f"{config.embargo} + 1 test row), nhận {n_sessions} phiên."
        )
    folds: list[FoldLayout] = []
    fold_id = 0
    while True:
        test_start = config.initial_train_size + config.embargo + fold_id * config.retrain_every
        if test_start >= n_sessions:
            break
        test_end = min(test_start + config.retrain_every, n_sessions)
        raw_train_end = test_start - config.embargo
        train_end = min(raw_train_end, test_start - config.horizon)
        folds.append(
            FoldLayout(
                fold_id=fold_id,
                train_start=0,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                purged_rows=raw_train_end - train_end,
                embargo=config.embargo,
            )
        )
        fold_id += 1
    return folds


# ---------------------------------------------------------------------------
# Metrics (tính trên mảng numpy — tái dùng cho từng fold và gộp)
# ---------------------------------------------------------------------------


def regression_metrics(
    actual_price: np.ndarray,
    predicted_price: np.ndarray,
    persistence_price: np.ndarray,
) -> dict[str, float | int | None]:
    """MAE/RMSE/MAPE + skill vs persistence, trên các dòng CÙNG window.

    Dòng bị bỏ: actual hoặc predicted là NaN. ``skill_vs_persistence_pct =
    100*(1 - mae/persistence_mae)`` (None khi persistence_mae = 0 hoặc không
    còn dòng). MAPE theo % (giá > 0 được bảo đảm bởi validate của engine).
    """
    actual = np.asarray(actual_price, dtype="float64")
    predicted = np.asarray(predicted_price, dtype="float64")
    persistence = np.asarray(persistence_price, dtype="float64")
    valid = ~(np.isnan(actual) | np.isnan(predicted) | np.isnan(persistence))
    n_rows = int(valid.sum())
    if n_rows == 0:
        return {
            "reg_n_rows": 0,
            "reg_mae": None,
            "reg_rmse": None,
            "reg_mape_pct": None,
            "reg_persistence_mae": None,
            "reg_persistence_rmse": None,
            "reg_skill_vs_persistence_pct": None,
        }
    err = predicted[valid] - actual[valid]
    persistence_err = persistence[valid] - actual[valid]
    mae = float(np.mean(np.abs(err)))
    persistence_mae = float(np.mean(np.abs(persistence_err)))
    skill: float | None = (
        100.0 * (1.0 - mae / persistence_mae) if persistence_mae > 0.0 else None
    )
    return {
        "reg_n_rows": n_rows,
        "reg_mae": mae,
        "reg_rmse": float(np.sqrt(np.mean(err**2))),
        "reg_mape_pct": float(np.mean(np.abs(err) / actual[valid]) * 100.0),
        "reg_persistence_mae": persistence_mae,
        "reg_persistence_rmse": float(np.sqrt(np.mean(persistence_err**2))),
        "reg_skill_vs_persistence_pct": skill,
    }


def majority_class_up(train_is_up: np.ndarray) -> bool | None:
    """Lớp đa số "LỚN vs không-LỚN" của MỘT cửa sổ train; None khi không có nhãn hợp lệ.

    Quy ước hoà 50/50 → chọn LỚN. Bất biến chống leakage: hàm CHỈ nhìn đúng
    các nhãn được truyền vào — caller phải bảo đảm mảng chỉ chứa nhãn train
    của CÙNG MỘT fold. Không bao giờ gộp nhãn train của nhiều fold rồi suy lớp
    đa số: cửa sổ expanding của fold sau chứa test block của fold trước, nên
    pool gộp chứa nhãn của chính các dòng evaluation (baseline có lợi thế
    thông tin — xem ``walk_forward`` và cột ``baseline_up``).
    """
    train = np.asarray(train_is_up, dtype="float64").ravel()
    train = train[~np.isnan(train)]
    if train.size == 0:
        return None
    return bool(np.mean(train) >= 0.5)  # hoà 50/50 → chọn LỚN


def classification_metrics(
    actual_is_up: np.ndarray,
    probability_up: np.ndarray,
    train_is_up: np.ndarray | None = None,
    *,
    baseline_is_up: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    """Accuracy/F1/AUC + edge vs base rate majority-class, khung "LỚN vs không-LỚN".

    - Dòng bị bỏ: actual (nhãn) hoặc probability là NaN.
    - Dự báo LỚN khi ``probability_up >= UP_PROBABILITY_THRESHOLD`` (0.5).
    - ``f1_up``: F1 của lớp LỚN (0.0 khi model không dự báo LỚN dòng nào).
    - ``roc_auc``: None khi còn < 2 dòng hoặc actual chỉ có một lớp.
    - ``base_rate_majority``: accuracy của baseline "luôn chọn lớp đa số của
      TRAIN" trên các dòng evaluation (tần suất lớp đa số trong actual);
      None khi baseline không xác định.
    - ``edge_vs_base_rate = accuracy - base_rate`` (điểm phần trăm; None khi
      base rate không xác định).

    Baseline xác định qua ĐÚNG MỘT trong hai tham số (truyền cả hai là lỗi):
    - ``train_is_up``: nhãn train của MỘT cửa sổ — hàm tự suy lớp đa số
      (``majority_class_up``). Dùng cho metrics một fold.
    - ``baseline_is_up``: dự báo baseline THEO DÒNG (1.0 = LỚN, 0.0 = không),
      đã ghép từ quyết định fold-honest của engine khi đánh giá gộp nhiều
      fold — vì pool gộp không có "một cửa sổ train" trung thực nào. Dòng
      baseline NaN bị loại khỏi base rate (accuracy vẫn chấm trên dòng đó).
    """
    if train_is_up is not None and baseline_is_up is not None:
        raise ValueError(
            "classification_metrics chỉ nhận MỘT trong hai: train_is_up (nhãn train của "
            "MỘT cửa sổ — tự suy lớp đa số) hoặc baseline_is_up (dự báo baseline theo "
            "dòng, ghép fold-honest cho đánh giá gộp) — truyền cả hai là mơ hồ."
        )
    actual = np.asarray(actual_is_up, dtype="float64")
    prob = np.asarray(probability_up, dtype="float64")
    valid = ~(np.isnan(actual) | np.isnan(prob))
    n_rows = int(valid.sum())
    base: dict[str, float | int | None] = {
        "cls_n_rows": n_rows,
        "cls_accuracy": None,
        "cls_f1_up": None,
        "cls_roc_auc": None,
        "cls_base_rate_majority": None,
        "cls_edge_vs_base_rate": None,
        "cls_up_rate": None,
    }
    if n_rows == 0:
        return base
    y_true = actual[valid].astype(int)
    prob_valid = prob[valid]
    y_pred = (prob_valid >= UP_PROBABILITY_THRESHOLD).astype(int)

    accuracy = float(np.mean(y_pred == y_true))
    base["cls_accuracy"] = accuracy
    base["cls_f1_up"] = float(f1_score(y_true, y_pred, zero_division=0))
    if len(np.unique(y_true)) > 1:
        base["cls_roc_auc"] = float(roc_auc_score(y_true, prob_valid))
    base["cls_up_rate"] = float(np.mean(y_true))

    if baseline_is_up is not None:
        baseline = np.asarray(baseline_is_up, dtype="float64").ravel()
        if baseline.shape[0] != actual.shape[0]:
            raise ValueError(
                f"baseline_is_up dài {baseline.shape[0]}, khác số dòng actual "
                f"{actual.shape[0]} — baseline phải theo đúng từng dòng evaluation."
            )
        valid_base = valid & ~np.isnan(baseline)
        if valid_base.any():
            base_rate = float(
                np.mean(actual[valid_base].astype(int) == baseline[valid_base].astype(int))
            )
            base["cls_base_rate_majority"] = base_rate
            base["cls_edge_vs_base_rate"] = accuracy - base_rate
        return base
    if train_is_up is not None:
        majority_up = majority_class_up(train_is_up)
        if majority_up is not None:
            base_rate = float(np.mean(y_true == int(majority_up)))
            base["cls_base_rate_majority"] = base_rate
            base["cls_edge_vs_base_rate"] = accuracy - base_rate
    return base


METRIC_KEYS: tuple[str, ...] = (
    "reg_n_rows",
    "reg_mae",
    "reg_rmse",
    "reg_mape_pct",
    "reg_persistence_mae",
    "reg_persistence_rmse",
    "reg_skill_vs_persistence_pct",
    "cls_n_rows",
    "cls_accuracy",
    "cls_f1_up",
    "cls_roc_auc",
    "cls_base_rate_majority",
    "cls_edge_vs_base_rate",
    "cls_up_rate",
)


# ---------------------------------------------------------------------------
# Kết quả
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FoldResult:
    """Kết quả một fold: hình học + metrics trên dòng test của fold đó."""

    layout: FoldLayout
    train_start_date: pd.Timestamp
    train_end_date: pd.Timestamp
    test_start_date: pd.Timestamp
    test_end_date: pd.Timestamp
    metrics: dict[str, float | int | None] = field(default_factory=dict)

    @property
    def fold_id(self) -> int:
        return self.layout.fold_id


@dataclass(frozen=True)
class WalkForwardResult:
    """Kết quả walk-forward: từng fold, metrics gộp và bảng dự báo từng dòng."""

    config: WalkForwardConfig
    folds: list[FoldResult]
    #: Mỗi dòng test đúng một lần (fold test block rời nhau): index = phiên t
    #: (thời điểm DỰ BÁO), cột: fold_id, target_date (phiên t+h, NaT nếu chưa
    #: có kết quả), close_t, predicted_price, persistence_price, actual_price,
    #: prob_up, predicted_up, actual_is_up, baseline_up (dự báo majority-class
    #: baseline — lớp đa số của train FOLD của dòng đó, không nhìn test).
    predictions: pd.DataFrame
    aggregate: dict[str, float | int | None]

    def fold_summary(self) -> pd.DataFrame:
        """Bảng một dòng/fold (hình học + metrics) cho dashboard/báo cáo."""
        rows: list[dict[str, Any]] = []
        for fold in self.folds:
            row: dict[str, Any] = {
                "fold_id": fold.fold_id,
                "train_start": fold.train_start_date,
                "train_end": fold.train_end_date,
                "test_start": fold.test_start_date,
                "test_end": fold.test_end_date,
                "n_train": fold.layout.n_train,
                "n_test": fold.layout.n_test,
                "embargo": fold.layout.embargo,
                "purged_rows": fold.layout.purged_rows,
            }
            row.update(fold.metrics)
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def _validate_frame(frame: pd.DataFrame, config: WalkForwardConfig, close_column: str) -> None:
    """Kiểm tra khung đầu vào của engine; lỗi thì raise với thông điệp rõ."""
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"frame phải là pandas DataFrame, nhận {type(frame).__name__}")
    if frame.empty:
        raise ValueError("frame rỗng — không có gì để walk-forward.")
    index = frame.index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError(
            f"frame.index phải là DatetimeIndex (phiên giao dịch), nhận {type(index).__name__}"
        )
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError(
            "frame.index phải tăng dần và không trùng ngày (không shuffle, không trùng phiên) — "
            "hãy sort/dedup ở tầng dữ liệu trước khi gọi walk_forward."
        )
    if close_column not in frame.columns:
        raise ValueError(f"frame thiếu cột giá '{close_column}' (cần cho persistence và quy đổi giá).")
    close = pd.to_numeric(frame[close_column], errors="coerce")
    if close.isna().any():
        raise ValueError(
            f"cột '{close_column}' chứa NaN — tầng dữ liệu phải cho ra chuỗi giá đầy đủ "
            "(không ffill ngày nghỉ; các phiên không giao dịch phải bị bỏ trước đó)."
        )
    if (close <= 0).any():
        raise ValueError(f"cột '{close_column}' phải > 0 để tính return/MAPE.")
    ret_col = f"target_ret_{config.horizon}d"
    dir_col = f"target_dir_{config.horizon}d"
    missing = [name for name in (ret_col, dir_col) if name not in frame.columns]
    if missing:
        raise ValueError(
            f"frame thiếu cột target {missing} cho horizon={config.horizon} — sinh bằng "
            f"labeling.build_targets(df, horizons=({config.horizon},), neutral_band_pct=...)."
        )


def _strip_target_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Bỏ mọi cột target (tiền tố TARGET_COLUMN_PREFIXES) — khung đưa vào predict()."""
    drop = [
        col
        for col in frame.columns
        if str(col).lower().startswith(TARGET_COLUMN_PREFIXES)
    ]
    return frame.drop(columns=drop)


def _check_model(model: Any) -> None:
    """Model phải có fit/predict callable — lỗi sớm với thông điệp rõ."""
    if not callable(getattr(model, "fit", None)) or not callable(getattr(model, "predict", None)):
        raise TypeError(
            "model từ model_factory phải hiện thực interface WalkForwardModel: "
            f"fit(train_df, val_df) và predict(test_df) → ModelPrediction; nhận {type(model).__name__}"
        )


def _as_1d(values: Any, name: str, n_expected: int) -> np.ndarray:
    """Chuẩn hoá một trường của ModelPrediction thành mảng float 1-D dài n_expected."""
    if values is None:
        return np.full(n_expected, np.nan)
    arr = np.asarray(values, dtype="float64").ravel()
    if arr.shape[0] != n_expected:
        raise ValueError(
            f"ModelPrediction.{name} dài {arr.shape[0]}, khác số dòng test {n_expected} — "
            "predict() phải trả dự báo cùng độ dài với test_df."
        )
    return arr


def _resolve_prediction(
    prediction: Any, test_rows: pd.DataFrame, close_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Kiểm tra đầu ra model và quy đổi về (predicted_price, probability_up).

    - predicted_price None mà có predicted_return → price = close_t * exp(return).
    - probability_up phải nằm trong [0, 1] (NaN được phép); predicted_price
      phải > 0 khi không phải NaN — vi phạm là lỗi hợp đồng model, raise.
    """
    if not isinstance(prediction, ModelPrediction):
        raise TypeError(
            f"predict() phải trả ModelPrediction, nhận {type(prediction).__name__} — "
            "xem hợp đồng interface chung trong src/evaluation.py."
        )
    n = len(test_rows)
    price = _as_1d(prediction.predicted_price, "predicted_price", n)
    ret = _as_1d(prediction.predicted_return, "predicted_return", n)
    prob = _as_1d(prediction.probability_up, "probability_up", n)
    if np.isnan(price).all() and not np.isnan(ret).all():
        price = close_values * np.exp(ret)
    if np.isnan(price).all() and np.isnan(prob).all():
        raise ValueError(
            "ModelPrediction rỗng: predicted_price/predicted_return/probability_up đều None/NaN "
            "— model phải điền ít nhất một phía."
        )
    known = price[~np.isnan(price)]
    if known.size and (known <= 0).any():
        raise ValueError(
            f"predicted_price phải > 0, nhận giá trị {known.min()}..{known.max()} — "
            "dự báo giá mức không hợp lệ."
        )
    prob_known = prob[~np.isnan(prob)]
    if prob_known.size and ((prob_known < 0.0) | (prob_known > 1.0)).any():
        raise ValueError(
            f"probability_up phải nằm trong [0, 1], nhận [{prob_known.min()}, {prob_known.max()}] "
            "— xác suất ngoài khoảng là lỗi hợp đồng model."
        )
    return price, prob


def walk_forward(
    frame: pd.DataFrame,
    model_factory: Callable[[], Any],
    config: WalkForwardConfig,
    *,
    close_column: str = "close",
) -> WalkForwardResult:
    """Chạy walk-forward expanding window và trả metrics từng fold + gộp.

    Tham số:
        frame: khung dòng theo phiên — cột giá ``close_column``, cột target
            ``target_ret_{h}d``/``target_dir_{h}d`` (labeling.build_targets)
            và các cột feature. Index DatetimeIndex tăng dần không trùng.
        model_factory: hàm KHÔNG tham số trả một model MỚI mỗi lần gọi —
            engine gọi đúng một lần mỗi fold (retrain mỗi fold, không mang
            trạng thái sang fold sau; scaler fit riêng trong fold là trách
            nhiệm của model nhưng bị engine ràng buộc bằng dữ liệu truyền vào).
        config: ``WalkForwardConfig`` (mặc định 750/21/embargo 5 khớp yaml).
        close_column: tên cột giá trong ``frame``.

    Trả ``WalkForwardResult``: ``folds`` (metrics từng fold), ``aggregate``
    (metrics gộp trên TOÀN BỘ dòng evaluation — mỗi dòng test đúng một lần vì
    các test block rời nhau) và ``predictions`` (bảng dự báo từng dòng test,
    gồm các dòng cuối khung chưa có kết quả — actual NaN, không vào metrics).
    """
    _validate_frame(frame, config, close_column)
    folds = make_folds(len(frame), config)
    n_sessions = len(frame)
    close_values = frame[close_column].to_numpy(dtype="float64")
    ret_col = f"target_ret_{config.horizon}d"
    dir_col = f"target_dir_{config.horizon}d"
    direction = frame[dir_col].to_numpy(dtype="float64", na_value=np.nan)  # NaN ở h phiên cuối

    fold_results: list[FoldResult] = []
    pred_parts: list[pd.DataFrame] = []

    for layout in folds:
        train_region = frame.iloc[layout.train_start : layout.train_end]
        if config.val_size > 0:
            fit_df = train_region.iloc[: len(train_region) - config.val_size].copy()
            val_df = train_region.iloc[len(train_region) - config.val_size :].copy()
        else:
            fit_df = train_region.copy()
            val_df = None

        model = model_factory()
        _check_model(model)
        model.fit(fit_df, val_df)

        test_rows = frame.iloc[layout.test_start : layout.test_end]
        n_test = len(test_rows)
        prediction = model.predict(_strip_target_columns(test_rows).copy())
        positions = np.arange(layout.test_start, layout.test_end)
        close_t = close_values[positions]
        target_positions = positions + config.horizon
        has_actual = target_positions < n_sessions
        actual_price = np.where(has_actual, close_values[np.minimum(target_positions, n_sessions - 1)], np.nan)
        actual_is_up = np.where(
            has_actual & ~np.isnan(direction[positions]),
            (direction[positions] == float(UP)).astype("float64"),
            np.nan,
        )

        predicted_price, prob_up = _resolve_prediction(prediction, test_rows, close_t)

        # Baseline majority-class là quyết định THEO FOLD: lớp đa số của train
        # fold này (majority_class_up chỉ nhìn slice train), cố định cho mọi
        # dòng test của fold — gộp sau này ghép các quyết định, không gộp nhãn.
        majority = majority_class_up(direction[layout.train_start : layout.train_end])
        baseline_up = (
            np.full(n_test, np.nan, dtype="float64")
            if majority is None
            else np.full(n_test, float(majority), dtype="float64")
        )

        reg = regression_metrics(actual_price, predicted_price, close_t)  # persistence = close_t
        cls = classification_metrics(actual_is_up, prob_up, baseline_is_up=baseline_up)
        metrics: dict[str, float | int | None] = {**reg, **cls}

        fold_results.append(
            FoldResult(
                layout=layout,
                train_start_date=frame.index[layout.train_start],
                train_end_date=frame.index[layout.train_end - 1],
                test_start_date=frame.index[layout.test_start],
                test_end_date=frame.index[layout.test_end - 1],
                metrics=metrics,
            )
        )
        logger.info(
            "walk_forward fold %d: train %d phiên (purge %d), test %d phiên, "
            "MAE %.6f vs persistence %.6f, acc %.3f vs base %.3f",
            layout.fold_id,
            layout.n_train,
            layout.purged_rows,
            n_test,
            reg["reg_mae"] if reg["reg_mae"] is not None else float("nan"),
            reg["reg_persistence_mae"] if reg["reg_persistence_mae"] is not None else float("nan"),
            cls["cls_accuracy"] if cls["cls_accuracy"] is not None else float("nan"),
            cls["cls_base_rate_majority"] if cls["cls_base_rate_majority"] is not None else float("nan"),
        )

        target_dates = pd.Series(pd.NaT, index=test_rows.index, dtype="datetime64[ns]")
        target_dates.iloc[has_actual] = frame.index[
            np.minimum(target_positions[has_actual], n_sessions - 1)
        ]
        prob_known = ~np.isnan(prob_up)
        actual_known = ~np.isnan(actual_is_up)
        predicted_up = pd.Series(pd.NA, index=test_rows.index, dtype="boolean")
        predicted_up.iloc[prob_known] = (
            prob_up[prob_known] >= UP_PROBABILITY_THRESHOLD
        )
        actual_up_col = pd.Series(pd.NA, index=test_rows.index, dtype="boolean")
        actual_up_col.iloc[actual_known] = actual_is_up[actual_known] == 1.0
        baseline_known = ~np.isnan(baseline_up)
        baseline_up_col = pd.Series(pd.NA, index=test_rows.index, dtype="boolean")
        baseline_up_col.iloc[baseline_known] = baseline_up[baseline_known] == 1.0
        pred_parts.append(
            pd.DataFrame(
                {
                    "fold_id": layout.fold_id,
                    "target_date": target_dates,
                    "close_t": close_t,
                    "predicted_price": predicted_price,
                    "persistence_price": close_t,
                    "actual_price": actual_price,
                    "prob_up": prob_up,
                    "predicted_up": predicted_up,
                    "actual_is_up": actual_up_col,
                    "baseline_up": baseline_up_col,
                },
                index=test_rows.index,
            )
        )

    predictions = pd.concat(pred_parts) if pred_parts else pd.DataFrame()

    # Gộp: mọi fold evaluation trên đúng các dòng của mình → gộp = micro (pooled).
    actual_all = predictions["actual_price"].to_numpy(dtype="float64") if len(predictions) else np.array([])
    price_all = predictions["predicted_price"].to_numpy(dtype="float64") if len(predictions) else np.array([])
    persistence_all = (
        predictions["persistence_price"].to_numpy(dtype="float64") if len(predictions) else np.array([])
    )
    up_all = (
        predictions["actual_is_up"].to_numpy(dtype="float64", na_value=np.nan)
        if len(predictions)
        else np.array([])
    )
    prob_all = predictions["prob_up"].to_numpy(dtype="float64") if len(predictions) else np.array([])
    # Baseline gộp = ghép QUYẾT ĐỊNH fold-honest (mỗi dòng test mang lớp đa số
    # của train FOLD CỦA CHÍNH NÓ). KHÔNG gộp nhãn train của mọi fold thành một
    # pool rồi suy lớp đa số: cửa sổ expanding của fold k+1 chứa gần trọn test
    # block của fold k → nhãn evaluation lọt vào "train" của baseline gộp và
    # ``cls_base_rate_majority``/``cls_edge_vs_base_rate`` bị lợi thế thông tin
    # (leakage của riêng baseline; model không bao giờ thấy dữ liệu như vậy).
    baseline_all = (
        predictions["baseline_up"].to_numpy(dtype="float64", na_value=np.nan)
        if len(predictions)
        else np.array([])
    )

    aggregate: dict[str, float | int | None] = {
        "n_folds": len(folds),
        "n_test_rows": int(len(predictions)),
        **regression_metrics(actual_all, price_all, persistence_all),
        **classification_metrics(up_all, prob_all, baseline_is_up=baseline_all),
    }

    return WalkForwardResult(
        config=config,
        folds=fold_results,
        predictions=predictions,
        aggregate=aggregate,
    )


__all__ = [
    "METRIC_KEYS",
    "TARGET_COLUMN_PREFIXES",
    "UP_PROBABILITY_THRESHOLD",
    "FoldLayout",
    "FoldResult",
    "InsufficientHistoryError",
    "ModelPrediction",
    "WalkForwardConfig",
    "WalkForwardModel",
    "WalkForwardResult",
    "classification_metrics",
    "majority_class_up",
    "make_folds",
    "regression_metrics",
    "walk_forward",
]
