"""Kiểm thử src/evaluation — walk-forward đúng cấu hình, đúng biên, chống leakage.

Toàn bộ OFFLINE trên khung giá giả lập nhỏ (vài chục phiên, random seed cố định).
Chiến lược: model GIẢ (dummy) ghi lại mọi khung engine đưa vào để khẳng định
ranh giới fold; một số model giả "biết tương lai" một cách CỐ Ý từ phía TEST
(test tự đọc đáp án trong khung gốc và truyền cho model — engine thì KHÔNG
bao giờ đưa target vào predict) chỉ để kiểm tra đúng phép toán metrics.

Các nhóm kiểm chứng:
1. Hình học fold: đúng số fold, đúng biên embargo/purge, train/test không trùng,
   test block rời nhau và phủ kín phần test.
2. Ràng buộc dữ liệu cho model: fit chỉ nhận train fold (val là phần đuôi
   train), predict không bao giờ thấy cột target; model mới hoàn toàn mỗi fold
   (scaler fit riêng trong fold là hệ quả bắt buộc — kiểm chứng bằng
   StandardScaler probe với một giá trị "mồi" chỉ xuất hiện ở vùng test).
3. Phép toán metrics: MAE/RMSE/MAPE/skill vs persistence và Accuracy/F1/AUC/
   edge vs base rate được tính lại ĐỘC LẬP trong test rồi so khớp.
4. Hợp đồng model/input: lỗi sớm với thông điệp rõ khi frame sai hoặc model
   trả dự báo sai độ dài/sai miền giá trị.
"""
from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.labeling import build_targets

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Horizon dùng chung cho khung giả lập (config thật là [1, 5] — engine chạy 1 horizon/lần).
H = 2
FEATURES = ["f_mom", "f_ma3", "f_sentinel"]
RET_COL = f"target_ret_{H}d"
DIR_COL = f"target_dir_{H}d"


def make_frame(
    n: int = 60,
    seed: int = 7,
    drift: float = 0.0,
    band: float = 0.3,
    horizon: int = H,
    sentinel_pos: int | None = None,
) -> pd.DataFrame:
    """Khung feature+close+target giả lập: index phiên tăng dần, target đúng chuẩn labeling."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(
        50.0 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx, name="close"
    )
    frame = pd.DataFrame({"close": close})
    frame["f_mom"] = close.pct_change().fillna(0.0)
    frame["f_ma3"] = frame["f_mom"].rolling(3, min_periods=1).mean()
    frame["f_sentinel"] = 0.0
    if sentinel_pos is not None:
        frame.iloc[sentinel_pos, frame.columns.get_loc("f_sentinel")] = 1000.0
    targets = build_targets(frame[["close"]], horizons=(horizon,), neutral_band_pct=band)
    return pd.concat([frame, targets], axis=1)


def cfg(**kwargs: Any) -> ev.WalkForwardConfig:
    """Config nhỏ cho test: train đầu 30, retrain 10, embargo 3, horizon H."""
    return ev.WalkForwardConfig(initial_train_size=30, retrain_every=10, embargo=3, horizon=H, **kwargs)


class RecordingModel:
    """Model giả ghi lại mọi khung engine đưa vào fit/predict để khẳng định ranh giới.

    price_mode / prob_mode:
      - "close": dự báo giá = close_t (giống hệt persistence → skill phải bằng 0);
      - callable(test_df) -> array: công thức tuỳ ý (expected do test tự tính);
      - mảng/số tường minh.
    """

    def __init__(
        self,
        price_mode: Any = "close",
        prob_mode: Any = 0.5,
    ) -> None:
        self.price_mode = price_mode
        self.prob_mode = prob_mode
        self.fit_calls: list[tuple[pd.Index, pd.Index | None, list[str]]] = []
        self.predict_calls: list[tuple[pd.Index, list[str]]] = []

    @staticmethod
    def _resolve(mode: Any, test_df: pd.DataFrame, fallback: float) -> np.ndarray:
        if mode == "close":
            return test_df["close"].to_numpy(dtype="float64")
        if callable(mode):
            return np.asarray(mode(test_df), dtype="float64")
        if np.isscalar(mode):
            return np.full(len(test_df), float(mode), dtype="float64")
        return np.asarray(mode, dtype="float64")

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> "RecordingModel":
        self.fit_calls.append(
            (
                train_df.index.copy(),
                None if val_df is None else val_df.index.copy(),
                list(train_df.columns),
            )
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> ev.ModelPrediction:
        self.predict_calls.append((test_df.index.copy(), list(test_df.columns)))
        price = self._resolve(self.price_mode, test_df, 0.0)
        prob = self._resolve(self.prob_mode, test_df, 0.5)
        return ev.ModelPrediction(predicted_price=price, probability_up=prob)


def tracking_factory(maker: Callable[[], Any]) -> Any:
    """Factory đếm số lần được gọi và giữ lại các instance đã tạo (retrain mỗi fold)."""
    calls: list[Any] = []

    def factory() -> Any:
        model = maker()
        calls.append(model)
        return model

    factory.calls = calls  # type: ignore[attr-defined]
    return factory


def positions_of(frame: pd.DataFrame, dates: pd.Index) -> np.ndarray:
    """Vị trí tuyệt đối của các dòng trong khung gốc."""
    return np.array([frame.index.get_loc(d) for d in dates])


# ---------------------------------------------------------------------------
# 1. Cấu hình & hình học fold
# ---------------------------------------------------------------------------


class TestConfig:
    def test_from_real_config_yaml_matches_common_750_21_5(self) -> None:
        """Hợp đồng COMMON: from_mapping phải đọc đúng config.yaml thật của dự án."""
        config = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text(encoding="utf-8"))
        parsed = ev.WalkForwardConfig.from_mapping(config, horizon=1)
        assert (parsed.initial_train_size, parsed.retrain_every, parsed.embargo) == (750, 21, 5)
        # Không có mục cấu hình cũng phải rơi về đúng Common 750/21/5.
        default = ev.WalkForwardConfig.from_mapping({})
        assert (default.initial_train_size, default.retrain_every, default.embargo) == (750, 21, 5)
        assert (default.val_size, default.horizon) == (0, 1)

    def test_rejects_shuffle_and_bad_params(self) -> None:
        with pytest.raises(ValueError, match="shuffle"):
            ev.WalkForwardConfig.from_mapping({"walk_forward": {"shuffle": True}})
        with pytest.raises(ValueError, match="initial_train_size"):
            ev.WalkForwardConfig(initial_train_size=0)
        with pytest.raises(ValueError, match="retrain_every"):
            ev.WalkForwardConfig(retrain_every=0)
        with pytest.raises(ValueError, match="embargo"):
            ev.WalkForwardConfig(embargo=-1)
        with pytest.raises(ValueError, match="horizon"):
            ev.WalkForwardConfig(horizon=0)
        with pytest.raises(ValueError, match="val_size"):
            ev.WalkForwardConfig(initial_train_size=30, val_size=30)


class TestFoldGeometry:
    def test_make_folds_exact_layout(self) -> None:
        """n=60, train 30 / retrain 10 / embargo 3 / horizon 2 → 3 fold đúng biên."""
        folds = ev.make_folds(60, cfg())
        actual = [(f.test_start, f.test_end, f.train_end, f.purged_rows) for f in folds]
        # test block: [33,43) [43,53) [53,60); train_end = min(test−embargo, test−horizon)
        assert actual == [(33, 43, 30, 0), (43, 53, 40, 0), (53, 60, 50, 0)]
        assert all(f.train_start == 0 for f in folds)  # expanding window
        assert all(f.gap_rows == 3 for f in folds)  # đúng biên embargo theo config

    def test_purge_enforced_when_embargo_below_horizon(self) -> None:
        """embargo 1 < horizon 3: engine tự nới khoảng cách lên horizon (purge 2 dòng)."""
        config = ev.WalkForwardConfig(initial_train_size=30, retrain_every=10, embargo=1, horizon=3)
        folds = ev.make_folds(55, config)
        assert (folds[0].test_start, folds[0].train_end, folds[0].purged_rows) == (31, 28, 2)
        # Bất biến leakage: mọi nhãn train chỉ dùng giá tại vị trí < test_start.
        for fold in folds:
            assert fold.train_end + config.horizon <= fold.test_start
            assert fold.gap_rows >= config.horizon

    def test_default_common_layout(self) -> None:
        """Config Common 750/21/5 trên 900 phiên: fold đầu test [755, 776), train_end 750, purge 0."""
        folds = ev.make_folds(900, ev.WalkForwardConfig())
        assert (
            folds[0].test_start,
            folds[0].test_end,
            folds[0].train_end,
            folds[0].purged_rows,
        ) == (755, 776, 750, 0)
        # test_start = 755 + 21k < 900 → k = 0..6 → 7 fold, block cuối [880, 900).
        assert len(folds) == 7
        assert folds[-1].test_end == 900
        assert all(fold.purged_rows == 0 for fold in folds)  # embargo 5 >= mọi horizon của config

    def test_insufficient_history_raises(self) -> None:
        config = ev.WalkForwardConfig()  # cần >= 750 + 5 + 1 = 756 phiên
        with pytest.raises(ev.InsufficientHistoryError, match="756"):
            ev.make_folds(755, config)
        frame = make_frame(n=20)
        with pytest.raises(ev.InsufficientHistoryError, match="phiên"):
            ev.walk_forward(frame, lambda: RecordingModel(), cfg())


# ---------------------------------------------------------------------------
# 2. Ranh giới dữ liệu đưa vào model (chống leakage ở tầng engine)
# ---------------------------------------------------------------------------


class TestEngineBoundaries:
    def test_fold_count_and_no_train_test_overlap(self) -> None:
        frame = make_frame(n=60)
        result = ev.walk_forward(frame, lambda: RecordingModel(), cfg())
        folds = [f.layout for f in result.folds]
        assert len(folds) == 3  # đúng số fold
        seen_test: set[int] = set()
        for fold in folds:
            train_pos = set(range(fold.train_start, fold.train_end))
            test_pos = set(range(fold.test_start, fold.test_end))
            assert train_pos.isdisjoint(test_pos)  # train/test không trùng
            assert max(train_pos) + H < fold.test_start  # nhãn train không chạm vùng test
            assert fold.test_start - fold.train_end >= 3  # biên embargo được tôn trọng
            assert seen_test.isdisjoint(test_pos)  # test block các fold rời nhau
            seen_test |= test_pos
        assert sorted(seen_test) == list(range(33, 60))  # phủ kín, mỗi dòng test đúng 1 lần

    def test_fresh_model_per_fold_and_order_preserved(self) -> None:
        frame = make_frame(n=60)
        recorded = RecordingModel()
        factory = tracking_factory(lambda: RecordingModel())
        ev.walk_forward(frame, factory, cfg())
        assert len(factory.calls) == 3  # retrain: một model mới mỗi fold
        assert len({id(m) for m in factory.calls}) == 3  # không tái dùng instance
        recorded = factory.calls[0]
        for i, (train_idx, val_idx, _cols) in enumerate(recorded.fit_calls):
            assert train_idx.is_monotonic_increasing  # không shuffle
            test_idx = recorded.predict_calls[i][0]
            assert test_idx.is_monotonic_increasing
            assert train_idx.max() < test_idx.min()  # toàn bộ train đứng trước test
            if val_idx is not None:
                assert val_idx.max() < test_idx.min()

    def test_predict_never_sees_target_columns(self) -> None:
        """Guard chống peek: predict() nhận khung ĐÃ BỎ cột target; fit thì có."""
        frame = make_frame(n=60)
        recorded = RecordingModel()
        ev.walk_forward(frame, lambda: recorded, cfg())
        for _train_idx, _val_idx, train_cols in recorded.fit_calls:
            assert RET_COL in train_cols and DIR_COL in train_cols
        for _test_idx, test_cols in recorded.predict_calls:
            assert not any(
                str(col).lower().startswith(("target_", "fwd_", "label_")) for col in test_cols
            )
            assert "close" in test_cols and "f_mom" in test_cols  # giá tại t và feature vẫn còn

    def test_val_df_is_train_tail_not_test(self) -> None:
        frame = make_frame(n=60)
        recorded = RecordingModel()
        ev.walk_forward(frame, lambda: recorded, cfg(val_size=5))
        for i, (fit_idx, val_idx, _cols) in enumerate(recorded.fit_calls):
            assert val_idx is not None and len(val_idx) == 5
            assert fit_idx.max() < val_idx.min()  # val là ĐUÔI của vùng train
            assert val_idx.is_monotonic_increasing
            test_idx = recorded.predict_calls[i][0]
            assert val_idx.max() < test_idx.min()  # val không đụng test
            if i == 0:  # fold 0: train_end = 30 → fit [0,25), val [25,30)
                assert list(val_idx) == list(frame.index[25:30])
                assert list(fit_idx) == list(frame.index[0:25])

    def test_scaler_fit_per_fold_on_train_only(self) -> None:
        """Scaler probe: mean fit đúng trên train-only của từng fold; giá trị mồi
        nằm ở vùng test KHÔNG được lọt vào thống kê của fold trước."""
        frame = make_frame(n=60, seed=11, sentinel_pos=45)  # 45 ∈ test block fold 1 [43,53)
        scalers: list[StandardScaler] = []

        class ScalerProbe:
            def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> "ScalerProbe":
                scalers.append(StandardScaler().fit(train_df[FEATURES]))
                return self

            def predict(self, test_df: pd.DataFrame) -> ev.ModelPrediction:
                return ev.ModelPrediction(
                    predicted_price=test_df["close"].to_numpy(dtype="float64"),
                    probability_up=np.full(len(test_df), 0.5),
                )

        result = ev.walk_forward(frame, lambda: ScalerProbe(), cfg())
        assert len(scalers) == len(result.folds) == 3
        for scaler, fold_result in zip(scalers, result.folds):
            expected = (
                frame.iloc[fold_result.layout.train_start : fold_result.layout.train_end][FEATURES]
                .mean()
                .to_numpy(dtype="float64")
            )
            assert np.allclose(scaler.mean_, expected, rtol=0.0, atol=1e-12)
        # Train-only: fold 0 chưa từng thấy sentinel (nó nằm ở test của fold 1) → mean 0,
        # trong khi mean toàn chuỗi bị giá trị mồi làm lệch hẳn (bài học: scaler fit
        # trên toàn chuỗi trước split là leakage).
        assert scalers[0].mean_[FEATURES.index("f_sentinel")] == 0.0
        assert not np.isclose(
            scalers[0].mean_[FEATURES.index("f_sentinel")], frame["f_sentinel"].mean()
        )
        # Expanding window: thống kê hai fold khác nhau → fit lại scaler mỗi fold là bắt buộc.
        assert not np.isclose(scalers[0].mean_[0], scalers[1].mean_[0])


# ---------------------------------------------------------------------------
# 3. Phép toán metrics — tính lại độc lập trong test
# ---------------------------------------------------------------------------


class TestRegressionMetrics:
    def test_hand_computed_mae_rmse_mape_skill(self) -> None:
        """Model dự báo +1% so với close_t: expected tính trực tiếp từ khung, so khớp engine."""
        frame = make_frame(n=60, seed=3)
        markup = 0.01
        model = RecordingModel(
            price_mode=lambda td: td["close"].to_numpy(dtype="float64") * (1.0 + markup)
        )
        result = ev.walk_forward(frame, lambda: model, cfg())

        close = frame["close"].to_numpy(dtype="float64")
        positions = np.arange(33, 58)  # mọi dòng CÓ kết quả (58,59 chưa có target h=2)
        actual = close[positions + H]
        pred = close[positions] * (1.0 + markup)
        persistence = close[positions]
        expected = {
            "reg_n_rows": len(positions),
            "reg_mae": float(np.mean(np.abs(pred - actual))),
            "reg_rmse": float(np.sqrt(np.mean((pred - actual) ** 2))),
            "reg_mape_pct": float(np.mean(np.abs(pred - actual) / actual) * 100.0),
            "reg_persistence_mae": float(np.mean(np.abs(persistence - actual))),
            "reg_persistence_rmse": float(np.sqrt(np.mean((persistence - actual) ** 2))),
            "reg_skill_vs_persistence_pct": float(
                100.0
                * (1.0 - np.mean(np.abs(pred - actual)) / np.mean(np.abs(persistence - actual)))
            ),
        }
        assert result.aggregate["n_test_rows"] == 27  # 58,59 là dòng live, không vào metrics
        for key, value in expected.items():
            assert result.aggregate[key] == pytest.approx(value, rel=1e-12), key

    def test_persistence_identical_model_zero_skill(self) -> None:
        """Dự báo = close_t chính là persistence → skill đúng bằng 0, MAE trùng khớp."""
        frame = make_frame(n=60, seed=5)
        result = ev.walk_forward(frame, lambda: RecordingModel(price_mode="close"), cfg())
        agg = result.aggregate
        assert agg["reg_mae"] == pytest.approx(agg["reg_persistence_mae"], rel=1e-12)
        assert agg["reg_skill_vs_persistence_pct"] == pytest.approx(0.0, abs=1e-9)
        # Trong bảng dự báo: persistence_price phải đúng là close_t (cùng window).
        assert np.allclose(
            result.predictions["persistence_price"], result.predictions["close_t"], rtol=0.0
        )

    def test_perfect_model_full_scores(self) -> None:
        """Model giả BIẾT TƯƠNG LAI (test tự cấp đáp án để kiểm tra phép toán):
        MAE=0 → skill 100; prob = nhãn thật → accuracy/F1/AUC = 1, edge = 1 − base rate."""
        frame = make_frame(n=60, seed=9, drift=0.006)  # drift dương → đa số train là LỚN
        close = frame["close"].to_numpy(dtype="float64")
        n = len(frame)
        labels = frame[DIR_COL].to_numpy(dtype="float64", na_value=np.nan)

        def price_fn(test_df: pd.DataFrame) -> np.ndarray:
            pos = positions_of(frame, test_df.index)
            return close[np.minimum(pos + H, n - 1)]  # dòng live trùng close_t — không vào metrics

        def prob_fn(test_df: pd.DataFrame) -> np.ndarray:
            pos = positions_of(frame, test_df.index)
            return np.where(np.isnan(labels[pos]), 0.5, labels[pos] == 1.0).astype("float64")

        model = RecordingModel(price_mode=price_fn, prob_mode=prob_fn)
        result = ev.walk_forward(frame, lambda: model, cfg())
        agg = result.aggregate
        assert agg["reg_mae"] == pytest.approx(0.0, abs=1e-12)
        assert agg["reg_skill_vs_persistence_pct"] == pytest.approx(100.0)
        assert agg["cls_accuracy"] == pytest.approx(1.0)
        assert agg["cls_f1_up"] == pytest.approx(1.0)
        assert agg["cls_roc_auc"] == pytest.approx(1.0)
        # Base rate majority-class: lớp đa số của TRAIN, chấm trên CÙNG dòng evaluation.
        train_up = frame[DIR_COL].iloc[0:50].dropna().astype(float) == 1.0  # train fold cuối [0,50)
        expected_majority = bool(np.mean(train_up) >= 0.5)
        eval_up = frame[DIR_COL].iloc[33:58].dropna().astype(float) == 1.0
        expected_base_rate = float(np.mean(eval_up == expected_majority))
        assert agg["cls_base_rate_majority"] == pytest.approx(expected_base_rate)
        assert agg["cls_edge_vs_base_rate"] == pytest.approx(1.0 - expected_base_rate)
        # Cột predicted_up trong bảng dự báo phải nhất quán với quy ước prob >= 0.5:
        # prob ở đây chính là nhãn thật → predicted_up == actual_is_up trên dòng có kết quả.
        known = result.predictions["actual_is_up"].notna()
        assert (
            result.predictions.loc[known, "predicted_up"]
            == result.predictions.loc[known, "actual_is_up"]
        ).all()


class TestClassificationMetrics:
    def test_majority_base_rate_and_edge(self) -> None:
        """prob hằng 0.5 → dự báo LỚN mọi dòng → accuracy == base rate (edge 0);
        prob 0.0 → dự báo không-LỚN mọi dòng → edge âm, đúng công thức."""
        frame = make_frame(n=60, seed=9, drift=0.006)  # đa số train: LỚN
        all_up = ev.walk_forward(frame, lambda: RecordingModel(prob_mode=0.5), cfg())
        all_down = ev.walk_forward(frame, lambda: RecordingModel(prob_mode=0.0), cfg())

        eval_labels = frame[DIR_COL].iloc[33:58].dropna().astype(float) == 1.0
        up_rate = float(np.mean(eval_labels))
        assert all_up.aggregate["cls_accuracy"] == pytest.approx(up_rate)
        assert all_up.aggregate["cls_base_rate_majority"] == pytest.approx(up_rate)
        assert all_up.aggregate["cls_edge_vs_base_rate"] == pytest.approx(0.0, abs=1e-12)

        assert all_down.aggregate["cls_accuracy"] == pytest.approx(1.0 - up_rate)
        assert all_down.aggregate["cls_edge_vs_base_rate"] == pytest.approx((1.0 - up_rate) - up_rate)
        assert all_down.aggregate["cls_edge_vs_base_rate"] < 0.0  # thua majority baseline

    def test_metrics_function_edge_cases(self) -> None:
        """Đơn vị hàm: actual một lớp → AUC None; hết dòng hợp lệ → None toàn bộ;
        persistence_mae = 0 → skill None (không chia 0); NaN bị bỏ khỏi phép tính."""
        single = ev.classification_metrics(
            np.array([1.0, 1.0, 1.0]), np.array([0.2, 0.8, 0.4]), np.array([1.0, 0.0, 1.0])
        )
        assert single["cls_roc_auc"] is None
        assert single["cls_n_rows"] == 3
        # prob [0.2, 0.8, 0.4] ≥ 0.5 → dự báo LỚN ở đúng 1/3 dòng (nhãn thật cả 3 đều LỚN).
        assert single["cls_accuracy"] == pytest.approx(1.0 / 3.0)

        empty = ev.classification_metrics(
            np.array([np.nan, np.nan]), np.array([0.4, 0.6]), np.array([1.0])
        )
        assert empty["cls_n_rows"] == 0
        assert empty["cls_accuracy"] is None and empty["cls_edge_vs_base_rate"] is None

        normal = ev.regression_metrics(
            np.array([10.0, 10.1]), np.array([10.05, 10.05]), np.array([10.0, 10.0])
        )
        assert normal["reg_persistence_mae"] > 0.0
        assert normal["reg_skill_vs_persistence_pct"] is not None
        flat = ev.regression_metrics(
            np.array([10.0, 10.0]), np.array([10.0, 10.0]), np.array([10.0, 10.0])
        )
        assert flat["reg_persistence_mae"] == 0.0
        assert flat["reg_skill_vs_persistence_pct"] is None

        nan_dropped = ev.classification_metrics(
            np.array([1.0, np.nan, 0.0]), np.array([0.9, 0.9, 0.9]), np.array([1.0, 1.0])
        )
        assert nan_dropped["cls_n_rows"] == 2  # chỉ dòng actual NaN bị bỏ
        assert nan_dropped["cls_accuracy"] == pytest.approx(0.5)  # [pred 1 vs 1; pred 1 vs 0]

        reg_nan = ev.regression_metrics(
            np.array([10.0, np.nan, 11.0]),
            np.array([10.5, 10.5, np.nan]),
            np.array([10.0, 10.0, 10.0]),
        )
        assert reg_nan["reg_n_rows"] == 1  # actual NaN hoặc pred NaN đều bị bỏ


# ---------------------------------------------------------------------------
# 4. Bảng dự báo, hợp đồng model và validate đầu vào
# ---------------------------------------------------------------------------


class TestOutputAndContract:
    def test_predictions_frame_structure(self) -> None:
        frame = make_frame(n=60, seed=3)
        result = ev.walk_forward(frame, lambda: RecordingModel(), cfg())
        preds = result.predictions
        assert preds.index.is_unique  # mỗi dòng test đúng một lần (test block rời nhau)
        assert len(preds) == 27
        for col in (
            "fold_id",
            "target_date",
            "close_t",
            "predicted_price",
            "persistence_price",
            "actual_price",
            "prob_up",
            "predicted_up",
            "actual_is_up",
        ):
            assert col in preds.columns, col
        # Dòng live cuối khung (58, 59): actual NaN/NaT/NA — vẫn nằm trong bảng, chưa chấm điểm.
        assert np.isnan(preds["actual_price"].iloc[-1])
        assert pd.isna(preds["target_date"].iloc[-1])
        assert pd.isna(preds["actual_is_up"].iloc[-1])
        # Dòng có kết quả: actual đúng close tại t+h, target_date đúng phiên t+h.
        close = frame["close"].to_numpy(dtype="float64")
        for date, row in preds.iloc[:-2].iterrows():
            pos = frame.index.get_loc(date)
            assert row["actual_price"] == pytest.approx(close[pos + H])
            assert row["target_date"] == frame.index[pos + H]
        # Phân bổ fold theo test block.
        assert list(preds[preds["fold_id"] == 0].index) == list(frame.index[33:43])
        assert list(preds[preds["fold_id"] == 2].index) == list(frame.index[53:60])

    def test_fold_summary_reports_layout(self) -> None:
        frame = make_frame(n=60, seed=3)
        result = ev.walk_forward(frame, lambda: RecordingModel(), cfg())
        summary = result.fold_summary()
        assert len(summary) == 3
        assert (summary["purged_rows"] == 0).all()  # embargo 3 >= horizon 2
        assert (summary["embargo"] == 3).all()
        assert list(summary["n_test"]) == [10, 10, 7]
        assert summary["train_start"].iloc[0] == frame.index[0]
        assert summary["test_end"].iloc[-1] == frame.index[59]

    def test_model_contract_violations_raise(self) -> None:
        frame = make_frame(n=40, seed=3)
        n_test = 40 - 30 - 3  # 7 dòng test của fold duy nhất

        class Fixed:
            def __init__(self, prediction: Any) -> None:
                self.prediction = prediction

            def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> "Fixed":
                return self

            def predict(self, test_df: pd.DataFrame) -> Any:
                return self.prediction

        def run(model: Any) -> None:
            ev.walk_forward(frame, lambda: model, cfg())

        with pytest.raises(ValueError, match="độ dài"):
            run(Fixed(ev.ModelPrediction(predicted_price=np.zeros(n_test + 1))))
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            run(Fixed(ev.ModelPrediction(probability_up=np.full(n_test, 1.5))))
        with pytest.raises(ValueError, match="> 0"):
            run(Fixed(ev.ModelPrediction(predicted_price=np.full(n_test, -5.0))))
        with pytest.raises(TypeError, match="ModelPrediction"):
            run(Fixed(np.zeros(n_test)))
        with pytest.raises(ValueError, match="ít nhất"):
            run(Fixed(ev.ModelPrediction()))
        with pytest.raises(TypeError, match="interface"):
            run(object())

    def test_return_only_prediction_is_converted_to_price(self) -> None:
        """Model chỉ trả predicted_return (log-return t→t+h): engine quy đổi giá close_t*exp(r)."""
        frame = make_frame(n=40, seed=3)
        ret = 0.02

        class ReturnOnly:
            def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None) -> "ReturnOnly":
                return self

            def predict(self, test_df: pd.DataFrame) -> ev.ModelPrediction:
                return ev.ModelPrediction(predicted_return=np.full(len(test_df), ret))

        result = ev.walk_forward(frame, lambda: ReturnOnly(), cfg())
        expected = frame["close"].iloc[33:40].to_numpy(dtype="float64") * np.exp(ret)
        assert np.allclose(result.predictions["predicted_price"], expected, rtol=1e-12)

    def test_frame_validation_errors(self) -> None:
        frame = make_frame(n=40, seed=3)
        # Index không tăng dần (bị shuffle) → từ chối.
        shuffled = frame.iloc[np.random.default_rng(0).permutation(len(frame))]
        with pytest.raises(ValueError, match="tăng dần"):
            ev.walk_forward(shuffled, lambda: RecordingModel(), cfg())
        # Trùng ngày phiên → từ chối.
        dup = pd.concat([frame, frame.iloc[[-1]]]).sort_index()
        with pytest.raises(ValueError, match="trùng"):
            ev.walk_forward(dup, lambda: RecordingModel(), cfg())
        # Thiếu cột target cho horizon cấu hình → chỉ dẫn sinh bằng labeling.
        with pytest.raises(ValueError, match="build_targets"):
            ev.walk_forward(frame.drop(columns=[RET_COL]), lambda: RecordingModel(), cfg())
        # Close NaN → từ chối (không học trên giá bịa).
        nan_close = frame.copy()
        nan_close.iloc[5, nan_close.columns.get_loc("close")] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ev.walk_forward(nan_close, lambda: RecordingModel(), cfg())
        # Index không phải DatetimeIndex → từ chối.
        with pytest.raises(ValueError, match="DatetimeIndex"):
            ev.walk_forward(frame.reset_index(drop=True), lambda: RecordingModel(), cfg())


# ---------------------------------------------------------------------------
# Baseline majority-class GỘP phải fold-honest (chặn leakage của baseline)
# ---------------------------------------------------------------------------


def make_regime_frame(n: int = 120, split: int = 40) -> pd.DataFrame:
    """Khung giả lập ĐỔI REGIME rõ: đầu kỳ tăng (nhãn LỚN), sau đó giảm (không-LỚN).

    Biên độ 1%/phiên >> vùng neutral 0.3% → nhãn theo đúng labeling.build_targets
    xác định rõ từng vùng; lớp đa số train của các fold ĐẦU khác với lớp đa số
    của pool gộp (union các cửa sổ expanding) → phân biệt được hai cách gộp.
    """
    idx = pd.bdate_range("2024-01-02", periods=n)
    rets = np.where(np.arange(n) < split, 0.01, -0.01)
    close = pd.Series(50.0 * np.exp(np.cumsum(rets)), index=idx, name="close")
    frame = pd.DataFrame({"close": close})
    frame["f_mom"] = close.pct_change().fillna(0.0)
    frame["f_ma3"] = frame["f_mom"].rolling(3, min_periods=1).mean()
    frame["f_sentinel"] = 0.0
    targets = build_targets(frame[["close"]], horizons=(H,), neutral_band_pct=0.3)
    return pd.concat([frame, targets], axis=1)


class TestAggregateBaselineFoldHonest:
    """cls_base_rate_majority GỘP phải ghép quyết định majority THEO FOLD.

    Lỗi bị chặn: gộp nhãn train của TẤT CẢ các fold (cửa sổ expanding chồng
    nhau — train của fold sau chứa gần trọn test block của fold trước) rồi suy
    lớp đa số từ pool đó → baseline gộp nhìn thấy nhãn của chính các dòng
    evaluation, ``cls_base_rate_majority``/``cls_edge_vs_base_rate`` có lợi thế
    thông tin mà model không bao giờ được hưởng. Baseline trung thực: mỗi fold
    dùng lớp đa số train CỦA FOLD ĐÓ (không nhìn test), gộp micro theo dòng.
    """

    @staticmethod
    def _fold_majority(labels: np.ndarray, layout: ev.FoldLayout) -> bool:
        seg = labels[layout.train_start : layout.train_end]
        return bool(np.mean(seg[~np.isnan(seg)]) >= 0.5)  # hoà 50/50 → LỚN, đúng engine

    def test_aggregate_baseline_is_fold_honest_not_pooled(self) -> None:
        frame = make_regime_frame()
        n = len(frame)
        folds = ev.make_folds(n, cfg())
        # Tiền đề của lỗi: train fold kế chứa các dòng test của fold trước.
        overlap = folds[1].train_end - folds[0].test_start
        assert overlap > 0, "thiết lập test phải có cửa sổ train chồng lên test block fold trước"

        labels = frame[DIR_COL].to_numpy(dtype="float64", na_value=np.nan)
        result = ev.walk_forward(frame, lambda: RecordingModel(prob_mode=0.5), cfg())
        agg = result.aggregate

        # Tính lại ĐỘC LẬP baseline trung thực: đa số train từng fold, chấm trên
        # đúng các dòng test có kết quả của fold đó, gộp micro (theo dòng).
        hits = 0
        scored = 0
        fold_majors = [self._fold_majority(labels, layout) for layout in folds]
        for layout, majority in zip(folds, fold_majors, strict=True):
            pos = np.arange(layout.test_start, layout.test_end)
            actual = np.where(pos + H < n, labels[pos] == float(ev.UP), np.nan)
            valid = ~np.isnan(actual)
            hits += int(np.sum(actual[valid].astype(int) == int(majority)))
            scored += int(valid.sum())
        honest = hits / scored

        # Test này phải PHÂN BIỆT được hai cách gộp: đa số pool gộp (union các
        # cửa sổ expanding = train fold cuối) phải khác đa số của fold đầu.
        pooled_majority = self._fold_majority(labels, folds[-1])
        assert fold_majors[0] != pooled_majority
        # Giá trị cũ (leak): một lớp đa số duy nhất từ pool, chấm trên mọi dòng.
        pos_all = np.arange(folds[0].test_start, folds[-1].test_end)
        actual_all = np.where(pos_all + H < n, labels[pos_all] == float(ev.UP), np.nan)
        contaminated = float(np.mean(actual_all[~np.isnan(actual_all)].astype(int) == int(pooled_majority)))
        assert contaminated != pytest.approx(honest)

        assert agg["cls_base_rate_majority"] == pytest.approx(honest)
        assert agg["cls_base_rate_majority"] != pytest.approx(contaminated)
        assert agg["cls_edge_vs_base_rate"] == pytest.approx(agg["cls_accuracy"] - honest)

    def test_baseline_up_column_matches_own_fold_train_majority(self) -> None:
        """Cột baseline_up: mỗi dòng test mang lớp đa số của train FOLD CỦA NÓ."""
        frame = make_regime_frame()
        folds = ev.make_folds(len(frame), cfg())
        labels = frame[DIR_COL].to_numpy(dtype="float64", na_value=np.nan)
        result = ev.walk_forward(frame, lambda: RecordingModel(prob_mode=0.5), cfg())
        assert "baseline_up" in result.predictions.columns
        for layout in folds:
            majority = self._fold_majority(labels, layout)
            block = result.predictions.loc[frame.index[layout.test_start : layout.test_end]]
            assert block["baseline_up"].notna().all()
            assert (block["baseline_up"] == majority).all()

    def test_majority_class_up_tie_and_empty(self) -> None:
        """Quy ước lớp đa số: hoà 50/50 → LỚN; không có nhãn hợp lệ → None."""
        assert ev.majority_class_up(np.array([1.0, 0.0, 1.0])) is True
        assert ev.majority_class_up(np.array([0.0, 0.0, 1.0])) is False
        assert ev.majority_class_up(np.array([1.0, 0.0])) is True  # hoà → LỚN
        assert ev.majority_class_up(np.array([np.nan, np.nan])) is None
        assert ev.majority_class_up(np.array([])) is None

    def test_classification_metrics_baseline_is_up_and_exclusivity(self) -> None:
        """baseline_is_up theo dòng: chấm đúng, bỏ dòng NaN; truyền cả hai baseline là lỗi."""
        actual = np.array([1.0, 0.0, np.nan, 1.0])
        prob = np.array([0.9, 0.9, 0.9, 0.2])
        out = ev.classification_metrics(actual, prob, baseline_is_up=np.array([1.0, 1.0, 0.0, 0.0]))
        assert out["cls_n_rows"] == 3  # dòng actual NaN bị bỏ
        # y_pred = [LỚN, LỚN, không] vs y_true = [LỚN, không, LỚN] → accuracy 1/3
        assert out["cls_accuracy"] == pytest.approx(1.0 / 3.0)
        # baseline hits: [đúng, sai, sai] → 1/3; edge = 0
        assert out["cls_base_rate_majority"] == pytest.approx(1.0 / 3.0)
        assert out["cls_edge_vs_base_rate"] == pytest.approx(0.0)

        # Dòng baseline NaN bị loại khỏi base rate (accuracy vẫn chấm đủ 3 dòng).
        out_nan = ev.classification_metrics(actual, prob, baseline_is_up=np.array([1.0, np.nan, 0.0, 0.0]))
        assert out_nan["cls_base_rate_majority"] == pytest.approx(0.5)  # hits: [đúng, sai] trên 2 dòng
        assert out_nan["cls_accuracy"] == pytest.approx(1.0 / 3.0)

        # baseline sai độ dài → lỗi; truyền cả train_is_up lẫn baseline_is_up → lỗi mơ hồ.
        with pytest.raises(ValueError, match="baseline"):
            ev.classification_metrics(actual, prob, baseline_is_up=np.array([1.0, 1.0]))
        with pytest.raises(ValueError, match="MỘT trong hai"):
            ev.classification_metrics(
                actual, prob, np.array([1.0, 0.0, 1.0, 0.0]), baseline_is_up=np.array([1.0, 1.0, 1.0, 1.0])
            )
        # Không truyền baseline nào → base rate None (giữ hợp đồng cũ).
        out_none = ev.classification_metrics(actual, prob)
        assert out_none["cls_base_rate_majority"] is None
        assert out_none["cls_edge_vs_base_rate"] is None
