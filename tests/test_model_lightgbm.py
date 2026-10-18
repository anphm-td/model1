"""Kiểm thử src/models/lightgbm_model — LightGBM theo hợp đồng engine walk-forward.

Toàn bộ OFFLINE trên khung giá giả lập nhỏ (60 phiên, seed cố định), tham số
LightGBM thu nhỏ — đủ chứng minh hợp đồng mà không kéo thời lượng:

1. Lựa chọn feature: loại giá mức OHLCV (blocklist chung với assemble) và mọi
   cột target; cột không quy được thành số bị bỏ kèm warning.
2. Hợp đồng fit/predict của WalkForwardModel: ModelPrediction đúng trường/độ
   dài; predict trước fit raise; thiếu cột feature khi predict raise; KHÔNG
   sửa khung đầu vào; cột thừa bỏ qua.
3. NaN/±inf gốc: feature NaN/inf không chết model và dự báo đủ mọi dòng (không
   dropna, không fill); dòng target NaN (h phiên cuối "live") bị loại khỏi fit.
4. Chạy QUA ENGINE evaluation.walk_forward (retrain mỗi fold, embargo/purge do
   engine bảo đảm): metrics không None, predicted_price > 0, xác suất [0, 1].
5. Tham số & config: mặc định từ file; config.models.lightgbm ghi đè; khoá
   dành riêng (early_stopping_rounds/shap_max_rows) KHÔNG lọt vào LightGBM.
6. Early stopping trên val nội bộ (đuôi vùng train): best_iteration < n_estimators.
7. Deterministic: fit hai lần → dự báo giống hệt (không shuffle, seed cố định).
8. Classifier train một lớp → model hằng theo prior + warning (bản thân
   LightGBM sinh xác suất SAI khi thiếu lớp — đo trực tiếp trong môi trường).
9. SHAP: TreeSHAP gốc LightGBM cộng đúng (regressor sum(contrib) == f(x);
   classifier sum(contrib) == raw margin, sigmoid(margin) == predict_proba);
   export CSV reports/{ticker}/shap_importance.csv gộp fold, sắp giảm dần.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.labeling import UP, build_targets
from src.models.lightgbm_model import (
    CLASSIFICATION_PARAMS,
    DEFAULT_OPTIONS,
    DEFAULT_PARAMS,
    SHAP_IMPORTANCE_FILENAME,
    LightGBMClassifierModel,
    LightGBMModel,
    LightGBMRegressorModel,
    export_shap_importance,
)

#: Horizon của khung giả lập (engine chạy 1 horizon/lần; config thật là [1, 5]).
H = 2
RET_COL = f"target_ret_{H}d"
DIR_COL = f"target_dir_{H}d"
#: Tham số thu nhỏ cho test: cây thật (có split) nhưng nhẹ — mỗi fit < 0.1s.
FAST_PARAMS = {"n_estimators": 40, "learning_rate": 0.1, "min_child_samples": 4}


def make_frame(n: int = 60, seed: int = 7, band: float = 0.3) -> pd.DataFrame:
    """Khung close + feature + target giả lập (đúng chuẩn labeling/build_targets).

    ``f_gap`` mô phỏng warm-up NaN của PriceFeatures (7 dòng đầu chưa đủ cửa sổ).
    """
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.012, n))), index=idx, name="close")
    frame = pd.DataFrame({"close": close})
    ret1 = close.pct_change()
    frame["f_mom"] = ret1.fillna(0.0)
    frame["f_ma3"] = ret1.rolling(3, min_periods=1).mean().fillna(0.0)
    frame["f_gap"] = ret1.rolling(7).sum()  # warm-up NaN như PriceFeatures
    targets = build_targets(frame[["close"]], horizons=(H,), neutral_band_pct=band)
    return pd.concat([frame, targets], axis=1)


def make_up_only_frame(n: int = 45) -> pd.DataFrame:
    """Khung giá TĂNG MẠNH đơn điệu → mọi nhãn hướng là LỚN (train một lớp)."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = pd.Series(50.0 * np.exp(np.linspace(0.0, 1.2, n)), index=idx, name="close")
    frame = pd.DataFrame({"close": close})
    frame["f_mom"] = 0.05
    targets = build_targets(frame[["close"]], horizons=(H,), neutral_band_pct=0.3)
    return pd.concat([frame, targets], axis=1)


def reg_model(**overrides: object) -> LightGBMRegressorModel:
    return LightGBMRegressorModel(horizon=H, **{**FAST_PARAMS, **overrides})  # type: ignore[arg-type]


def cls_model(**overrides: object) -> LightGBMClassifierModel:
    return LightGBMClassifierModel(horizon=H, **{**FAST_PARAMS, **overrides})  # type: ignore[arg-type]


def cfg() -> ev.WalkForwardConfig:
    """Config walk-forward thu nhỏ: train đầu 30, retrain 10, embargo 3, horizon H."""
    return ev.WalkForwardConfig(initial_train_size=30, retrain_every=10, embargo=3, horizon=H)


def stripped(frame: pd.DataFrame) -> pd.DataFrame:
    """Khung như engine đưa vào predict(): đã bỏ cột target."""
    return frame.drop(columns=[c for c in frame.columns if str(c).startswith(("target_", "fwd_", "label_"))])


# ---------------------------------------------------------------------------
# 1. Lựa chọn feature — không giá mức, không target
# ---------------------------------------------------------------------------


class TestFeatureSelection:
    def test_excludes_price_levels_and_targets(self) -> None:
        model = reg_model()
        raw = make_frame().assign(open=51.0, high=52.0, low=49.0, volume=1_000.0)
        model.fit(raw, None)
        assert model.feature_columns_ == ["f_mom", "f_ma3", "f_gap"]  # đúng thứ tự cột
        for excluded in ("close", "open", "high", "low", "volume", RET_COL, DIR_COL):
            assert excluded not in model.feature_columns_, excluded

    def test_non_numeric_column_dropped_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        model = reg_model()
        frame = make_frame().assign(f_text="abc")
        with caplog.at_level(logging.WARNING, logger="src.models.lightgbm_model"):
            model.fit(frame, None)
        assert "f_text" not in model.feature_columns_
        assert any("f_text" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# 2. Hợp đồng fit/predict của WalkForwardModel
# ---------------------------------------------------------------------------


class TestModelContract:
    def test_implements_engine_protocol(self) -> None:
        # isinstance/issubclass trên Protocol không @runtime_checkable bị cấm ở
        # runtime — kiểm tra kế thừa tường minh (hiện thực hợp đồng) qua __mro__.
        assert ev.WalkForwardModel in LightGBMModel.__mro__
        assert ev.WalkForwardModel in LightGBMRegressorModel.__mro__
        assert ev.WalkForwardModel in LightGBMClassifierModel.__mro__

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            reg_model().predict(stripped(make_frame()))

    def test_predict_missing_feature_column_raises(self) -> None:
        model = reg_model().fit(make_frame(), None)
        with pytest.raises(ValueError, match="f_gap"):
            model.predict(stripped(make_frame()).drop(columns=["f_gap"]))

    def test_extra_columns_ignored(self) -> None:
        frame = make_frame()
        model = reg_model().fit(frame, None)
        base = model.predict(stripped(frame))
        extra = model.predict(stripped(frame).assign(junk=np.arange(len(frame))))
        np.testing.assert_array_equal(base.predicted_return, extra.predicted_return)

    def test_does_not_mutate_input_frames(self) -> None:
        frame = make_frame()
        train, val, test = frame.iloc[:38].copy(), frame.iloc[38:48].copy(), stripped(frame.iloc[48:]).copy()
        train_snapshot, val_snapshot, test_snapshot = train.copy(), val.copy(), test.copy()
        model = reg_model().fit(train, val)
        model.predict(test)
        pd.testing.assert_frame_equal(train, train_snapshot)
        pd.testing.assert_frame_equal(val, val_snapshot)
        pd.testing.assert_frame_equal(test, test_snapshot)

    def test_missing_target_column_raises(self) -> None:
        with pytest.raises(ValueError, match="build_targets"):
            reg_model().fit(make_frame().drop(columns=[RET_COL]), None)

    def test_all_nan_target_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            reg_model().fit(make_frame().iloc[-H:], None)  # chỉ còn h phiên "live"

    def test_prediction_lengths_match(self) -> None:
        frame = make_frame()
        n = len(frame)
        reg_out = reg_model().fit(frame, None).predict(stripped(frame))
        assert reg_out.predicted_return is not None and len(reg_out.predicted_return) == n
        assert reg_out.probability_up is None  # regressor không tự chế phía khác
        cls_out = cls_model().fit(frame, None).predict(stripped(frame))
        assert cls_out.probability_up is not None and len(cls_out.probability_up) == n
        assert cls_out.predicted_return is None


# ---------------------------------------------------------------------------
# 3. NaN/±inf gốc và dòng target NaN ("live")
# ---------------------------------------------------------------------------


class TestNativeNan:
    def test_nan_inf_features_predict_every_row(self) -> None:
        frame = make_frame()
        test = stripped(frame).copy()
        test.iloc[3, test.columns.get_loc("f_mom")] = np.nan
        test.iloc[5, test.columns.get_loc("f_ma3")] = np.inf
        test.iloc[7, test.columns.get_loc("f_gap")] = -np.inf
        for model in (reg_model().fit(frame, None), cls_model().fit(frame, None)):
            out = model.predict(test)
            values = out.predicted_return if out.predicted_return is not None else out.probability_up
            assert values is not None and len(values) == len(test)
            assert np.isfinite(values).all()  # NaN/inf vào → dự báo hợp lệ ra, không rớt dòng

    def test_nan_target_rows_dropped_from_fit_not_filled(self) -> None:
        frame = make_frame()
        for model in (reg_model().fit(frame, None), cls_model().fit(frame, None)):
            # h phiên cuối của khung không có target (NaN) — bị loại, không fill.
            assert model.n_target_rows_used_ == len(frame) - H


# ---------------------------------------------------------------------------
# 4. Chạy qua ENGINE walk_forward — tuân thủ interface của src/evaluation.py
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_regression_through_engine(self) -> None:
        result = ev.walk_forward(make_frame(), lambda: reg_model(), cfg())
        agg = result.aggregate
        assert agg["n_folds"] == 3 and agg["reg_n_rows"] > 0
        assert agg["reg_mae"] is not None and agg["reg_mae"] >= 0.0
        assert agg["reg_persistence_mae"] is not None and agg["reg_persistence_mae"] > 0.0
        assert agg["reg_skill_vs_persistence_pct"] is not None  # dương/âm đều hợp lệ — engine chấm
        # Model trả log-return → engine quy đổi close_t*exp(ret) > 0 (engine tự validate).
        assert (result.predictions["predicted_price"] > 0).all()

    def test_classification_through_engine(self) -> None:
        result = ev.walk_forward(make_frame(), lambda: cls_model(), cfg())
        agg = result.aggregate
        assert agg["cls_n_rows"] > 0
        assert agg["cls_accuracy"] is not None
        assert 0.0 <= agg["cls_accuracy"] <= 1.0
        prob = result.predictions["prob_up"]
        assert prob.notna().all() and ((prob >= 0.0) & (prob <= 1.0)).all()

    def test_expanding_window_and_data_driven_predictions(self) -> None:
        """Cùng factory nhưng mỗi fold fit model riêng trên dữ liệu fold (expanding
        window) — và dự báo phản ánh dữ liệu được cấp, không phải trạng thái nhớ."""
        result = ev.walk_forward(make_frame(seed=7), lambda: reg_model(), cfg())
        summary = result.fold_summary()
        assert (summary["n_train"] == [30, 40, 50]).all()  # expanding window
        # Khung khác seed → dự báo khác (model học từ dữ liệu fold, không nhớ cũ).
        other = ev.walk_forward(make_frame(seed=11), lambda: reg_model(), cfg())
        first_a = result.predictions["predicted_price"].iloc[0]
        first_b = other.predictions["predicted_price"].iloc[0]
        assert first_a != first_b


# ---------------------------------------------------------------------------
# 5. Tham số & config — mặc định trong file, config.models.lightgbm ghi đè
# ---------------------------------------------------------------------------


class TestParamsAndConfig:
    def test_defaults_from_file_when_no_config(self) -> None:
        model = LightGBMRegressorModel(horizon=H)
        assert model.params == DEFAULT_PARAMS
        classifier = LightGBMClassifierModel(horizon=H)
        assert classifier.params == CLASSIFICATION_PARAMS
        assert CLASSIFICATION_PARAMS["objective"] == "binary"
        for m in (model, classifier):
            assert m.early_stopping_rounds == DEFAULT_OPTIONS["early_stopping_rounds"]
            assert m.shap_max_rows == DEFAULT_OPTIONS["shap_max_rows"]

    def test_config_models_section_overrides_defaults(self) -> None:
        config = {
            "models": {
                "lightgbm": {
                    "n_estimators": 7,
                    "min_child_samples": 3,
                    "early_stopping_rounds": 9,
                    "shap_max_rows": 11,
                }
            }
        }
        # Dựng trực tiếp (không qua helper reg_model vì helper đẩy FAST_PARAMS vào
        # constructor-override — override luôn thắng config đúng theo thiết kế).
        model = LightGBMRegressorModel(horizon=H, config=config)
        assert model.params["n_estimators"] == 7
        assert model.params["min_child_samples"] == 3
        assert model.early_stopping_rounds == 9
        assert model.shap_max_rows == 11
        # Khoá dành riêng KHÔNG lọt vào tham số LightGBM (khi đó LightGBM sẽ hiểu nhầm).
        assert "early_stopping_rounds" not in model.params
        assert "shap_max_rows" not in model.params

    def test_config_without_lightgbm_section_uses_defaults(self) -> None:
        model = LightGBMRegressorModel(
            horizon=H, config={"data": {"source": "cafef"}, "walk_forward": {"embargo": 5}}
        )
        assert model.params == DEFAULT_PARAMS

    def test_constructor_overrides_beat_config(self) -> None:
        model = reg_model(config={"models": {"lightgbm": {"n_estimators": 7}}}, n_estimators=3)
        assert model.params["n_estimators"] == 3

    def test_invalid_task_horizon_and_config(self) -> None:
        with pytest.raises(ValueError, match="task"):
            LightGBMModel(task="quant", horizon=H)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="horizon"):
            LightGBMRegressorModel(horizon=0)
        with pytest.raises(ValueError, match="config"):
            LightGBMModel(task="regression", horizon=H, config="config.yaml")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mapping"):
            LightGBMModel(task="regression", horizon=H, config={"models": {"lightgbm": [1]}})

    def test_reserved_keys_do_not_break_lightgbm_fit(self) -> None:
        """early_stopping_rounds/shap_max_rows chỉ là tuỳ chọn wrapper — fit thật phải chạy."""
        model = reg_model(early_stopping_rounds=3, shap_max_rows=10).fit(make_frame().iloc[:40], None)
        assert model.estimator_ is not None


# ---------------------------------------------------------------------------
# 6. Early stopping trên val nội bộ
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    def test_val_df_triggers_early_stopping(self) -> None:
        frame = make_frame()
        model = reg_model(n_estimators=1500, early_stopping_rounds=5).fit(
            frame.iloc[:38], frame.iloc[38:48]
        )
        best = model.estimator_.booster_.best_iteration
        assert 1 <= best < 1500  # dừng TRƯỚC khi hết cây — val được dùng thật

    def test_val_without_valid_target_skips_early_stopping(self, caplog: pytest.LogCaptureFixture) -> None:
        frame = make_frame()
        model = reg_model(n_estimators=20).fit(frame.iloc[:40], frame.iloc[-H:])  # val toàn target NaN
        assert model.estimator_ is not None  # vẫn fit (không early stopping), không rớt
        assert any("early stopping" in record.message for record in caplog.records)

    def test_val_missing_target_raises(self) -> None:
        frame = make_frame()
        val = frame.iloc[38:48].drop(columns=[RET_COL, DIR_COL])
        with pytest.raises(ValueError, match="target"):
            reg_model(n_estimators=10).fit(frame.iloc[:38], val)


# ---------------------------------------------------------------------------
# 7. Deterministic — fit lại cho cùng dự báo (không shuffle, seed cố định)
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_data_same_predictions(self) -> None:
        frame = make_frame()
        test = stripped(frame.iloc[40:])
        first = reg_model().fit(frame.iloc[:40], None).predict(test)
        second = reg_model().fit(frame.iloc[:40], None).predict(test)
        np.testing.assert_array_equal(first.predicted_return, second.predicted_return)


# ---------------------------------------------------------------------------
# 8. Classifier train một lớp → model hằng theo prior
# ---------------------------------------------------------------------------


class TestOneClassFallback:
    def test_up_only_train_predicts_constant_prior_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        frame = make_up_only_frame()
        train = frame.iloc[:40]
        assert (train[DIR_COL].dropna().astype(float) == UP).all()  # tiền đề: train 100% LỚN
        with caplog.at_level(logging.WARNING, logger="src.models.lightgbm_model"):
            model = cls_model().fit(train, None)
        expected_prior = float((train[DIR_COL].dropna().astype(float) == UP).mean())
        assert model.constant_proba_ == pytest.approx(expected_prior)
        assert model.estimator_ is None  # không fit LightGBM trên một lớp
        prob = model.predict(stripped(frame)).probability_up
        assert prob is not None and np.allclose(prob, expected_prior)
        assert any("MỘT lớp" in record.message for record in caplog.records)

    def test_constant_model_has_no_shap(self) -> None:
        model = cls_model().fit(make_up_only_frame().iloc[:40], None)
        with pytest.raises(ValueError, match="hằng"):
            model.shap_importance()
        with pytest.raises(ValueError, match="hằng"):
            export_shap_importance([model], "TCB", reports_dir="reports")


# ---------------------------------------------------------------------------
# 9. SHAP — TreeSHAP gốc LightGBM (không cần package shap) + export CSV
# ---------------------------------------------------------------------------


class TestShap:
    def test_regressor_contrib_sums_to_prediction(self) -> None:
        model = reg_model(n_estimators=30).fit(make_frame(), None)
        X = model._shap_reference_
        contrib = model.estimator_.booster_.predict(X, pred_contrib=True)
        # sum(contrib) == f(x) — tính chất SHAP (cột cuối là base value).
        np.testing.assert_allclose(contrib.sum(axis=1), model.estimator_.predict(X), atol=1e-6)

    def test_classifier_contrib_sums_to_raw_margin(self) -> None:
        model = cls_model(n_estimators=30).fit(make_frame(), None)
        X = model._shap_reference_
        contrib = model.estimator_.booster_.predict(X, pred_contrib=True)
        raw = model.estimator_.booster_.predict(X, raw_score=True)
        np.testing.assert_allclose(contrib.sum(axis=1), raw, atol=1e-6)
        proba = model.estimator_.predict_proba(X)[:, 1]
        np.testing.assert_allclose(expit(raw), proba, atol=1e-9)

    def test_shap_importance_table(self) -> None:
        model = reg_model(n_estimators=30).fit(make_frame(), None)
        table = model.shap_importance()
        assert list(table.columns) == ["feature", "mean_abs_shap"]
        assert set(table["feature"]) == {"f_mom", "f_ma3", "f_gap"}  # đủ 3 feature, không thêm
        assert (table["mean_abs_shap"] >= 0.0).all()
        assert table["mean_abs_shap"].is_monotonic_decreasing  # sắp giảm dần

    def test_shap_importance_with_explicit_X(self) -> None:
        frame = make_frame()
        model = reg_model(n_estimators=30).fit(frame.iloc[:40], None)
        table = model.shap_importance(stripped(frame.iloc[40:]))  # dữ liệu khác mẫu fit
        assert len(table) == len(model.feature_columns_)

    def test_shap_max_rows_subsamples_reference(self) -> None:
        model = reg_model(n_estimators=10, shap_max_rows=5).fit(make_frame(), None)
        assert len(model._shap_reference_) == 5

    def test_export_aggregates_folds_and_sorts(self, tmp_path: Path) -> None:
        frame = make_frame()
        models = [
            reg_model().fit(frame.iloc[:40], None),  # "fold 0"
            reg_model().fit(frame.iloc[20:], None),  # "fold 1"
        ]
        out = export_shap_importance(models, "tcb", reports_dir=tmp_path, fold_ids=[0, 1])
        assert out == tmp_path / "TCB" / SHAP_IMPORTANCE_FILENAME  # ticker chuẩn hoá IN HOA
        table = pd.read_csv(out)
        assert list(table.columns) == ["feature", "mean_abs_shap", "fold_0", "fold_1"]
        assert len(table) == 3
        assert table["mean_abs_shap"].is_monotonic_decreasing
        assert (table[["mean_abs_shap", "fold_0", "fold_1"]] >= 0.0).all().all()
        expected_mean = table[["fold_0", "fold_1"]].mean(axis=1)
        # CSV ghi %.6g (6 chữ số có nghĩa) → mean ghi trước khi làm tròn lệch nhẹ
        # so với mean tính lại từ giá trị đã làm tròn — chấp nhận sai số định dạng.
        np.testing.assert_allclose(table["mean_abs_shap"], expected_mean, rtol=1e-4, atol=1e-9)

    def test_export_errors(self, tmp_path: Path) -> None:
        model = reg_model().fit(make_frame(), None)
        with pytest.raises(ValueError, match="ít nhất"):
            export_shap_importance([], "TCB", reports_dir=tmp_path)
        with pytest.raises(ValueError, match="chưa fit"):
            export_shap_importance([reg_model()], "TCB", reports_dir=tmp_path)
        with pytest.raises(ValueError, match="một id mỗi fold"):
            export_shap_importance([model], "TCB", reports_dir=tmp_path, fold_ids=[0, 1])
