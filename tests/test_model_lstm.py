"""Kiểm thử src/models/lstm_model.py — nhanh, dữ liệu nhỏ, LSTM 1-2 epoch.

Toàn bộ OFFLINE, self-contained (không phụ thuộc fixture ngoài — conftest.py
đang là stub). Tham số test thu nhỏ: window=20 (mặc định sản xuất là 60),
hidden_size=8, max_epochs=1-2 → mỗi test dưới vài giây, cả file thuộc ngân
sách < 60 giây của cả bộ.

Các nhóm kiểm chứng:
1. Hợp đồng engine: chạy qua walk_forward THẬT (src/evaluation) với val nội
   bộ — ModelPrediction đủ độ dài, prob trong [0,1], metrics không rỗng.
2. Chống leakage (probe chạy được, không phải comment): sửa/cắt các dòng
   TƯƠNG LAI của khung predict không đổi dự báo các dòng trước; predict bị
   TỪ CHỐI khi còn cột target; scaler fit đúng bằng mean/std của CHỈ tập
   train và predict không đổi scaler.
3. Chính sách NaN: cửa sổ chứa NaN → NaN đúng các dòng chịu ảnh hưởng
   (không ffill, không bịa).
4. Hợp đồng tham số: from_config đọc config.models.lstm (không sửa yaml,
   đối chiếu cả config.yaml THẬT của dự án), khoá lạ → lỗi; lỗi sớm khi
   thiếu target, thiếu lịch sử, index bị shuffle, khung test rỗng.
5. Hành vi học: history ghi nhận; cùng seed → cùng dự báo; fit không sửa
   khung đầu vào; early stopping chạy với val nội bộ.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.labeling import build_targets
from src.models.lstm_model import DEFAULT_PARAMS, LstmModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

#: Horizon dùng chung cho khung giả lập (engine chạy MỘT horizon/lần).
H = 1
RET_COL = f"target_ret_{H}d"
DIR_COL = f"target_dir_{H}d"
FEATURES = ["f_ret", "f_ma3", "f_vol"]


def make_frame(n: int = 150, seed: int = 11, band: float = 0.3) -> pd.DataFrame:
    """Khung feature + close + target nhỏ: feature causal, warm-up ≤ 5 phiên."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(
        50.0 * np.exp(np.cumsum(rng.normal(0.0005, 0.012, n))), index=idx, name="close"
    )
    frame = pd.DataFrame({"close": close})
    log_close = np.log(close)
    frame["f_ret"] = log_close.diff()
    frame["f_ma3"] = close / close.rolling(3).mean() - 1.0
    frame["f_vol"] = frame["f_ret"].rolling(5).std()
    targets = build_targets(frame[["close"]], horizons=(H,), neutral_band_pct=band)
    return pd.concat([frame, targets], axis=1)


def small_model(**overrides: object) -> LstmModel:
    """LSTM thu nhỏ cho test: window 20, hidden 8, 1-2 epoch — chạy nhanh."""
    params: dict[str, object] = dict(
        window=20,
        hidden_size=8,
        num_layers=1,
        batch_size=32,
        max_epochs=2,
        seed=7,
        horizon=H,
    )
    params.update(overrides)
    return LstmModel(**params)  # type: ignore[arg-type]


def strip_targets(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[RET_COL, DIR_COL])


# ---------------------------------------------------------------------------
# 1. Hợp đồng engine walk-forward
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_walk_forward_end_to_end_with_val(self) -> None:
        """Chạy hết engine thật: 3 fold, val nội bộ, cả hai phía metrics có số."""
        frame = make_frame(n=170)
        config = ev.WalkForwardConfig(
            initial_train_size=100, retrain_every=20, embargo=3, horizon=H, val_size=8
        )
        result = ev.walk_forward(frame, lambda: small_model(), config)

        # test_start = 103/123/143/163 (đều < 170) → 4 fold expanding
        assert result.aggregate["n_folds"] == 4
        assert result.aggregate["reg_n_rows"] > 0
        assert result.aggregate["reg_mae"] is not None
        assert np.isfinite(float(result.aggregate["reg_mae"]))  # type: ignore[arg-type]
        assert result.aggregate["reg_persistence_mae"] is not None
        assert result.aggregate["cls_n_rows"] > 0
        assert result.aggregate["cls_accuracy"] is not None

        prob = result.predictions["prob_up"].to_numpy(dtype="float64")
        prob = prob[~np.isnan(prob)]
        assert prob.size > 0
        assert ((prob >= 0.0) & (prob <= 1.0)).all()

        price = result.predictions["predicted_price"].to_numpy(dtype="float64")
        price = price[~np.isnan(price)]  # engine quy đổi close_t * exp(ret) — phải > 0
        assert price.size > 0 and (price > 0).all()

    def test_predict_output_contract_direct(self) -> None:
        frame = make_frame(n=120)
        train, test = frame.iloc[:100], frame.iloc[100:]
        model = small_model().fit(train, None)
        out = model.predict(strip_targets(test))

        assert isinstance(out, ev.ModelPrediction)
        assert out.predicted_return is not None and len(out.predicted_return) == len(test)
        assert out.probability_up is not None and len(out.probability_up) == len(test)
        assert np.isfinite(out.predicted_return).all()
        assert np.isfinite(out.probability_up).all()
        assert ((out.probability_up >= 0.0) & (out.probability_up <= 1.0)).all()

    def test_predict_before_fit_raises(self) -> None:
        model = small_model()
        with pytest.raises(RuntimeError, match="fit"):
            model.predict(strip_targets(make_frame(n=50)))

    def test_predict_empty_frame_raises(self) -> None:
        frame = make_frame(n=80)
        model = small_model().fit(frame.iloc[:60], None)
        with pytest.raises(ValueError, match="rỗng"):
            model.predict(strip_targets(frame.iloc[60:]).iloc[:0])


# ---------------------------------------------------------------------------
# 2. Chống leakage — probe chạy được
# ---------------------------------------------------------------------------


class TestLeakage:
    def test_future_rows_do_not_affect_past_predictions(self) -> None:
        """(a) Cắt đuôi test và (b) đâm loạn dòng CUỐI — dự báo quá khứ không đổi."""
        frame = make_frame(n=150, seed=3)
        train, test = frame.iloc[:110], frame.iloc[110:]
        model = small_model().fit(train, None)
        test_x = strip_targets(test)
        base = model.predict(test_x)

        # (a) chỉ truyền 7 dòng đầu: dự báo 7 dòng đó phải y hệt
        truncated = model.predict(test_x.iloc[:7])
        np.testing.assert_allclose(
            base.predicted_return[:7], truncated.predicted_return, rtol=1e-6, atol=1e-7
        )
        np.testing.assert_allclose(
            base.probability_up[:7], truncated.probability_up, rtol=1e-6, atol=1e-7
        )

        # (b) sửa feature của dòng CUỐI (và close — không phải feature): chỉ dòng
        # cuối được phép đổi; cửa sổ 20 phiên của các dòng trước không chạm nó.
        perturbed = test_x.copy()
        perturbed.iloc[-1, perturbed.columns.get_loc("f_ret")] = 99.0
        perturbed.iloc[-1, perturbed.columns.get_loc("close")] = 1.0
        after = model.predict(perturbed)
        np.testing.assert_array_equal(base.predicted_return[:-1], after.predicted_return[:-1])
        np.testing.assert_array_equal(base.probability_up[:-1], after.probability_up[:-1])
        assert not np.isclose(base.predicted_return[-1], after.predicted_return[-1])

    def test_predict_rejects_target_columns(self) -> None:
        """Tripwire: khung predict còn cột target là lỗi hợp đồng — raise, không nuốt."""
        frame = make_frame(n=120)
        train, test = frame.iloc[:100], frame.iloc[100:]
        model = small_model().fit(train, None)
        with pytest.raises(ValueError, match="target"):
            model.predict(test)  # chưa strip target

    def test_scaler_fit_on_train_only(self) -> None:
        """Scaler mean/std phải đúng bằng thống kê của CHỈ tập train (ddof=0)."""
        train = make_frame(n=110, seed=5)
        test = make_frame(n=30, seed=23).iloc[10:]  # phân phối khác train (seed khác)
        model = small_model().fit(train, None)

        cols = model.feature_columns_
        raw_train = train[cols].to_numpy(dtype="float64")
        np.testing.assert_allclose(
            model.scaler_mean_, np.nanmean(raw_train, axis=0), rtol=1e-6, atol=1e-12
        )
        # test phân phối lệch hẳn — nếu scaler nhìn thấy test thì mean sẽ khác
        pooled = np.vstack([raw_train, test[cols].to_numpy(dtype="float64") * 5.0 + 0.01])
        assert not np.allclose(model.scaler_mean_, np.nanmean(pooled, axis=0), rtol=1e-3)
        # sau chuẩn hoá, trung bình các giá trị hữu hạn của train ≈ 0
        scaled = model.train_scaled_
        finite = np.isfinite(scaled)
        col_means = np.where(finite, scaled, 0.0).sum(axis=0) / finite.sum(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-6)
        # predict không được refit scaler
        before = model.scaler_mean_.copy()
        model.predict(strip_targets(test))
        np.testing.assert_array_equal(model.scaler_mean_, before)


# ---------------------------------------------------------------------------
# 3. Chính sách NaN
# ---------------------------------------------------------------------------


class TestNanPolicy:
    def test_nan_window_rows_are_skipped(self) -> None:
        """NaN tại 1 dòng test → NaN đúng các dòng có cửa sổ (20 phiên) chạm nó."""
        frame = make_frame(n=140)
        train, test = frame.iloc[:100], frame.iloc[100:]
        model = small_model(window=20).fit(train, None)
        test_x = strip_targets(test).copy()
        test_x.iloc[5, test_x.columns.get_loc("f_ret")] = np.nan

        out = model.predict(test_x)
        ret = out.predicted_return
        assert np.isfinite(ret[4])
        assert np.isnan(ret[5]) and np.isnan(ret[24])
        assert np.isfinite(ret[25])
        assert np.isnan(out.probability_up[5]) and np.isfinite(out.probability_up[4])


# ---------------------------------------------------------------------------
# 4. Hợp đồng tham số và lỗi sớm
# ---------------------------------------------------------------------------


class TestParamsAndErrors:
    def test_from_config_reads_models_lstm(self) -> None:
        model = LstmModel.from_config({"models": {"lstm": {"window": 10, "hidden_size": 4}}})
        assert model.window == 10
        assert model.hidden_size == 4
        assert model.max_epochs == DEFAULT_PARAMS["max_epochs"]  # phần còn lại giữ mặc định

        assert LstmModel.from_config(None).window == DEFAULT_PARAMS["window"]
        assert LstmModel.from_config({}).horizon == DEFAULT_PARAMS["horizon"]
        assert LstmModel.from_config({"models": {}}).hidden_size == DEFAULT_PARAMS["hidden_size"]
        # horizon do tầng pipeline truyền thắng giá trị trong section
        assert LstmModel.from_config({"models": {"lstm": {"horizon": 5}}}, horizon=1).horizon == 1

    def test_from_config_with_real_project_yaml(self) -> None:
        """config.yaml THẬT của dự án chưa có models.lstm → nhận toàn bộ mặc định.

        Chỉ ĐỌC file — chứng minh model không phụ thuộc việc sửa config.yaml.
        """
        with open(PROJECT_ROOT / "config.yaml", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        assert "models" not in config
        model = LstmModel.from_config(config)
        assert model.window == DEFAULT_PARAMS["window"] == 60
        assert model.horizon == DEFAULT_PARAMS["horizon"]

    def test_from_config_rejects_unknown_keys(self) -> None:
        with pytest.raises(ValueError, match="so_layers"):
            LstmModel.from_config({"models": {"lstm": {"so_layers": 3}}})

    def test_fit_requires_target_columns(self) -> None:
        frame = make_frame(n=80)
        with pytest.raises(ValueError, match="target_ret_1d"):
            small_model().fit(frame.drop(columns=[RET_COL]), None)

    def test_fit_raises_when_window_larger_than_history(self) -> None:
        frame = make_frame(n=40)
        with pytest.raises(ValueError, match="window"):
            small_model(window=60).fit(frame, None)

    def test_fit_rejects_shuffled_rows(self) -> None:
        frame = make_frame(n=80)
        with pytest.raises(ValueError, match="tăng dần"):
            small_model().fit(frame.iloc[::-1], None)

    def test_init_rejects_bad_params(self) -> None:
        with pytest.raises(ValueError):
            small_model(window=0)
        with pytest.raises(ValueError):
            small_model(max_epochs=0)
        with pytest.raises(ValueError):
            small_model(dropout=1.0)
        with pytest.raises(ValueError):
            small_model(loss_weight_ret=0.0, loss_weight_cls=0.0)


# ---------------------------------------------------------------------------
# 5. Hành vi học
# ---------------------------------------------------------------------------


class TestLearningBehaviour:
    def test_history_and_determinism_same_seed(self) -> None:
        frame = make_frame(n=140)
        train, test = frame.iloc[:110], strip_targets(frame.iloc[110:])
        first = small_model(max_epochs=2).fit(train, None)
        second = small_model(max_epochs=2).fit(train, None)

        assert first.epochs_trained_ == 2
        assert len(first.history_) == 2
        assert all(np.isfinite(h["train_loss"]) for h in first.history_)

        out1, out2 = first.predict(test), second.predict(test)
        np.testing.assert_allclose(out1.predicted_return, out2.predicted_return, rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(out1.probability_up, out2.probability_up, rtol=1e-5, atol=1e-7)

    def test_fit_does_not_mutate_inputs(self) -> None:
        frame = make_frame(n=90)
        train, val = frame.iloc[:70], frame.iloc[70:80]
        train_before, val_before = train.copy(deep=True), val.copy(deep=True)
        small_model().fit(train, val)
        pd.testing.assert_frame_equal(train, train_before)
        pd.testing.assert_frame_equal(val, val_before)

    def test_early_stopping_runs_with_val(self) -> None:
        frame = make_frame(n=150)
        train, val = frame.iloc[:120], frame.iloc[120:140]
        model = small_model(max_epochs=5, patience=1).fit(train, val)

        assert model.n_val_samples_ > 0
        assert 1 <= model.epochs_trained_ <= 5
        assert model.history_[-1]["val_loss"] is not None
        assert all(np.isfinite(h["val_loss"]) for h in model.history_)
