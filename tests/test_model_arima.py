"""Kiểm thử src/models/arima_model.py — ARIMA trên log-return, nhanh và offline.

Dữ liệu nhỏ giả lập (seed cố định, không mạng): close = 50·exp(cumsum(r_t)),
r_t sinh từ AR(1)/iid noise. Các nhóm kiểm chứng:

1. ADF TRƯỚC KHI FIT: chọn d qua adfuller đúng trên chuỗi I(0)/I(1)/I(2),
   tôn trọng max_d, lịch sử kiểm định được ghi lại.
2. Grid (p,d,q) ≤ 2: phủ đủ lưới, chọn AIC nhỏ nhất, nhận ra tín hiệu AR.
3. Toán học dự báo: drift → pred ≈ h·μ (bắt lỗi quên trừ F(m'), trả level,
   hay trả return/ngày); AR(1) zero-mean → transient tắt dần theo khoảng cách
   tới biên train; hướng = dấu dự báo (prob ∈ {0,1}).
4. CHỐNG LEAKAGE (ưu tiên số 1): predicted_return không đổi khi biến dụng
   dữ liệu vùng test/c embargo; ĐỔI khi biến dụng dữ liệu train; predict()
   không bao giờ nhận cột target (probe trên engine thật).
5. Hợp đồng engine: chạy được qua ev.walk_forward cho h=1 và h=5 (kèm val_df),
   lỗi sớm với thông điệp rõ khi dữ liệu không đạt.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.labeling import build_targets
from src.models.arima_model import ARIMAModel, adf_diff_order, grid_search_arima, log_returns

# ---------------------------------------------------------------------------
# Helpers — dữ liệu giả lập nhỏ, tất định
# ---------------------------------------------------------------------------


def make_returns(n: int, seed: int, mu: float = 0.0, sigma: float = 0.01, phi: float = 0.0) -> np.ndarray:
    """Sinh n log-return: r_t = mu + phi·(r_{t-1} − mu) + ε_t (AR(1) khi phi ≠ 0)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, n)
    out = np.empty(n, dtype="float64")
    prev = mu
    for t in range(n):
        prev = mu + phi * (prev - mu) + eps[t]
        out[t] = prev
    return out


def closes_from_returns(returns: np.ndarray, base: float = 50.0) -> np.ndarray:
    """Close từ chuỗi return: close_0 = base, close_t = close_{t-1}·exp(r_t)."""
    return base * np.exp(np.concatenate([[0.0], np.cumsum(returns)]))


def make_frame(n: int, horizon: int, seed: int, mu: float = 0.0, sigma: float = 0.01, phi: float = 0.0) -> pd.DataFrame:
    """Khung dòng theo phiên (close + 1 feature chỉ-quá-khứ + target chuẩn labeling)."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    close = closes_from_returns(make_returns(n - 1, seed, mu=mu, sigma=sigma, phi=phi))
    frame = pd.DataFrame({"close": close}, index=idx)
    frame["f_mom"] = frame["close"].pct_change().fillna(0.0)  # feature tại t chỉ dùng ≤ t
    targets = build_targets(frame[["close"]], horizons=(horizon,), neutral_band_pct=0.3)
    return pd.concat([frame, targets], axis=1)


def small_frame_for_test_df(n: int, start: str = "2025-06-02") -> pd.DataFrame:
    """Khung test giả cho unit test predict (model chỉ dùng độ dài khung này)."""
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"close": np.full(n, 50.0)}, index=idx)


class ProbeARIMA(ARIMAModel):
    """ARIMAModel ghi lại đầu ra predict và cột được engine đưa vào (probe leakage)."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.predicted_returns: list[np.ndarray] = []
        self.predict_columns: list[list[str]] = []
        self.fit_rows: list[int] = []

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "ProbeARIMA":
        super().fit(train_df, val_df)
        self.fit_rows.append(int(self.n_train_rows_ or 0))
        return self

    def predict(self, test_df: pd.DataFrame) -> Any:
        out = super().predict(test_df)
        self.predicted_returns.append(np.asarray(out.predicted_return, dtype="float64").copy())
        self.predict_columns.append([str(c) for c in test_df.columns])
        return out


def small_cfg(horizon: int, embargo: int = 2, val_size: int = 0) -> ev.WalkForwardConfig:
    """Config engine thu nhỏ: train đầu 40, retrain 10 (nhanh, đủ cho ARIMA nhỏ)."""
    return ev.WalkForwardConfig(initial_train_size=40, retrain_every=10, embargo=embargo, horizon=horizon, val_size=val_size)


def small_model(horizon: int, embargo: int | None = None, **kwargs: Any) -> ARIMAModel:
    """ARIMAModel cấu hình nhỏ cho test: lưới p,q ≤ 1, min 24 phiên.

    ``embargo`` mặc định = horizon (gap tối thiểu mà engine đảm bảo: evaluation.py:235).
    """
    if embargo is None:
        embargo = horizon
    return ARIMAModel(horizon=horizon, embargo=embargo, max_p=1, max_q=1, min_train_rows=24, **kwargs)


# ---------------------------------------------------------------------------
# 1. Kiểm định dừng ADF trước khi fit
# ---------------------------------------------------------------------------


class TestADFSelection:
    def test_white_noise_is_i0(self) -> None:
        """Return iid → dừng ngay: d = 0, bước đầu là chuỗi gốc (n_diffs=0)."""
        d, steps = adf_diff_order(make_returns(400, seed=1, sigma=0.01), max_d=2)
        assert d == 0
        assert len(steps) == 1 and steps[0].n_diffs == 0 and steps[0].stationary
        assert steps[0].p_value <= 0.05

    def test_random_walk_needs_d1(self) -> None:
        """Return là random walk (I(1)): bước 0 chưa dừng, sau 1 lần sai phân dừng."""
        rw = np.cumsum(make_returns(400, seed=2, sigma=0.01))
        d, steps = adf_diff_order(rw, max_d=2)
        assert d == 1
        assert not steps[0].stationary and steps[1].n_diffs == 1 and steps[1].stationary

    def test_i2_needs_d2(self) -> None:
        """I(2) (tích luỹ hai lần): chỉ dừng sau 2 lần sai phân."""
        i2 = np.cumsum(np.cumsum(make_returns(400, seed=3, sigma=0.01)))
        d, steps = adf_diff_order(i2, max_d=2)
        assert d == 2 and len(steps) == 3
        assert [s.stationary for s in steps] == [False, False, True]

    def test_max_d_cap_is_recorded_not_silent(self) -> None:
        """Chạm trần max_d: trả d = max_d dù bước cuối chưa dừng — ghi rõ, không âm thầm."""
        i2 = np.cumsum(np.cumsum(make_returns(400, seed=4, sigma=0.01)))
        d, steps = adf_diff_order(i2, max_d=1)
        assert d == 1 and len(steps) == 2
        assert not steps[-1].stationary

    def test_model_records_adf_history_consistently(self) -> None:
        """fit() chạy ADF trước grid: d_ khớp chạy lại adf_diff_order độc lập trên cùng chuỗi."""
        frame = make_frame(120, horizon=1, seed=5, phi=0.6)
        model = ARIMAModel(horizon=1, max_p=1, max_q=1, min_train_rows=24).fit(frame.iloc[:100])
        expected_d, _ = adf_diff_order(log_returns(frame["close"].to_numpy()[:100]), max_d=2)
        assert model.d_ == expected_d
        assert model.adf_history_[0].n_diffs == 0
        assert model.adf_history_[model.d_].stationary or model.d_ == 2


# ---------------------------------------------------------------------------
# 2. Grid search (p, d, q) nhỏ
# ---------------------------------------------------------------------------


class TestGridSearch:
    def test_grid_covers_all_orders_and_best_is_min_aic(self) -> None:
        """Lưới 3×3 (p,q ≤ 2): mọi order được thử (fit hoặc ghi lỗi), best = min AIC."""
        returns = make_returns(500, seed=6, sigma=0.01)
        order, result, table, errors = grid_search_arima(returns, d=0, max_p=2, max_q=2)
        all_orders = {(p, 0, q) for p in range(3) for q in range(3)}
        assert set(table) | {ast.literal_eval(k) for k in errors} == all_orders
        assert len(table) >= 1
        assert order in table
        assert table[order] == min(table.values())
        assert np.isfinite(float(result.aic))

    def test_grid_respects_max_p_max_q(self) -> None:
        """max_p=1, max_q=0 → chỉ (0,d,0) và (1,d,0) được thử."""
        returns = make_returns(300, seed=7, sigma=0.01)
        _order, _result, table, errors = grid_search_arima(returns, d=0, max_p=1, max_q=0)
        assert set(table) <= {(0, 0, 0), (1, 0, 0)}
        assert all(k.startswith(("(0, 0, 0)", "(1, 0, 0)")) for k in errors)

    def test_grid_detects_ar_signal(self) -> None:
        """AR(1) φ=0.7 rõ ràng: AIC phải chọn order có p ≥ 1 (không chọn thuần mean)."""
        returns = make_returns(600, seed=8, mu=0.0, sigma=0.01, phi=0.7)
        order, _result, _table, _errors = grid_search_arima(returns, d=0, max_p=2, max_q=2)
        assert order[0] >= 1


# ---------------------------------------------------------------------------
# 3. Toán học dự báo và hợp đồng predict
# ---------------------------------------------------------------------------


class TestPredictMath:
    def test_drift_forecast_equals_h_mu(self) -> None:
        """Return ~ N(μ, σ rất nhỏ): pred phải ≈ h·μ với MỌI dòng test.

        Bắt 3 lỗi cổ điển cùng lúc: trả giá mức (≈ log 50), trả return/ngày
        (thiếu ×h), quên trừ F(m') (pred tăng tuyến tính theo vị trí dòng).
        """
        mu, sigma, horizon = 0.002, 1e-4, 5
        train_closes = closes_from_returns(make_returns(249, seed=9, mu=mu, sigma=sigma))
        model = ARIMAModel(horizon=horizon, max_p=2, max_q=2, min_train_rows=30).fit(
            pd.DataFrame({"close": train_closes})
        )
        pred = np.asarray(model.predict(small_frame_for_test_df(12)).predicted_return)
        assert pred.shape == (12,)
        assert np.all(np.isfinite(pred))
        assert float(np.max(np.abs(pred - horizon * mu))) < 2e-3

    def test_ar1_transient_decays_with_distance_to_train_edge(self) -> None:
        """AR(1) zero-mean φ=0.75: |pred| tắt dần khi dòng test xa biên train (hệ quả trung thực).

        Center return theo mẫu để triệt tiêu thành phần drift h·μ̂ (không tắt theo
        khoảng cách) — test cô lập đúng phần transient của dự báo ARMA.
        """
        horizon, n_test = 3, 12
        returns = make_returns(599, seed=10, sigma=0.01, phi=0.75)
        returns -= returns.mean()  # μ̂ = 0 chính xác → pred thuần transient
        train_closes = closes_from_returns(returns)
        model = ARIMAModel(horizon=horizon, max_p=1, max_q=0, min_train_rows=30).fit(
            pd.DataFrame({"close": train_closes})
        )
        assert model.order_[0] == 1  # AIC nhận ra AR(1)
        pred = np.asarray(model.predict(small_frame_for_test_df(n_test)).predicted_return)
        assert float(np.abs(pred[0])) > 1e-6
        assert float(np.abs(pred[-1])) <= 0.25 * float(np.abs(pred[0]))

    def test_direction_follows_forecast_sign(self) -> None:
        """Hướng theo dấu dự báo: drift mạnh dương → prob toàn 1; âm → toàn 0."""
        test_df = small_frame_for_test_df(10)
        for mu, expected in ((0.01, 1.0), (-0.01, 0.0)):
            closes = closes_from_returns(make_returns(299, seed=11, mu=mu, sigma=0.005))
            out = ARIMAModel(horizon=1, max_p=1, max_q=0, min_train_rows=30).fit(
                pd.DataFrame({"close": closes})
            ).predict(test_df)
            prob = np.asarray(out.probability_up)
            ret = np.asarray(out.predicted_return)
            assert np.all(prob == expected)
            assert np.all((ret > 0.0) == (prob == 1.0))

    def test_predict_is_deterministic_and_stateless(self) -> None:
        """Gọi predict hai lần trên cùng model: kết quả giống hệt (không đổi trạng thái)."""
        closes = closes_from_returns(make_returns(199, seed=12, phi=0.5))
        model = ARIMAModel(horizon=2, max_p=1, max_q=1, min_train_rows=24).fit(pd.DataFrame({"close": closes}))
        test_df = small_frame_for_test_df(8)
        first = np.asarray(model.predict(test_df).predicted_return)
        second = np.asarray(model.predict(test_df).predicted_return)
        assert np.array_equal(first, second)

    def test_val_df_concat_equals_full_train_fit(self) -> None:
        """val_df (đuôi train) được ghép: fit(train, val) ≡ fit(toàn vùng train)."""
        closes = closes_from_returns(make_returns(49, seed=13, phi=0.4))
        full = pd.DataFrame({"close": closes})
        m_concat = ARIMAModel(horizon=1, max_p=1, max_q=1, min_train_rows=24).fit(full.iloc[:40], full.iloc[40:])
        m_full = ARIMAModel(horizon=1, max_p=1, max_q=1, min_train_rows=24).fit(full)
        assert m_concat.n_train_rows_ == m_full.n_train_rows_ == 50
        test_df = small_frame_for_test_df(6)
        assert np.array_equal(
            np.asarray(m_concat.predict(test_df).predicted_return),
            np.asarray(m_full.predict(test_df).predicted_return),
        )

    def test_embargo_param_shifts_forecast_alignment(self) -> None:
        """Đổi embargo (khoảng cách biên train→test) → canh bước dự báo khác đi."""
        closes = closes_from_returns(make_returns(599, seed=14, sigma=0.01, phi=0.75))
        test_df = small_frame_for_test_df(10)
        preds = {}
        for gap in (2, 7):
            model = ARIMAModel(horizon=3, embargo=gap, max_p=1, max_q=0, min_train_rows=30).fit(
                pd.DataFrame({"close": closes})
            )
            preds[gap] = np.asarray(model.predict(test_df).predicted_return)
        assert not np.array_equal(preds[2], preds[7])

    def test_nonstationary_returns_use_d1_path(self) -> None:
        """Return là random walk → ADF chọn d=1, trend='n', fit + predict chạy được."""
        rw_returns = np.cumsum(make_returns(249, seed=15, sigma=0.01))
        closes = closes_from_returns(rw_returns)
        model = ARIMAModel(horizon=2, max_p=1, max_q=1, min_train_rows=24).fit(pd.DataFrame({"close": closes}))
        assert model.d_ == 1 and model.trend_ == "n" and model.order_[1] == 1
        out = model.predict(small_frame_for_test_df(6))
        assert np.all(np.isfinite(np.asarray(out.predicted_return)))
        assert set(np.unique(np.asarray(out.probability_up))) <= {0.0, 1.0}


# ---------------------------------------------------------------------------
# 4. CHỐNG LEAKAGE — probe trên dữ liệu thật của engine
# ---------------------------------------------------------------------------


class TestLeakageProbes:
    N = 90

    def run_engine(self, frame: pd.DataFrame, horizon: int, embargo: int) -> list[ProbeARIMA]:
        factory_calls: list[ProbeARIMA] = []

        def factory() -> ProbeARIMA:
            model = ProbeARIMA(horizon=horizon, embargo=embargo, max_p=1, max_q=1, min_train_rows=24)
            factory_calls.append(model)
            return model

        ev.walk_forward(frame, factory, small_cfg(horizon, embargo=embargo))
        return factory_calls

    def test_predict_output_ignores_test_close_values(self) -> None:
        """predict() không đọc cột close của test: thây close test ×3 → đầu ra giống hệt."""
        closes = closes_from_returns(make_returns(199, seed=16, phi=0.5))
        model = ARIMAModel(horizon=2, max_p=1, max_q=1, min_train_rows=24).fit(pd.DataFrame({"close": closes}))
        test_df = small_frame_for_test_df(10)
        out1 = model.predict(test_df)
        tampered = test_df.copy()
        tampered["close"] = tampered["close"] * 3.0
        out2 = model.predict(tampered)
        assert np.array_equal(np.asarray(out1.predicted_return), np.asarray(out2.predicted_return))
        assert np.array_equal(np.asarray(out1.probability_up), np.asarray(out2.probability_up))

    def test_engine_predictions_invariant_to_test_and_gap_data(self) -> None:
        """Probe engine: biến dụng close TỪ vị trí train_end (vùng embargo + test của fold 0)
        → predicted_return của FOLD 0 KHÔNG đổi, dù cột target/feature của các dòng đó
        (được engine đưa vào fit_df) đã thay đổi theo. Model ARIMA chỉ đọc close của vùng
        train 0..39 (chứng minh không leakage ở tầng model). Với expanding window, các dòng
        ≥ 40 trở thành TRAIN của fold sau nên dự báo fold đó CÓ quyền đổi — không assert."""
        horizon, embargo = 1, 2
        frame = make_frame(self.N, horizon, seed=17, phi=0.4)
        baseline = self.run_engine(frame, horizon, embargo)

        tampered = frame.copy()
        tampered.iloc[40:, tampered.columns.get_loc("close")] *= 1.5
        rerun = self.run_engine(tampered, horizon, embargo)

        assert len(baseline) == len(rerun) == 5
        assert len(baseline[0].predicted_returns) == 1
        # Fold 0: train 0..39 không đổi; embargo (40..41) và test (42..51) đã bị biến dụng.
        assert np.array_equal(baseline[0].predicted_returns[0], rerun[0].predicted_returns[0])
        # Đối chiếu ngược: fold 0 của bản gốc và bản biến đổi cùng thấy fit_df giống hệt nhau.
        assert baseline[0].fit_rows == rerun[0].fit_rows == [40]

    def test_engine_predictions_react_to_train_data(self) -> None:
        """Biến dụng close TRONG vùng train → predicted_return PHẢI đổi (model không phải no-op)."""
        horizon, embargo = 1, 2
        frame = make_frame(self.N, horizon, seed=18, phi=0.4)
        baseline = self.run_engine(frame, horizon, embargo)

        tampered = frame.copy()
        tampered.iloc[20, tampered.columns.get_loc("close")] *= 1.25  # dòng 20 nằm trong mọi fold train
        rerun = self.run_engine(tampered, horizon, embargo)

        changed = any(
            not np.array_equal(m1.predicted_returns[0], m2.predicted_returns[0])
            for m1, m2 in zip(baseline, rerun)
        )
        assert changed

    def test_predict_never_receives_target_columns(self) -> None:
        """Engine phải strip cột target trước predict — probe ghi lại cột thực tế nhận được."""
        frame = make_frame(self.N, 1, seed=19, phi=0.4)
        models = self.run_engine(frame, 1, 2)
        assert len(models) == 5
        for model in models:
            for cols in model.predict_columns:
                assert not any(c.lower().startswith(("target_", "fwd_", "label_")) for c in cols)
                assert "close" in cols and "f_mom" in cols


# ---------------------------------------------------------------------------
# 5. Hợp đồng engine và lỗi sớm
# ---------------------------------------------------------------------------


class TestEngineIntegration:
    def test_walk_forward_end_to_end_h1(self) -> None:
        """h=1 chạy trọn walk-forward: đúng số fold/dòng, metrics không rỗng, prob ∈ {0,1}."""
        frame = make_frame(90, horizon=1, seed=20, phi=0.3)
        result = ev.walk_forward(
            frame,
            lambda: small_model(1, embargo=2),
            small_cfg(1, embargo=2),
        )
        assert result.aggregate["n_folds"] == 5
        assert result.aggregate["n_test_rows"] == 90 - 42  # test bắt đầu sau 40 + embargo 2
        assert result.aggregate["reg_mae"] is not None
        assert result.aggregate["reg_persistence_mae"] is not None
        assert result.aggregate["cls_n_rows"] == 90 - 42 - 1  # dòng cuối chưa có kết quả h=1
        prob = result.predictions["prob_up"].dropna().to_numpy(dtype="float64")
        assert set(np.unique(prob)) <= {0.0, 1.0}
        known = result.predictions["prob_up"].notna()
        assert (
            result.predictions.loc[known, "predicted_up"]
            == (result.predictions.loc[known, "prob_up"] >= ev.UP_PROBABILITY_THRESHOLD)
        ).all()

    def test_walk_forward_h5_with_val_split(self) -> None:
        """h=5, embargo 5 ≥ horizon (purge 0) và val_size=5: engine cấp val, model ghép đủ train."""
        frame = make_frame(90, horizon=5, seed=21, phi=0.3)
        calls: list[ProbeARIMA] = []

        def factory() -> ProbeARIMA:
            model = ProbeARIMA(horizon=5, embargo=5, max_p=1, max_q=1, min_train_rows=24)
            calls.append(model)
            return model

        result = ev.walk_forward(frame, factory, small_cfg(5, embargo=5, val_size=5))
        assert result.aggregate["n_folds"] == 5
        assert (result.fold_summary()["purged_rows"] == 0).all()
        # Mỗi fold: fit_df + val_df ghép lại đúng vùng train (40, 50, 60, 70, 80 phiên).
        assert [m.fit_rows[0] for m in calls] == [40, 50, 60, 70, 80]
        assert result.aggregate["reg_mae"] is not None

    def test_return_only_prediction_converted_by_engine(self) -> None:
        """Model chỉ trả predicted_return → engine quy đổi predicted_price = close_t·exp(ret)
        đúng hệ số từng fold (so với đầu ra predict thu được bởi probe)."""
        frame = make_frame(60, horizon=1, seed=22, mu=0.002, sigma=0.005)
        calls: list[ProbeARIMA] = []

        def factory() -> ProbeARIMA:
            model = ProbeARIMA(horizon=1, embargo=2, max_p=1, max_q=1, min_train_rows=24)
            calls.append(model)
            return model

        result = ev.walk_forward(frame, factory, small_cfg(1, embargo=2))
        assert len(calls) == result.aggregate["n_folds"] == 2  # n=60: test [42,52) và [52,60)
        for fold_id, model in enumerate(calls):
            rows = result.predictions[result.predictions["fold_id"] == fold_id]
            implied = np.log(
                rows["predicted_price"].to_numpy(dtype="float64") / rows["close_t"].to_numpy(dtype="float64")
            )
            assert np.allclose(implied, model.predicted_returns[0], rtol=1e-9, atol=0.0)


class TestErrorHandling:
    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(RuntimeError, match="fit"):
            ARIMAModel(horizon=1).predict(small_frame_for_test_df(3))

    def test_constructor_validation(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            ARIMAModel(horizon=0)
        with pytest.raises(ValueError, match="embargo"):
            ARIMAModel(horizon=1, embargo=-1)
        with pytest.raises(ValueError, match="adf_alpha"):
            ARIMAModel(horizon=1, adf_alpha=1.5)

    def test_insufficient_history_raises(self) -> None:
        closes = closes_from_returns(make_returns(14, seed=23))
        with pytest.raises(ValueError, match="phiên train"):
            ARIMAModel(horizon=1, min_train_rows=30).fit(pd.DataFrame({"close": closes}))

    def test_nan_and_nonpositive_close_raise(self) -> None:
        closes = closes_from_returns(make_returns(39, seed=24))
        frame = pd.DataFrame({"close": closes})
        broken = frame.copy()
        broken.iloc[10, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            ARIMAModel(horizon=1, min_train_rows=24).fit(broken)
        zero = frame.copy()
        zero.iloc[5, 0] = 0.0
        with pytest.raises(ValueError, match="> 0"):
            ARIMAModel(horizon=1, min_train_rows=24).fit(zero)

    def test_constant_returns_raise(self) -> None:
        with pytest.raises(ValueError, match="hằng"):
            ARIMAModel(horizon=1, min_train_rows=24).fit(pd.DataFrame({"close": np.full(40, 50.0)}))

    def test_missing_close_column_raises(self) -> None:
        frame = pd.DataFrame({"price": np.linspace(50, 51, 40)})
        with pytest.raises(ValueError, match="close"):
            ARIMAModel(horizon=1, min_train_rows=24).fit(frame)
