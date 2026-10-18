"""Kiểm thử src/features + src/labeling — CHỨNG MINH chống leakage (ưu tiên số 1).

Ba lớp bằng chứng, đều chạy offline trên dữ liệu giả lập cố định:
1. Perturbation: sửa giá tại t+k (k>0) → feature tại t KHÔNG đổi (so khớp
   bitwise bằng assert_frame_equal(check_exact=True) — phép tính tại t chỉ
   chạm dữ liệu <= t nên kết quả phải giống hệt).
2. Truncation probe (mô hình probe_lookahead của StockIntel): cắt khung tại
   nhiều điểm trong 1/3 cuối, tính lại feature từ khung đã cắt, so giá trị
   NGAY ĐIỂM CẮT — lệch 1 ulp cũng là feature nhìn thấy tương lai (bắt được
   center=True, shift(-1), bfill...).
3. Kiểm giá trị độc lập: target tại t == log(close[t+h]) − log(close[t]) tính
   lại theo vị trí; đổi giá tại t+3 phải KHÔNG đổi target h=5 tại t (chỉ t và
   t+h được phép ảnh hưởng); dịch ngược chiều (log(close_t/close_{t-h})) bị bắt.

Kèm: hợp đồng assemble (toggle nhóm theo config, join theo ngày, không ffill,
blocklist giá mức/target), chính sách warm-up NaN, và dữ liệu chỉ có close
(open/high/low NaN — lịch sử nguồn HOSE chính thức) không làm chết feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import (
    MarketContext,
    PriceFeatures,
    assemble_features,
    feature_groups,
)
from src.features.assemble import (
    DEFAULT_FEATURE_GROUPS,
    NON_FEATURE_BLOCKLIST,
    NON_FEATURE_PREFIXES,
)
from src.features.base import FeatureProvider, FeatureProviderError
from src.labeling import (
    DOWN,
    NEUTRAL,
    UP,
    build_targets,
    classification_target,
    forward_log_return,
)

# ---------------------------------------------------------------------------
# Fixture OHLCV giả lập — tất định, đủ dài cho warm-up, OHLC nhất quán
# ---------------------------------------------------------------------------

N_SESSIONS = 140


def make_ohlcv(
    n: int = N_SESSIONS,
    start: str = "2024-01-02",
    seed: int = 42,
    close_only_upto: int | None = None,
) -> pd.DataFrame:
    """OHLCV ngẫu nhiên tất định; ``close_only_upto`` đặt open/high/low = NaN
    cho các phiên đầu (mô phỏng lịch sử ghép từ nguồn chỉ có close+volume)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n, name="date")
    log_ret = rng.normal(0.0005, 0.018, size=n)
    close = 100.0 * np.exp(np.cumsum(log_ret))
    open_ = np.roll(close, 1)
    open_[0] = close[0] * (1 - 0.004)
    spread = np.abs(rng.normal(0.008, 0.004, size=n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype("float64")
    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )
    if close_only_upto is not None:
        df.iloc[:close_only_upto, df.columns.get_indexer(["open", "high", "low"])] = np.nan
    return df


@pytest.fixture()
def ohlcv() -> pd.DataFrame:
    return make_ohlcv()


# ---------------------------------------------------------------------------
# 1. Hợp đồng chung của provider
# ---------------------------------------------------------------------------


class TestProviderContract:
    def test_output_index_matches_input(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        assert features.index.equals(ohlcv.index)
        assert len(features) == len(ohlcv)

    def test_input_not_mutated(self, ohlcv: pd.DataFrame) -> None:
        snapshot = ohlcv.copy(deep=True)
        PriceFeatures().provide(ohlcv)
        assert ohlcv.equals(snapshot)

    def test_no_price_levels_in_output(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        assert not set(NON_FEATURE_BLOCKLIST) & set(features.columns)
        assert not any(
            str(col).lower().startswith(NON_FEATURE_PREFIXES) for col in features.columns
        )

    def test_expected_feature_columns_present(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        expected = (
            [f"ret_{k}d" for k in (1, 5, 20)]
            + [f"lag_ret_{k}" for k in range(1, 6)]
            + [f"sma_{n}_ratio" for n in (5, 10, 20, 50)]
            + [f"ema_{n}_ratio" for n in (5, 10, 20, 50)]
            + [
                "dist_52w_high",
                "dist_52w_low",
                "rsi_14",
                "macd_norm",
                "macd_signal_norm",
                "macd_hist_norm",
                "stoch_k_14",
                "stoch_d_14",
                "atr_14_pct",
                "bb_pos_20",
                "bb_width_20",
                "vol_ratio_20",
                "obv_delta_5",
                "obv_delta_20",
                "dow",
                "month",
            ]
        )
        for col in expected:
            assert col in features.columns, f"thiếu feature {col}"

    def test_rejects_bad_input(self) -> None:
        with pytest.raises(FeatureProviderError, match="rỗng"):
            PriceFeatures().provide(pd.DataFrame())
        frame = make_ohlcv(n=30).drop(columns=["close"])
        with pytest.raises(FeatureProviderError, match="close"):
            PriceFeatures().provide(frame)
        shuffled = make_ohlcv(n=30).iloc[::-1]
        with pytest.raises(FeatureProviderError, match="tăng dần"):
            PriceFeatures().provide(shuffled)


# ---------------------------------------------------------------------------
# 2. CHỐNG LEAKAGE — perturbation & truncation probe
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_perturb_future_prices_leaves_past_features_bitwise_identical(
        self, ohlcv: pd.DataFrame
    ) -> None:
        """Sửa TOÀ BỘ giá/khối lượng của 10 phiên cuối → mọi feature của 130
        phiên trước phải giống hệt (bitwise)."""
        n = len(ohlcv)
        tampered = ohlcv.copy()
        tampered.iloc[n - 10 :] = tampered.iloc[n - 10 :] * 1.03
        features = PriceFeatures().provide(ohlcv)
        features_tampered = PriceFeatures().provide(tampered)
        pd.testing.assert_frame_equal(
            features.iloc[: n - 10], features_tampered.iloc[: n - 10], check_exact=True
        )

    def test_perturb_single_next_session_leaves_current_row_identical(
        self, ohlcv: pd.DataFrame
    ) -> None:
        """Sửa đúng MỘT phiên t+1 → dòng feature tại t không đổi."""
        t = len(ohlcv) - 20
        tampered = ohlcv.copy()
        tampered.iloc[t + 1, tampered.columns.get_loc("close")] *= 1.07
        features = PriceFeatures().provide(ohlcv)
        features_tampered = PriceFeatures().provide(tampered)
        pd.testing.assert_frame_equal(
            features.iloc[[t]], features_tampered.iloc[[t]], check_exact=True
        )

    def test_features_do_depend_on_past(self, ohlcv: pd.DataFrame) -> None:
        """Ngược lại với test trên: sửa t-1 phải làm đổi feature tại t — đảm
        bảo các test không-leak không vô nghĩa (feature không phải hằng số)."""
        t = len(ohlcv) - 20
        tampered = ohlcv.copy()
        tampered.iloc[t - 1, tampered.columns.get_loc("close")] *= 1.07
        features = PriceFeatures().provide(ohlcv)
        features_tampered = PriceFeatures().provide(tampered)
        assert not np.isclose(
            features["ret_1d"].iloc[t], features_tampered["ret_1d"].iloc[t]
        )

    @pytest.mark.parametrize("cut", [100, 105, 110, 115, 120, 125, 130, 135])
    def test_truncation_probe_price_features(self, ohlcv: pd.DataFrame, cut: int) -> None:
        """Truncation probe: tính feature trên khung đã cắt tại ``cut`` (chỉ giữ
        dữ liệu <= t) — giá trị tại điểm cắt phải bitwise giống bản full."""
        features_full = PriceFeatures().provide(ohlcv)
        features_trunc = PriceFeatures().provide(ohlcv.iloc[: cut + 1])
        pd.testing.assert_frame_equal(
            features_full.iloc[[cut]], features_trunc.iloc[[-1]], check_exact=True
        )

    def test_market_context_perturb_future_index(self, ohlcv: pd.DataFrame) -> None:
        """Sửa giá VN-Index TƯƠNG LAI (sau phiên t) → feature index tại t không đổi."""
        index_frame = make_ohlcv(n=N_SESSIONS, seed=7)
        t = len(ohlcv) - 20
        tampered_index = index_frame.copy()
        tampered_index.iloc[t + 1 :] = tampered_index.iloc[t + 1 :] * 1.05
        features = MarketContext(index_ohlcv=index_frame).provide(ohlcv)
        features_tampered = MarketContext(index_ohlcv=tampered_index).provide(ohlcv)
        pd.testing.assert_frame_equal(
            features.iloc[: t + 1], features_tampered.iloc[: t + 1], check_exact=True
        )

    def test_market_context_truncation_probe(self, ohlcv: pd.DataFrame) -> None:
        index_frame = make_ohlcv(n=N_SESSIONS, seed=7)
        provider = MarketContext(index_ohlcv=index_frame)
        features_full = provider.provide(ohlcv)
        for cut in (100, 110, 120, 130, 135):
            features_trunc = provider.provide(ohlcv.iloc[: cut + 1])
            pd.testing.assert_frame_equal(
                features_full.iloc[[cut]], features_trunc.iloc[[-1]], check_exact=True
            )


# ---------------------------------------------------------------------------
# 3. Giá trị warm-up & công thức độc lập (feature từ dữ liệu quá khứ đúng nghĩa)
# ---------------------------------------------------------------------------


class TestFeatureValues:
    def test_ret_1d_exact_and_warmup_nan(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        close = ohlcv["close"].to_numpy()
        assert np.isnan(features["ret_1d"].iloc[0])
        expected_1 = np.log(close[1]) - np.log(close[0])
        assert features["ret_1d"].iloc[1] == expected_1
        expected_5 = np.log(close[5]) - np.log(close[0])
        assert features["ret_5d"].iloc[5] == expected_5

    def test_lag_ret_is_past_session_return(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        # lag_ret_3 tại t = log-return của phiên t-3
        t = 60
        close = ohlcv["close"].to_numpy()
        expected = np.log(close[t - 3]) - np.log(close[t - 4])
        assert features["lag_ret_3"].iloc[t] == expected

    def test_sma_ratio_exact_and_warmup(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        close = ohlcv["close"].to_numpy()
        assert np.isnan(features["sma_5_ratio"].iloc[3])
        expected = close[4] / np.mean(close[0:5]) - 1.0
        # np.mean vs rolling-mean của pandas có thể lệch 1 ulp (thứ tự cộng khác
        # nhau) — so sánh tương đối chặt, không phải test chống leakage
        assert features["sma_5_ratio"].iloc[4] == pytest.approx(expected, rel=1e-12)

    def test_rsi_warmup_and_range(self, ohlcv: pd.DataFrame) -> None:
        features = PriceFeatures().provide(ohlcv)
        rsi = features["rsi_14"]
        assert rsi.iloc[:14].isna().all()  # min_periods=14
        valid = rsi.iloc[60:]
        assert valid.between(0.0, 100.0).all()

    def test_dist_52w_needs_full_year(self, ohlcv: pd.DataFrame) -> None:
        # khung 140 phiên < 252 → cột 52 tuần toàn NaN (chính sách min_periods)
        features = PriceFeatures().provide(ohlcv)
        assert features["dist_52w_high"].isna().all()
        assert features["dist_52w_low"].isna().all()

    def test_no_inf_on_flat_segment(self) -> None:
        """Đoạn đứng giá (std=0, dải high/low=0) → 0/0 phải thành NaN, không inf."""
        frame = make_ohlcv(n=60)
        flat_start, flat_end = 25, 45  # 20 phiên đứng giá giữa khung
        price = float(frame["close"].iloc[flat_start])
        frame.iloc[flat_start:flat_end] = price
        frame.iloc[flat_start:flat_end, frame.columns.get_loc("volume")] = 1_000_000.0
        features = PriceFeatures().provide(frame)
        numeric = features.select_dtypes(include=[np.number])
        assert not np.isinf(numeric.to_numpy(dtype="float64")).any()

    def test_close_only_history_degrades_gracefully(self) -> None:
        """Lịch sử chỉ có close (open/high/low NaN — nguồn HOSE chính thức):
        feature từ close vẫn có giá trị; Stochastic (cần high/low) NaN;
        ATR hạ cấp về biên độ close-to-close, không chết."""
        frame = make_ohlcv(n=140, close_only_upto=80)
        features = PriceFeatures().provide(frame)

        # first valid: phiên 80..93 là 14 phiên high/low đủ cửa sổ → dòng 93
        assert features["stoch_k_14"].iloc[: 80 + 13].isna().all()
        assert features["stoch_k_14"].iloc[100:].notna().all()

        atr = features["atr_14_pct"].iloc[20:80]  # vùng chỉ có close
        assert atr.notna().all() and (atr > 0).all()
        # TR trong vùng đó = |close_t − close_{t-1}| → tái lập độc lập cùng công thức
        close = frame["close"]
        tr = (close - close.shift(1)).abs()
        atr_expected = tr.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean() / close
        # chỉ so trong vùng chỉ có close (ngoài đó TR chuẩn dùng high/low)
        pd.testing.assert_series_equal(
            features["atr_14_pct"].iloc[:80],
            atr_expected.iloc[:80],
            check_names=False,
        )

        for col in ("ret_1d", "ret_5d", "rsi_14", "macd_norm", "bb_pos_20"):
            assert features[col].iloc[60:80].notna().all(), col


# ---------------------------------------------------------------------------
# 4. MarketContext — cùng phiên, beta, lệch lịch không ffill
# ---------------------------------------------------------------------------


class TestMarketContext:
    def test_vni_ret_same_session_exact(self, ohlcv: pd.DataFrame) -> None:
        index_frame = make_ohlcv(n=N_SESSIONS, seed=7)
        features = MarketContext(index_ohlcv=index_frame).provide(ohlcv)
        t = 60
        idx_close = index_frame["close"].to_numpy()
        expected = np.log(idx_close[t]) - np.log(idx_close[t - 1])
        assert features["vni_ret_1d"].iloc[t] == expected

    def test_beta_is_one_when_index_is_scaled_ticker(self, ohlcv: pd.DataFrame) -> None:
        """Index = 2× close của mã → cùng returns → beta phải đúng 1."""
        scaled = ohlcv.copy()
        scaled[["open", "high", "low", "close"]] = scaled[["open", "high", "low", "close"]] * 2.0
        features = MarketContext(index_ohlcv=scaled).provide(ohlcv)
        beta_last = features["vni_beta_60"].iloc[-1]
        assert np.isclose(beta_last, 1.0, atol=1e-9)

    def test_beta_is_minus_one_for_inverted_index(self, ohlcv: pd.DataFrame) -> None:
        """Index đi ngược dấu mã từng phiên → beta ≈ −1."""
        log_ret = np.log(ohlcv["close"]).diff().to_numpy()
        inverted_close = 1000.0 * np.exp(np.nancumsum(-np.nan_to_num(log_ret)))
        dates = ohlcv.index
        inverted = pd.DataFrame(
            {
                "open": inverted_close,
                "high": inverted_close,
                "low": inverted_close,
                "close": inverted_close,
                "volume": ohlcv["volume"].to_numpy(),
            },
            index=dates,
        )
        features = MarketContext(index_ohlcv=inverted).provide(ohlcv)
        assert np.isclose(features["vni_beta_60"].iloc[-1], -1.0, atol=1e-6)

    def test_calendar_mismatch_no_ffill(self, ohlcv: pd.DataFrame) -> None:
        """Index thiếu phiên của mã → NaN tại phiên đó (KHÔNG ffill); phiên dư
        của index không tạo dòng trong output (join theo lịch MÃ)."""
        partial_index = ohlcv.copy()
        missing_date = ohlcv.index[50]
        partial_index = partial_index.drop(index=missing_date)  # index nghỉ phiên 50
        extra_date = ohlcv.index[-1] + pd.Timedelta(days=1)  # phiên index dư
        extra_row = partial_index.iloc[[-1]].copy()
        extra_row.index = [extra_date]
        partial_index = pd.concat([partial_index, extra_row])

        features = MarketContext(index_ohlcv=partial_index).provide(ohlcv)
        assert features.index.equals(ohlcv.index)  # không thêm dòng cho phiên dư
        assert np.isnan(features["vni_ret_1d"].loc[missing_date])
        # phiền kề vẫn có giá trị — không phải toàn cột NaN
        assert not np.isnan(features["vni_ret_1d"].iloc[49])
        assert features["vni_ret_1d"].iloc[51:].notna().all()

    def test_short_index_degrades_to_nan(self, ohlcv: pd.DataFrame) -> None:
        """Index chỉ có 20 phiên cuối (như cache VNINDEX ngắn): feature NaN ở
        ngoài vùng giao nhau; beta cần 60 cặp → toàn NaN, không lỗi."""
        short_index = ohlcv.iloc[-20:].copy()
        features = MarketContext(index_ohlcv=short_index).provide(ohlcv)
        assert features["vni_ret_1d"].iloc[:-19].isna().all()
        assert features["vni_ret_1d"].iloc[-19:].notna().all()
        assert features["vni_beta_60"].isna().all()

    def test_empty_index_all_nan_no_crash(self, ohlcv: pd.DataFrame) -> None:
        features = MarketContext(index_ohlcv=pd.DataFrame()).provide(ohlcv)
        assert features["vni_ret_1d"].isna().all()
        assert features["vni_beta_60"].isna().all()
        assert features.index.equals(ohlcv.index)


# ---------------------------------------------------------------------------
# 5. Labeling — target dịch đúng chiều (chứng minh bằng perturbation)
# ---------------------------------------------------------------------------


class TestLabelingTargets:
    def test_regression_target_matches_independent_recomputation(
        self, ohlcv: pd.DataFrame
    ) -> None:
        close = ohlcv["close"].to_numpy()
        for h in (1, 5):
            target = forward_log_return(ohlcv["close"], h)
            expected = np.log(close[h:]) - np.log(close[:-h])  # theo vị trí, không shift
            np.testing.assert_allclose(target.to_numpy()[:-h], expected, rtol=0, atol=0)
            assert target.iloc[-h:].isna().all()  # h phiên cuối chưa biết tương lai

    def test_target_is_forward_not_backward(self, ohlcv: pd.DataFrame) -> None:
        """Dịch ngược chiều (target = log(close_t/close_{t-h})) phải bị test bắt:
        trên chuỗi không đơn điệu hai cách làm cho khác nhau."""
        close = ohlcv["close"]
        target = forward_log_return(close, 1)
        backward = np.log(close) - np.log(close.shift(1))  # cách làm SAI
        both = target.notna() & backward.notna()  # giao 2 vùng hợp lệ: 1..n-2
        assert both.sum() == len(close) - 2
        assert not np.allclose(target[both], backward[both])
        assert target.notna().sum() == len(close) - 1

    def test_target_h5_uses_only_t_and_t_plus_h(self, ohlcv: pd.DataFrame) -> None:
        """Perturbation ma trận cho h=5 tại vị trí t0: chỉ close[t0] và
        close[t0+5] được phép ảnh hưởng target[t0]."""
        h = 5
        t0 = len(ohlcv) - 20
        base = forward_log_return(ohlcv["close"], h)

        def target_after_perturbing(positions: list[int]) -> float:
            frame = ohlcv.copy()
            for pos in positions:
                frame.iloc[pos, frame.columns.get_loc("close")] *= 1.05
            return float(forward_log_return(frame["close"], h).iloc[t0])

        base_value = float(base.iloc[t0])
        # các vị trí KHÔNG được phép ảnh hưởng
        for positions in ([t0 - 1], [t0 + 3], [t0 + 6]):
            assert target_after_perturbing(positions) == base_value, positions
        # các vị trí PHẢI ảnh hưởng
        assert target_after_perturbing([t0]) != base_value
        assert target_after_perturbing([t0 + h]) != base_value

    def test_target_h1_same_property(self, ohlcv: pd.DataFrame) -> None:
        h = 1
        t0 = len(ohlcv) - 20
        base = forward_log_return(ohlcv["close"], h)
        frame = ohlcv.copy()
        frame.iloc[t0 + 2, frame.columns.get_loc("close")] *= 1.05  # ngoài t0, t0+1
        assert float(forward_log_return(frame["close"], h).iloc[t0]) == float(base.iloc[t0])
        frame2 = ohlcv.copy()
        frame2.iloc[t0 + 1, frame2.columns.get_loc("close")] *= 1.05
        assert float(forward_log_return(frame2["close"], h).iloc[t0]) != float(base.iloc[t0])

    def test_classification_three_classes_and_boundaries(self) -> None:
        """Nhãn 3 lớp với band ±0.3%: chuỗi cài tay giá trị kỳ vọng từng bước;
        r = 0 đúng bằng (vùng đóng |r| <= band) → neutral."""
        close = pd.Series(
            [100.0, 100.2, 100.1, 100.13, 101.0, 100.9, 100.0, 99.5, 99.0, 99.0],
            index=pd.bdate_range("2024-01-02", periods=10),
        )
        labels = classification_target(close, horizon=1, neutral_band_pct=0.3)
        expected = [
            NEUTRAL,  # +0.20%  (trong vùng ±0.3%)
            NEUTRAL,  # -0.0998%
            NEUTRAL,  # +0.03%
            UP,       # +0.867% > +0.3%
            NEUTRAL,  # -0.099%
            DOWN,     # -0.892% < -0.3%
            DOWN,     # -0.50%
            DOWN,     # -0.503%
            NEUTRAL,  # 0% đúng bằng → vùng đóng → neutral
            np.nan,   # không biết tương lai
        ]
        for got, want in zip(labels.tolist(), expected, strict=True):
            if want is np.nan:
                assert pd.isna(got)
            else:
                assert got == want

    def test_classification_band_margin(self) -> None:
        # target của chuỗi 2 điểm nằm tại dòng 0 (dòng t, nhìn tới t+1)
        up = classification_target(
            pd.Series([100.0, 100.4]), horizon=1, neutral_band_pct=0.3
        )  # +0.4% > band
        down = classification_target(
            pd.Series([100.0, 99.6]), horizon=1, neutral_band_pct=0.3
        )  # -0.4%
        inside = classification_target(
            pd.Series([100.0, 100.29]), horizon=1, neutral_band_pct=0.3
        )  # +0.29%
        assert int(up.iloc[0]) == UP and pd.isna(up.iloc[1])
        assert int(down.iloc[0]) == DOWN
        assert int(inside.iloc[0]) == NEUTRAL

    def test_classification_h5_last_rows_nan(self, ohlcv: pd.DataFrame) -> None:
        labels = classification_target(ohlcv["close"], horizon=5, neutral_band_pct=0.3)
        assert labels.iloc[-5:].isna().all()
        assert set(labels.dropna().unique()).issubset({-1, 0, 1})

    def test_build_targets_columns_and_defaults(self, ohlcv: pd.DataFrame) -> None:
        targets = build_targets(ohlcv, horizons=(1, 5), neutral_band_pct=0.3)
        assert list(targets.columns) == [
            "target_ret_1d",
            "target_dir_1d",
            "target_ret_5d",
            "target_dir_5d",
        ]
        assert targets.index.equals(ohlcv.index)
        # regression và classification đồng nhất hướng trên cùng dữ liệu
        up_rows = targets["target_dir_1d"] == UP
        assert (targets.loc[up_rows, "target_ret_1d"] > np.log(1.003)).all()
        down_rows = targets["target_dir_1d"] == DOWN
        assert (targets.loc[down_rows, "target_ret_1d"] < np.log(0.997)).all()

    def test_labeling_rejects_bad_input(self) -> None:
        with pytest.raises(ValueError, match="horizon"):
            forward_log_return(pd.Series([1.0, 2.0]), 0)
        with pytest.raises(ValueError, match="NaN"):
            forward_log_return(pd.Series([1.0, np.nan, 2.0]), 1)
        with pytest.raises(ValueError, match="neutral_band_pct"):
            classification_target(pd.Series([1.0, 2.0]), 1, 0.0)


# ---------------------------------------------------------------------------
# 6. Assemble — toggle nhóm theo config, join theo ngày, blocklist
# ---------------------------------------------------------------------------


class _FakeProvider(FeatureProvider):
    """Provider giả để kiểm hợp đồng assemble mà không đụng provider thật."""

    def __init__(self, name: str, columns: dict[str, float] | None = None) -> None:
        self._name = name
        self._columns = columns or {"fake_value": 1.0}

    @property
    def name(self) -> str:
        return self._name

    def provide(self, df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self._columns, index=df.index)


class TestAssemble:
    def test_groups_from_real_config_yaml(self) -> None:
        config_path = Path(__file__).resolve().parents[1] / "config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        groups = feature_groups(config)
        assert groups["price"] is True
        assert groups["market_context"] is True
        assert groups["news_sentiment"] is False  # chỗ cắm module RAG sau này

    def test_defaults_match_config(self) -> None:
        assert feature_groups(None) == DEFAULT_FEATURE_GROUPS

    def test_assemble_joins_enabled_groups_by_date(self, ohlcv: pd.DataFrame) -> None:
        index_frame = make_ohlcv(n=N_SESSIONS, seed=7)
        config = {
            "features": {"feature_groups": {"price": True, "market_context": True}}
        }
        features = assemble_features(ohlcv, config=config, index_frame=index_frame)
        price = PriceFeatures().provide(ohlcv)
        context = MarketContext(index_ohlcv=index_frame).provide(ohlcv)
        assert features.index.equals(ohlcv.index)
        assert list(features.columns) == list(price.columns) + list(context.columns)
        pd.testing.assert_frame_equal(features[price.columns], price)
        pd.testing.assert_frame_equal(features[context.columns], context)

    def test_price_only_when_market_context_disabled(self, ohlcv: pd.DataFrame) -> None:
        config = {"features": {"feature_groups": {"price": True, "market_context": False}}}
        features = assemble_features(ohlcv, config=config)
        assert "vni_ret_1d" not in features.columns
        assert "ret_1d" in features.columns

    def test_market_context_only(self, ohlcv: pd.DataFrame) -> None:
        index_frame = make_ohlcv(n=N_SESSIONS, seed=7)
        config = {"features": {"feature_groups": {"price": False, "market_context": True}}}
        features = assemble_features(ohlcv, config=config, index_frame=index_frame)
        assert list(features.columns) == ["vni_ret_1d", "vni_beta_60"]

    def test_no_group_enabled_raises(self, ohlcv: pd.DataFrame) -> None:
        config = {"features": {"feature_groups": {"price": False, "market_context": False}}}
        with pytest.raises(ValueError, match="feature nào được bật"):
            assemble_features(ohlcv, config=config)

    def test_news_sentiment_enabled_raises_not_implemented(self, ohlcv: pd.DataFrame) -> None:
        config = {"features": {"feature_groups": {"news_sentiment": True}}}
        with pytest.raises(NotImplementedError, match="news_sentiment"):
            assemble_features(ohlcv, config=config)

    def test_blocklist_tripwire_raw_price_level(self, ohlcv: pd.DataFrame) -> None:
        bad = _FakeProvider("bad", columns={"close": 55.0})
        with pytest.raises(ValueError, match="blocklist"):
            assemble_features(ohlcv, providers=[bad])

    def test_blocklist_tripwire_target_like_column(self, ohlcv: pd.DataFrame) -> None:
        bad = _FakeProvider("bad", columns={"target_ret_1d": 0.01})
        with pytest.raises(ValueError, match="tiền tố target"):
            assemble_features(ohlcv, providers=[bad])

    def test_duplicate_column_across_providers_raises(self, ohlcv: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Trùng feature"):
            assemble_features(
                ohlcv,
                providers=[
                    _FakeProvider("a", columns={"same_col": 1.0}),
                    _FakeProvider("b", columns={"same_col": 2.0}),
                ],
            )

    def test_provider_returning_wrong_index_raises(self, ohlcv: pd.DataFrame) -> None:
        shifted = _FakeProvider("shifted")
        original_provide = shifted.provide

        def provide_shifted(df: pd.DataFrame) -> pd.DataFrame:
            return original_provide(df).iloc[1:]

        shifted.provide = provide_shifted  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="index khác"):
            assemble_features(ohlcv, providers=[shifted])
