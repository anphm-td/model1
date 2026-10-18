"""Kiểm thử src/models/naive.py — baseline persistence + quy tắc hướng không học.

Toàn bộ OFFLINE trên khung giá giả lập nhỏ (vài chục phiên). Naive là model
stateless, nên trọng tâm kiểm chứng:

1. Hợp đồng interface (src/evaluation.py): fit(train_df, val_df) → self;
   predict(test_df) → ModelPrediction cùng độ dài, thứ tự dòng giữ nguyên;
   predicted_price = close_t (persistence), predicted_return = 0.0;
   probability_up theo direction (constant 0.5 mặc định / last_return).
2. Quy tắc last_return: đúng dấu return 1 phiên gần nhất; hoà (diff = 0) và
   dòng đầu khung (không có phiên trước trong phạm vi model được thấy — vùng
   embargo không bao giờ được truyền cho model) → 0.5.
3. Chống leakage — probe chạy được, không phải comment:
   - perturb giá tương lai trong khung predict → đầu ra tại các dòng trước
     KHÔNG đổi (prediction tại t chỉ đọc close_t và close_{t-1});
   - statelessness: hai model fit trên dữ liệu train KHÁC nhau cho dự báo
     GIỐNG HỆT nhau (fit không mang trạng thái nào sang predict);
   - fit/predict không sửa hai khung truyền vào.
4. Tích hợp engine walk_forward(): naive = persistence ⇒ skill_vs_persistence
   = 0 đúng trên từng fold và aggregate; prob 0.5 nhị phân hoá thành "LỚN"
   toàn bộ (quy ước ngưỡng >= 0.5 của engine) nên accuracy = tỉ lệ LỚN;
   last_return qua engine: đúng 0.5 tại dòng đầu MỖI fold (embargo), các dòng
   còn lại khớp dấu diff của chuỗi close liền kề.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import evaluation as ev
from src.labeling import build_targets
from src.models.naive import NaiveModel

#: Horizon của khung giả lập (engine chạy 1 horizon/lần; config thật là [1, 5]).
H = 2

#: Chuỗi close có đủ cả ba nhánh dấu của last_return: tăng / giảm / hoà.
SIGNS_CLOSE = [10.0, 11.0, 10.5, 10.5, 12.0]
#: prob kỳ vọng: [dòng đầu, tăng, giảm, hoà, tăng]
SIGNS_EXPECTED_PROB = [0.5, 1.0, 0.0, 0.5, 1.0]


def make_frame(n: int = 60, seed: int = 7, band: float = 0.3) -> pd.DataFrame:
    """Khung close + feature giả + target đúng chuẩn labeling (giống test_evaluation)."""
    idx = pd.bdate_range("2024-01-02", periods=n)
    rng = np.random.default_rng(seed)
    close = pd.Series(50.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, n))), index=idx, name="close")
    frame = pd.DataFrame({"close": close})
    frame["f_mom"] = close.pct_change().fillna(0.0)
    targets = build_targets(frame[["close"]], horizons=(H,), neutral_band_pct=band)
    return pd.concat([frame, targets], axis=1)


def make_signs_frame() -> pd.DataFrame:
    """Khung 5 dòng với close = SIGNS_CLOSE (feature giả để giống khung thật)."""
    idx = pd.bdate_range("2024-03-04", periods=len(SIGNS_CLOSE))
    frame = pd.DataFrame({"close": SIGNS_CLOSE}, index=idx)
    frame["f_mom"] = 0.0
    return frame


def cfg(**kwargs: Any) -> ev.WalkForwardConfig:
    """Config walk-forward nhỏ: train đầu 30, retrain 10, embargo 3, horizon 2."""
    return ev.WalkForwardConfig(initial_train_size=30, retrain_every=10, embargo=3, horizon=H, **kwargs)


def fitted_model(model: NaiveModel, train_df: pd.DataFrame) -> NaiveModel:
    assert model.fit(train_df, None) is model
    return model


# ---------------------------------------------------------------------------
# 1. Hợp đồng interface
# ---------------------------------------------------------------------------


class TestContract:
    def test_persistence_price_and_zero_return(self) -> None:
        """Dự báo giá t+h = close_t cho TỪNG dòng, predicted_return = 0.0."""
        frame = make_frame()
        model = fitted_model(NaiveModel(), frame)
        out = model.predict(frame)
        assert isinstance(out, ev.ModelPrediction)
        n = len(frame)
        assert out.predicted_price is not None and out.predicted_return is not None
        assert out.probability_up is not None
        assert out.predicted_price.shape == (n,)
        np.testing.assert_array_equal(out.predicted_price, frame["close"].to_numpy(dtype="float64"))
        np.testing.assert_array_equal(out.predicted_return, np.zeros(n))
        assert (out.predicted_price > 0).all()

    def test_default_probability_up_is_constant_half(self) -> None:
        """Mặc định: xác suất tăng 0.5 trên mọi dòng (không tuyên bố tín hiệu)."""
        frame = make_frame()
        model = fitted_model(NaiveModel(), frame)
        out = model.predict(frame)
        np.testing.assert_array_equal(out.probability_up, np.full(len(frame), 0.5))

    def test_custom_up_probability(self) -> None:
        frame = make_frame(n=20)
        model = fitted_model(NaiveModel(up_probability=0.7), frame)
        np.testing.assert_array_equal(model.predict(frame).probability_up, np.full(len(frame), 0.7))

    def test_fit_requires_frames_and_returns_self(self) -> None:
        model = NaiveModel()
        frame = make_frame()
        assert model.fit(frame, None) is model  # engine bỏ qua giá trị trả về, nhưng hợp đồng là self
        assert model.fit(frame, frame.iloc[-5:].copy()) is model  # val_df nội bộ được chấp nhận
        with pytest.raises(ValueError, match="train_df"):
            model.fit("không phải frame", None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="train_df"):
            model.fit(frame.iloc[:0], None)  # frame rỗng

    def test_predict_before_fit_raises(self) -> None:
        with pytest.raises(ValueError, match="fit"):
            NaiveModel().predict(make_signs_frame())

    def test_fit_and_predict_do_not_mutate_inputs(self) -> None:
        """Hợp đồng engine: model không được sửa hai khung truyền vào."""
        frame = make_frame()
        snapshot = frame.copy(deep=True)
        test_df = frame.iloc[30:].copy()
        test_snapshot = test_df.copy(deep=True)
        model = NaiveModel(direction="last_return")
        model.fit(frame, None)
        model.predict(test_df)
        pd.testing.assert_frame_equal(frame, snapshot)
        pd.testing.assert_frame_equal(test_df, test_snapshot)


# ---------------------------------------------------------------------------
# 2. Quy tắc hướng theo dấu return gần nhất
# ---------------------------------------------------------------------------


class TestLastReturn:
    def test_probability_follows_sign_of_last_return(self) -> None:
        """Tăng → 1.0, giảm → 0.0, hoà → 0.5, dòng đầu khung → 0.5."""
        frame = make_signs_frame()
        model = fitted_model(NaiveModel(direction="last_return"), frame)
        out = model.predict(frame)
        np.testing.assert_array_equal(out.probability_up, SIGNS_EXPECTED_PROB)

    def test_first_row_is_neutral_even_when_train_has_data(self) -> None:
        """Dòng đầu khung predict = 0.5 dù train có dữ liệu: phiên trước nó nằm
        trong vùng embargo không bao giờ được truyền cho model — dùng close cuối
        train sẽ biến return 1 phiên thành return (embargo+1) phiên."""
        frame = make_frame()
        model = fitted_model(NaiveModel(direction="last_return"), frame)
        out = model.predict(frame.iloc[35:])  # test block thứ 2 của engine
        assert out.probability_up[0] == 0.5

    def test_price_side_is_still_persistence(self) -> None:
        frame = make_signs_frame()
        model = fitted_model(NaiveModel(direction="last_return"), frame)
        out = model.predict(frame)
        np.testing.assert_array_equal(out.predicted_price, frame["close"].to_numpy(dtype="float64"))


# ---------------------------------------------------------------------------
# 3. Validation đầu vào
# ---------------------------------------------------------------------------


class TestValidation:
    def test_rejects_unknown_direction_and_bad_probability(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            NaiveModel(direction="martingale")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="up_probability"):
            NaiveModel(up_probability=1.5)
        with pytest.raises(ValueError, match="up_probability"):
            NaiveModel(up_probability=-0.1)

    def test_rejects_missing_or_invalid_close(self) -> None:
        frame = make_signs_frame()
        model = fitted_model(NaiveModel(), frame)
        with pytest.raises(ValueError, match="close"):
            model.predict(frame.drop(columns=["close"]))
        with_nan = frame.copy()
        with_nan.iloc[2, with_nan.columns.get_loc("close")] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            model.predict(with_nan)
        with_nonpositive = frame.copy()
        with_nonpositive.iloc[0, with_nonpositive.columns.get_loc("close")] = 0.0
        with pytest.raises(ValueError, match="> 0"):
            model.predict(with_nonpositive)

    def test_rejects_shuffled_datetime_index(self) -> None:
        """Hợp đồng không shuffle: index ngày lộn xộn phải bị bắt sớm."""
        frame = make_signs_frame().iloc[::-1]
        model = fitted_model(NaiveModel(), frame)
        with pytest.raises(ValueError, match="shuffle"):
            model.predict(frame)


# ---------------------------------------------------------------------------
# 4. Chống leakage — probe chạy được
# ---------------------------------------------------------------------------


class TestNoLookahead:
    def test_perturbing_last_row_does_not_change_earlier_predictions(self) -> None:
        """Giá dòng CUỐI đổi → chỉ dự báo của dòng cuối đổi (persistence đọc
        close_t); probability_up không đổi ở bất kỳ dòng nào (không có phiên sau
        nó để tiêu thụ tín hiệu) — model không nhìn thấy tương lai."""
        model = fitted_model(NaiveModel(direction="last_return"), make_frame())
        reference_df = make_signs_frame()
        flipped = make_signs_frame()
        flipped.iloc[-1, flipped.columns.get_loc("close")] = 99.0

        out_base = model.predict(reference_df)
        out_flip = model.predict(flipped)

        np.testing.assert_array_equal(
            out_base.probability_up, out_flip.probability_up
        )
        np.testing.assert_array_equal(out_base.predicted_price[:-1], out_flip.predicted_price[:-1])
        assert out_base.predicted_price[-1] != out_flip.predicted_price[-1]

    def test_perturbing_middle_row_affects_only_that_row_and_the_next(self) -> None:
        """Prediction tại t chỉ đọc close_t (persistence) và close_{t-1}
        (last_return): lật giá dòng 2 → giá đổi đúng 1 dòng, prob đổi đúng ở
        dòng 3 (nó dùng dòng 2 làm 'phiên trước'), mọi dòng khác bất biến."""
        model = fitted_model(NaiveModel(direction="last_return"), make_frame())
        base_df = make_signs_frame()
        flipped = make_signs_frame()
        flipped.iloc[2, flipped.columns.get_loc("close")] = 3.0  # 10.5 → 3.0

        out_base = model.predict(base_df)
        out_flip = model.predict(flipped)

        price_changed = out_base.predicted_price != out_flip.predicted_price
        np.testing.assert_array_equal(price_changed, [False, False, True, False, False])
        # Dòng 3: trước đó hoà (10.5 vs 10.5 → 0.5), sau khi lật → tăng (→ 1.0).
        np.testing.assert_array_equal(
            out_base.probability_up != out_flip.probability_up,
            [False, False, False, True, False],
        )
        assert out_flip.probability_up[3] == 1.0

    def test_predictions_are_independent_of_train_content(self) -> None:
        """Statelessness: hai model fit trên vùng train KHÁC nhau (thậm chí
        chồng lấn ngược) cho dự báo GIỐNG HỆT trên cùng khung test — fit không
        mang thống kê nào sang predict, nên không thể rò trạng thái fold."""
        frame = make_frame()
        test_df = frame.iloc[33:]
        model_a = fitted_model(NaiveModel(direction="last_return"), frame.iloc[:33])
        model_b = fitted_model(NaiveModel(direction="last_return"), frame.iloc[10:30])
        out_a = model_a.predict(test_df)
        out_b = model_b.predict(test_df)
        np.testing.assert_array_equal(out_a.predicted_price, out_b.predicted_price)
        np.testing.assert_array_equal(out_a.predicted_return, out_b.predicted_return)
        np.testing.assert_array_equal(out_a.probability_up, out_b.probability_up)


# ---------------------------------------------------------------------------
# 5. Tích hợp engine walk_forward()
# ---------------------------------------------------------------------------


class TestWalkForwardIntegration:
    def test_naive_equals_persistence_skill_zero_everywhere(self) -> None:
        """Naive CHÍNH LÀ persistence ⇒ skill_vs_persistence = 0 đúng trên từng
        fold và aggregate — và predicted_price khớp close_t từng dòng test."""
        frame = make_frame()
        result = ev.walk_forward(frame, lambda: NaiveModel(), cfg())
        assert result.aggregate["n_folds"] == 3
        assert result.aggregate["reg_skill_vs_persistence_pct"] == 0.0
        for fold in result.folds:
            assert fold.metrics["reg_skill_vs_persistence_pct"] == 0.0
        predictions = result.predictions
        np.testing.assert_array_equal(
            predictions["predicted_price"].to_numpy(dtype="float64"),
            predictions["close_t"].to_numpy(dtype="float64"),
        )
        np.testing.assert_array_equal(
            predictions["persistence_price"].to_numpy(dtype="float64"),
            predictions["predicted_price"].to_numpy(dtype="float64"),
        )

    def test_constant_half_binarizes_to_always_up(self) -> None:
        """Quy ước engine (ngưỡng >= 0.5): prob 0.5 → dự báo "LỚN" toàn bộ ⇒
        accuracy của baseline = tỉ lệ phiên LỚN (cls_up_rate) — đúng số học,
        không phải trùng hợp."""
        frame = make_frame()
        result = ev.walk_forward(frame, lambda: NaiveModel(), cfg())
        predictions = result.predictions
        assert predictions["predicted_up"].astype(bool).all()
        assert result.aggregate["cls_accuracy"] == result.aggregate["cls_up_rate"]

    def test_last_return_through_engine_first_row_of_each_fold_is_neutral(self) -> None:
        """Qua engine (embargo 3): prob = 0.5 ĐÚNG tại dòng đầu mỗi fold (phiên
        trước nó nằm trong vùng embargo, model không được thấy); các dòng còn
        lại khớp dấu diff close liền kề (test block của engine liền nhau nên
        diff trong bảng predictions là đúng cặp phiên kế tiếp)."""
        frame = make_frame()
        result = ev.walk_forward(
            frame, lambda: NaiveModel(direction="last_return"), cfg()
        )
        predictions = result.predictions
        prob = predictions["prob_up"].to_numpy(dtype="float64")
        close = predictions["close_t"].to_numpy(dtype="float64")

        starts = predictions.drop_duplicates(subset="fold_id", keep="first").index
        start_positions = [predictions.index.get_loc(d) for d in starts]
        assert start_positions == [0, 10, 20]  # retrain_every = 10
        for pos in start_positions:
            assert prob[pos] == 0.5

        diff = np.diff(close, prepend=close[0] - 1.0)  # dòng đầu khung đã khẳng định ở trên
        non_start = np.setdiff1d(np.arange(len(prob)), np.array(start_positions), assume_unique=True)
        assert np.isin(prob[non_start], [0.0, 1.0]).all()
        np.testing.assert_array_equal(prob[non_start], (diff[non_start] > 0).astype("float64"))

    def test_works_with_internal_validation_split(self) -> None:
        """val_size > 0: engine truyền thêm val_df vào fit — naive bỏ qua đúng
        nghĩa (không học gì) và kết quả không đổi."""
        frame = make_frame()
        result = ev.walk_forward(frame, lambda: NaiveModel(), cfg(val_size=5))
        assert result.aggregate["reg_skill_vs_persistence_pct"] == 0.0
        assert result.aggregate["n_test_rows"] == 27  # 60 - (30 + 3) dòng test

    def test_error_when_frame_lacks_close_column(self) -> None:
        """Engine strips cột target nhưng GIỮ cột close; nếu khung thiếu close,
        lỗi phải ra sớm với thông điệp rõ (không lệch im lặng)."""
        frame = make_frame().drop(columns=["close"])
        with pytest.raises(ValueError, match="close"):
            ev.walk_forward(frame, lambda: NaiveModel(), cfg())
