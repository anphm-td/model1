"""Đặc trưng (features): hợp đồng chung và các nhóm đặc trưng.

Module:
- base: interface FeatureProvider.provide(df) + validate_ohlcv_frame.
- price_features: PriceFeatures — đặc trưng kỹ thuật causal từ OHLCV.
- market_context: MarketContext — VN-Index cùng phiên + beta trượt 60 phiên.
- assemble: ghép các nhóm BẬT trong config (features.feature_groups) thành
  một ma trận feature, kèm blocklist chống giá mức/target.

Ràng buộc xuyên suốt (ưu tiên số 1 — chống leakage): feature tại thời điểm t
chỉ được tính từ dữ liệu có thời điểm <= t; tests/test_features.py chứng minh.
"""
from __future__ import annotations

from .assemble import (
    DEFAULT_FEATURE_GROUPS,
    NON_FEATURE_BLOCKLIST,
    NON_FEATURE_PREFIXES,
    assemble_features,
    build_providers,
    feature_groups,
)
from .base import FeatureProvider, FeatureProviderError, validate_ohlcv_frame
from .market_context import BETA_WINDOW, MarketContext
from .price_features import MA_WINDOWS, YEAR_BARS, PriceFeatures

__all__ = [
    "BETA_WINDOW",
    "DEFAULT_FEATURE_GROUPS",
    "MA_WINDOWS",
    "NON_FEATURE_BLOCKLIST",
    "NON_FEATURE_PREFIXES",
    "YEAR_BARS",
    "FeatureProvider",
    "FeatureProviderError",
    "MarketContext",
    "PriceFeatures",
    "assemble_features",
    "build_providers",
    "feature_groups",
    "validate_ohlcv_frame",
]
