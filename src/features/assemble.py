"""Ghép các nhóm đặc trưng BẬT trong config thành một ma trận feature duy nhất.

Luồng: ``assemble_features(df, config)`` → đọc ``features.feature_groups`` →
lần lượt gọi ``FeatureProvider.provide(df)`` của từng nhóm bật → JOIN THEO NGÀY
(index phiên giao dịch của mã) → kiểm định hợp lệ → trả DataFrame feature.

Chỗ cắm tương lai: nhóm ``news_sentiment`` (module RAG tin tức) đang tắt trong
config.yaml. Khi bật mà chưa có hiện thực, hàm raise ``NotImplementedError``
với thông điệp rõ (không âm thầm bỏ qua); khi hiện thực xong chỉ cần thêm
provider vào ``build_providers`` — mọi provider đi qua cùng blocklist chống
giá mức và kiểm tra trùng tên cột.

Hợp đồng join:
- Output index TRÙNG với index của ``df`` (các phiên của MÃ); provider khác
  lịch (index thị trường nhiều/thiếu phiên) tự reindex bên trong, phần thiếu
  là NaN — không ffill (xem market_context.py).
- Trùng tên cột giữa hai provider là lỗi cấu hình → raise (không đè im lặng).
- Blocklist chống giá mức/target: mọi cột output bị cấm mang tên OHLCV thô
  hoặc tên kiểu target (``target_*``, ``fwd_*``) — tripwire cho provider mới
  (bài học: fit trên levels → MAPE giả 0.05–0.3%).
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .base import FeatureProvider
from .market_context import MarketContext
from .price_features import PriceFeatures

logger = logging.getLogger(__name__)

#: Nhóm mặc định khi không truyền config (khớp config.yaml hiện tại).
DEFAULT_FEATURE_GROUPS: dict[str, bool] = {
    "price": True,
    "market_context": True,
    "news_sentiment": False,
}

#: Cột bị CẤM làm feature: giá mức OHLCV thô và mọi cột mang màu target.
NON_FEATURE_BLOCKLIST: tuple[str, ...] = (
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
)
#: Tiền tố bị cấm (target đang/tương lai không được lọt vào ma trận feature).
NON_FEATURE_PREFIXES: tuple[str, ...] = ("target_", "fwd_", "label_")


def feature_groups(config: Mapping[str, Any] | None = None) -> dict[str, bool]:
    """Đọc ``features.feature_groups`` từ config (None → mặc định khớp yaml)."""
    groups = dict(DEFAULT_FEATURE_GROUPS)
    if config is not None:
        section = config.get("features")
        if isinstance(section, Mapping):
            declared = section.get("feature_groups")
            if isinstance(declared, Mapping):
                for key, value in declared.items():
                    groups[str(key)] = bool(value)
    return groups


def build_providers(
    groups: Mapping[str, bool],
    index_frame: pd.DataFrame | None = None,
) -> list[FeatureProvider]:
    """Danh sách provider tương ứng các nhóm BẬT trong ``groups``.

    - ``news_sentiment`` bật → NotImplementedError (chỗ cắm module RAG sau này).
    - ``market_context`` bật nhưng ``index_frame`` không truyền → tải VN-Index
      qua ``data_loader.load_vnindex`` (cache-first; nguồn lỗi → frame rỗng,
      MarketContext tự hạ cấp NaN).
    """
    if groups.get("news_sentiment", False):
        raise NotImplementedError(
            "Nhóm feature 'news_sentiment' đang bật trong config nhưng chưa có "
            "hiện thực — đây là chỗ cắm module RAG tin tức sau này. Hãy đặt "
            "features.feature_groups.news_sentiment: false trong config.yaml."
        )
    providers: list[FeatureProvider] = []
    if groups.get("price", False):
        providers.append(PriceFeatures())
    if groups.get("market_context", False):
        if index_frame is None:
            from ..data_loader import load_vnindex  # import muộn: tránh đụng I/O khi import

            index_frame = load_vnindex()
        providers.append(MarketContext(index_ohlcv=index_frame))
    return providers


def assemble_features(
    df: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    index_frame: pd.DataFrame | None = None,
    providers: Sequence[FeatureProvider] | None = None,
) -> pd.DataFrame:
    """Ma trận feature của ``df`` theo các nhóm bật trong ``config``.

    Tham số:
        df: khung OHLCV chuẩn của mã (hợp đồng data_sources).
        config: dict đã parse từ config.yaml (None → nhóm mặc định).
        index_frame: OHLCV VN-Index truyền sẵn (test/offline); bỏ qua thì
            tự tải khi market_context bật.
        providers: ghi đè danh sách provider (dùng trong test); khi truyền,
            ``config``/``index_frame`` chỉ còn ý nghĩa với provider tự đọc.

    Trả DataFrame index bằng index ``df``; trùng cột giữa provider, cột nằm
    trong blocklist giá mức/target, hoặc provider trả sai index → raise.
    """
    if providers is None:
        providers = build_providers(feature_groups(config), index_frame=index_frame)
    if not providers:
        raise ValueError(
            "Không có nhóm feature nào được bật (features.feature_groups) — "
            "bật ít nhất 'price' trong config.yaml."
        )

    parts: list[pd.DataFrame] = []
    for provider in providers:
        frame = provider.provide(df)
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(
                f"Provider '{provider.name}' phải trả DataFrame, nhận {type(frame).__name__}"
            )
        if not frame.index.equals(df.index):
            raise ValueError(
                f"Provider '{provider.name}' trả index khác index đầu vào "
                f"({len(frame)} dòng vs {len(df)}) — hợp đồng provide() bị vi phạm."
            )
        parts.append(frame)

    features = pd.concat(parts, axis=1)
    _validate_no_duplicates(parts, providers)
    _enforce_blocklist(features)
    logger.info(
        "assemble_features: %d feature (%s) cho %d phiên %s→%s",
        features.shape[1],
        ", ".join(p.name for p in providers),
        len(features),
        features.index.min().date(),
        features.index.max().date(),
    )
    return features


def _validate_no_duplicates(
    parts: Sequence[pd.DataFrame], providers: Sequence[FeatureProvider]
) -> None:
    """Hai provider phát cùng tên cột là lỗi cấu hình — raise, không đè."""
    seen: dict[str, str] = {}
    for frame, provider in zip(parts, providers, strict=True):
        for column in frame.columns:
            if column in seen:
                raise ValueError(
                    f"Trùng feature '{column}' giữa provider '{seen[column]}' và "
                    f"'{provider.name}' — đặt tên feature khác nhau giữa các nhóm."
                )
            seen[column] = provider.name


def _enforce_blocklist(features: pd.DataFrame) -> None:
    """Chặn giá mức/target lọt vào ma trận feature (tripwire cho provider mới)."""
    lowered = {str(col).lower(): col for col in features.columns}
    for blocked in NON_FEATURE_BLOCKLIST:
        if blocked in lowered:
            raise ValueError(
                f"Feature '{lowered[blocked]}' nằm trong blocklist giá mức "
                f"{NON_FEATURE_BLOCKLIST} — fit trên levels làm model học 'giá "
                f"đang ở đâu' (MAPE giả); hãy phát dạng ratio/return."
            )
    for column in features.columns:
        name = str(column).lower()
        if name.startswith(NON_FEATURE_PREFIXES):
            raise ValueError(
                f"Feature '{column}' mang tiền tố target {NON_FEATURE_PREFIXES} — "
                "target phải đi qua src/labeling.py, không lọt vào ma trận feature."
            )


__all__ = [
    "DEFAULT_FEATURE_GROUPS",
    "NON_FEATURE_BLOCKLIST",
    "NON_FEATURE_PREFIXES",
    "assemble_features",
    "build_providers",
    "feature_groups",
]
