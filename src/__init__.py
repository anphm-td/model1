"""Gói mã nguồn chính của ứng dụng dự báo cổ phiếu sàn HOSE.

Các thư mục con:
- data_sources: interface PriceDataSource + CafefSource (CAFEF nguồn chính,
  fallback TCBS) + danh sách mã HOSE (cache data/symbols.parquet).
- features: interface FeatureProvider + PriceFeatures + MarketContext.
- models: interface fit/predict chung cho regression & classification.
Các module cấp cao: labeling, evaluation (walk-forward), db (SQLite),
pipeline (run_pipeline), forecast.
"""
