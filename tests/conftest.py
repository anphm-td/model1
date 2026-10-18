"""Fixture dùng chung cho pytest — dữ liệu nhỏ, chạy nhanh.

Trách nhiệm (sẽ hiện thực ở bước triển khai):
- DataFrame giá OHLCV giả lập ngắn (vài chục phiên) với chuỗi thời gian
  cố định, dùng lại cho mọi test.
- Cấu hình config.yaml thu nhỏ cho test (walk-forward ngắn: initial_train_size
  nhỏ, embargo nhỏ) để cả bộ test chạy < 60 giây.
- Không gọi API mạng trong test: dữ liệu giá luôn là fixture cài sẵn.
"""
