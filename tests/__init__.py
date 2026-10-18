"""Bộ kiểm thử pytest của dự án.

Nguyên tắc chung:
- Test nhanh cả bộ < 60 giây: dùng fixture nhỏ (conftest.py), LSTM 1-2 epoch.
- Ưu tiên số 1 là unit test chứng minh chống leakage (test_features.py,
  test_labeling.py).
"""
