"""Kiểm thử src/data_sources — chuẩn hoá cột, cache, báo lỗi.

Trách nhiệm (sẽ hiện thực ở bước triển khai):
- CafefClient chuẩn hoá cột OHLCV đúng (open, high, low, close, volume) và
  thứ tự thời gian tăng dần; fallback TCBS khi CAFEF lỗi/đổi hợp đồng.
- Cache parquet data/raw/{ticker}.parquet được ghi/đọc đúng (pyarrow),
  không gọi API lần hai khi cache còn hiệu lực.
- Khi API lỗi/hợp đồng đổi: phải báo lỗi tổng hợp rõ ràng (liệt kê các đường
  đã thử), và tầng ghép trả khung rỗng kèm cảnh báo thay vì im lặng.
"""
