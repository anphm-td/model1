"""Kiểm thử src/db — schema SQLite và ghi/đọc runs, predictions, outcomes.

Trách nhiệm (sẽ hiện thực ở bước triển khai):
- Khởi tạo schema tại database tạm (tmp_path), chạy lại không lỗi (idempotent).
- Ghi/đọc một run, prediction, outcome; kiểm tra khoá ngoài và trường thời gian.
- Dùng fixture giá/mô hình giả lập, không gọi mạng.
"""
