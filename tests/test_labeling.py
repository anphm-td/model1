"""Kiểm thử src/labeling — target dịch đúng chiều, ranh giới neutral đúng.

Trách nhiệm (sẽ hiện thực ở bước triển khai):
- Unit test chứng minh target tại t chỉ dùng giá ở thời điểm > t (dịch đúng
  chiều cho cả horizon 1 và 5; dịch sai chiều phải bị test bắt được).
- Kiểm tra ranh giới nhãn 3 lớp với neutral_band_pct = 0.3: giá trị đúng
  bằng ±0.3% phải rơi vào lớp nào theo quy ước đã ghi tài liệu.
"""
