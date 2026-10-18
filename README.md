# Dự báo cổ phiếu sàn HOSE

Ứng dụng web (Streamlit) + CLI để **duyệt mã cổ phiếu sàn HOSE, thêm vào danh mục theo dõi và huấn luyện & dự báo giá cho từng mã**, với đánh giá walk-forward trung thực: mọi model được so trực tiếp với baseline (persistence giá / majority-class hướng) trên **cùng các fold**.

> ⚠️ **KHÔNG PHẢI KHUYẾN NGHỊ ĐẦU TƯ.** Dự án phục vụ học tập và nghiên cứu. Số liệu dự báo có thể sai, mô hình có thể thua cả baseline (và thực tế đang thua — xem [Đọc hiểu số liệu](#đọc-hiểu-số-liệu-skill-vs-persistence-edge-vs-base-rate)). Xem đầy đủ ở [Cảnh báo](#cảnh-báo--không-phải-khuyến-nghị-đầu-tư).

## Phạm vi đã kiểm chứng

Tài liệu này mô tả trung thực những gì **đã được chạy và kiểm chứng**, và những gì chưa:

| Hạng mục | Trạng thái |
|---|---|
| Hệ điều hành / shell | Windows 10.0.26200 x64, chạy lệnh qua **Git Bash** (cú pháp ngoặc kép trong tài liệu cũng tương thích `cmd.exe`, nhưng không từng lệnh đều được chạy thử qua cmd) |
| Python | **Anaconda hệ thống của máy gốc**: `C:/ProgramData/anaconda3/python.exe`, Python 3.13.9 — **không tạo venv** |
| Ngày kiểm chứng | 2026-09-29: `pytest tests/ -q` → **249 passed in 26.34s**; `scripts/smoke_test.py` → thoát 0 (30 phiên TCB 2026-08-14 → 2026-09-29); `streamlit --version` → 1.51.0; run full-config TCB (62 fold × 2 horizon) nằm trong `data/app.db` |
| **Chưa** kiểm chứng | Linux/macOS; **máy khác / môi trường Python sạch**; Python ≠ 3.13; mã ngoài HOSE (HNX/UPCoM); thời gian chính xác từng tác vụ trên máy khác |

Thời gian chạy ghi trong tài liệu (trừ pytest và smoke test) là **số đo ghi nhận từ lần chạy phát triển trên máy gốc** — chỉ dùng để định lượng chờ đợi, không phải cam kết.

---

## Mục đích

- **Duyệt mã HOSE**: bảng 390 mã từ nguồn chính thức, tìm kiếm, thêm vào watchlist (`data/watchlist.json`).
- **Huấn luyện & dự báo từng mã**: một nút bấm chạy trọn đường — tải/cập nhật dữ liệu giá → sinh feature → walk-forward validation cho 4 model × từng horizon (1 và 5 phiên) → retrain toàn bộ khung → dự báo phiên kế (giá + hướng 3 lớp + xác suất tăng) → lưu DB, model, báo cáo.
- **Đo đúng**: walk-forward expanding window với purge + embargo chống leakage của nhãn (ưu tiên số 1 của dự án), scaler fit riêng từng fold, không shuffle, và **bắt buộc đối chiếu baseline** — `skill vs persistence` (MAE) và `edge vs base rate` (hướng).
- **Sống thật với kết quả**: model không thắng persistence thì dashboard báo cảnh báo, không khoe số đẹp. Kết quả dự báo được ghi lại và **tự đối chiếu (settle)** khi giá thực tế về.

## Kiến trúc module

```
┌────────────────────────────── app/dashboard.py — Streamlit, 4 tab ──────────────────────────────┐
│   Tab 1 Thị trường   │   Tab 2 Danh mục   │   Tab 3 Phân tích & dự báo   │   Tab 4 Lịch sử       │
└──────────┬───────────────────────────────────────────────────────────────────▲──────────────────┘
           │  CHỈ gọi tầng src/ (không tự tính feature/model)                  │ đọc qua src/forecast
┌──────────▼───────────────────────────────────────────────────────────────────┴──────────────────┐
│                       src/pipeline.py — run_pipeline(ticker, config)                            │
│  1 settle dự báo cũ → 2 tải giá → 3 feature + target → 4 walk-forward 4 model × horizon          │
│  → 5 retrain full + dự báo phiên kế → 6 ghi SQLite (runs/predictions) → 7 models/ + reports/     │
└──────┬──────────────────┬────────────────────┬────────────────────┬────────────────────────────┘
       ▼                  ▼                    ▼                    ▼
 src/data_loader.py   src/features/       src/labeling.py     src/evaluation.py
 (load_history:      (PriceFeatures,     (forward log-return, walk_forward():
  cache parquet,      MarketContext,      nhãn 3 lớp          expanding 750/21,
  cập nhật tăng dần)  assemble_features)  ±0.3% neutral)      embargo ≥ 5, purge h, không shuffle)
       │                                                       ▲
       ▼                                                       │ hợp đồng fit/predict chung
 src/data_sources/                                    src/models/
 ├ CafefSource (CAFEF PriceHistory, fallback TCBS)    ├ naive.py       (persistence — baseline)
 ├ HoseOfficialSource (api.hsx.vn, full history)      ├ arima_model.py (ADF + grid AIC)
 └ PriorityMergeSource (CAFEF ưu tiên ngày trùng)     ├ lightgbm_model.py (reg + cls + SHAP)
                                                      └ lstm_model.py  (PyTorch CPU, cửa sổ 60)
       │
       ▼
 data/raw/{ticker}.parquet · data/processed/{ticker}.parquet · data/symbols.parquet
 data/app.db (runs, predictions, outcomes) · models/{ticker}/*.pkl + metadata.json
 reports/{ticker}/metrics.csv · backtest_h{h}.png · shap_importance_h{h}.csv
```

| Vùng | Module | Vai trò chính |
|---|---|---|
| Nguồn dữ liệu | `src/data_sources/base.py` | Interface `PriceDataSource` + hợp đồng OHLCV chuẩn: `normalize_ohlcv` (DatetimeIndex ngày, 5 cột float64, sort/dedupe, bỏ phiên close không hợp lệ, **KHÔNG ffill ngày nghỉ lễ**), `FetchOutcome`, `SourceError` |
| | `src/data_sources/cafef_source.py` | `CafefSource` — CAFEF PriceHistory: header `request-with: XMLHttpRequest` + UA trình duyệt, phân trang `PageSize/PageIndex`, retry 3 lần backoff, failover 2 host, parser nhận cả key tiếng Việt lẫn tiếng Anh; bên trong có fallback `TcbsClient`; kèm `HoseSymbolDirectory` — danh sách mã HOSE (scrape nguồn công khai, cache `data/symbols.parquet`) |
| | `src/data_sources/hose_source.py` | `HoseOfficialSource` — API chính thức của SÀN HOSE (`api.hsx.vn`): lịch sử **đầy đủ từ ngày niêm yết** nhưng chỉ có **close + volume** (open/high/low = NaN — không bịa) |
| | `src/data_sources/composite.py` | `PriorityMergeSource` — chuỗi nguồn CAFEF → HOSE chính thức, ngày trùng ưu tiên CAFEF |
| Tải & cache | `src/data_loader.py` | `load_history()` — cache `data/raw` + `data/processed` parquet, **cập nhật tăng dần** (chỉ tải từ phiên cuối cache, overlap 1 phiên); nguồn lỗi → khung rỗng + cảnh báo, không chết pipeline; `load_vnindex()` cho VN-Index |
| Danh mục | `src/watchlist.py` | CRUD `data/watchlist.json` (add/remove/list/contains), validate mã `^[A-Z0-9]{2,10}$`, ghi nguyên tử (tmp + `os.replace`); **file chưa tồn tại → coi như danh mục rỗng** (`watchlist.py:97-99`) |
| Feature | `src/features/` | `FeatureProvider` (interface); `PriceFeatures` — 32 feature **causal, không giá mức** (log-return, lag, close/MA−1, RSI(14), MACD chuẩn hoá, stochastic, ATR/close, Bollinger, OBV-delta, dow/month…); `MarketContext` — `vni_ret_1d`, `vni_beta_60` từ VN-Index; `assemble_features` — join theo ngày phiên, chặn blocklist giá mức (`open/high/low/close/adj_close/volume`) và tiền tố `target_/fwd_/label_` |
| Nhãn | `src/labeling.py` | `build_targets()` — `target_ret_{h}d` = log(close[t+h]/close[t]) (dịch giá **tương lai** về dòng t — đúng chiều duy nhất) + `target_dir_{h}d` ∈ {+1, 0, −1} theo vùng neutral **±0.3%** (|r| ≤ band → neutral); h phiên cuối là NaN |
| Đánh giá | `src/evaluation.py` | Engine `walk_forward()` — fold layout purge + embargo, model mới hoàn toàn mỗi fold, predict không thấy cột target; metrics regression (MAE/RMSE/MAPE + skill) và classification (Accuracy/F1/AUC + edge) |
| Model | `src/models/` | 4 model cùng hợp đồng `WalkForwardModel` (`fit(train_df, val_df)` / `predict(test_df) → ModelPrediction`): naive persistence, ARIMA/SARIMAX trên log-return (ADF + grid AIC), LightGBM regressor + classifier (TreeSHAP gốc LightGBM), LSTM 1 tầng (PyTorch CPU, cửa sổ 60 phiên) |
| Pipeline | `src/pipeline.py` | `run_pipeline(ticker)` — 7 bước ở sơ đồ trên; CLI `python -m src.pipeline`; re-point `target_date` về phiên thực sau ngày lễ (`align_target_dates`) rồi settle lại |
| API đọc | `src/forecast.py` | Dashboard tab 3/4 đọc qua đây: `latest_run`, `latest_predictions`, `settlement_stats`, `forecast_summary` |
| DB | `src/db.py` | SQLite `data/app.db`: `runs`, `predictions`, `outcomes` (foreign keys bật — `db.py:87`); `settle_predictions()` — đối chiếu dự báo chưa settle với giá thực tế (idempotent, hướng so với close cách đúng `horizon` phiên, vùng ±0.3%) |
| App | `app/dashboard.py` | Streamlit 4 tab, tiếng Việt, chỉ gọi tầng `src/` |
| Scripts | `scripts/` | `smoke_test.py` — kiểm chứng hợp đồng CAFEF trực tiếp (tự chứa); `run_e2e.py` — e2e rút gọn (4 fold đầu, LSTM 3 epoch) |
| Tests | `tests/` | **249 test offline** (mạng mock, SQLite/parquet trong tmp) — gồm bộ test chứng minh chống leakage bằng perturbation + truncation probe |

**Lưu ý về nguồn giá (đã xác minh trực tiếp từ máy, 29/09/2026):** CAFEF hiện chỉ phục vụ **~63 phiên gần nhất** (`TotalCount` bị chốt, tham số ngày bị bỏ qua) và fallback TCBS không phân giải DNS từ máy này. Vì vậy chuỗi mặc định của loader là **CAFEF (OHLCV đầy đủ, cửa sổ gần) → API chính thức HOSE (lấp lịch sử cũ, chỉ close + volume)** — phần lịch sử cũ có open/high/low = NaN là **dữ liệu thật, không bịa, không ffill**; lớp feature hạ cấp êm (vd. ATR về biên độ close-to-close, feature quá thưa bị loại với cảnh báo). Đơn vị giá: **nghìn đồng**.

## Cài đặt & chạy

### Phiên bản Python

- **Đã kiểm chứng trên Python 3.13.9**; wheel PyTorch CPU được kiểm chứng là `cp313` (chỉ cài được trên CPython 3.13 — `requirements.txt:27-28`).
- Ràng buộc tối thiểu: **shap yêu cầu Python ≥ 3.12** (`requirements.txt:36`), nhưng phiên bản 3.12 chưa được kiểm chứng với dự án — **khuyến nghị dùng Python 3.13**.

### Cài trên máy gốc (Anaconda đã có sẵn package)

Môi trường chạy là **Anaconda hệ thống** — `C:/ProgramData/anaconda3/python.exe` (ghi rõ trong phần đầu `requirements.txt`); dự án **không tạo venv riêng** (`.gitignore` có mục `.venv/` nhưng toàn bộ kiểm chứng thực hiện trên Anaconda hệ thống).

- **17 package đã có sẵn** trong Anaconda (bản ghi môi trường, đọc trực tiếp từ interpreter ngày 2026-09-29, **không pin lại**): pandas 2.3.3, numpy 2.4.6, scikit-learn 1.9.0, lightgbm 4.7.0, statsmodels 0.14.5, scipy 1.16.3, matplotlib 3.11.0, plotly 6.3.0, streamlit 1.51.0, pyarrow 21.0.0, pytest 8.4.2, PyYAML 6.0.3, requests 2.32.5, beautifulsoup4 4.13.5, lxml 5.3.0, openpyxl 3.1.5.
- **Chỉ thiếu 2 package**, được pin trong `requirements.txt`: `torch==2.14.0+cpu` (từ index CPU chính thức của PyTorch — bản PyPI cho Windows kéo theo CUDA hàng GB) và `shap==0.51.0` (bản mới nhất có wheel cp313 win_amd64; 0.52.0 không có — pip sẽ rơi về build sdist cần compiler).

```text
"C:/ProgramData/anaconda3/python.exe" -m pip install -r requirements.txt
```

> Lệnh trên **chỉ đúng trên máy gốc** (hoặc máy nào đã có sẵn 17 package với phiên bản tương thích). `requirements.txt` cố ý **không** khai báo 17 package này thành dòng cài được — chúng chỉ được ghi nhận trong phần comment. Đây là quyết định thiết kế của dự án (môi trường đích là Anaconda đã có sẵn chúng); dự án không có cơ chế "phê duyệt cài đặt" tự động — nếu bạn cần thêm package cho máy mình, bạn tự quyết định và tự ghi lại, vì môi trường là của người dùng.

### Cài trên máy khác / môi trường sạch

Một Anaconda/Python **sạch không có sẵn** lightgbm/streamlit/plotly/statsmodels/… — chạy đúng lệnh phía trên sẽ chỉ cài 2 package rồi gặp `ModuleNotFoundError` khi chạy app. Trên máy khác, cài **đầy đủ 19 package** (17 bản ghi môi trường + 2 package thiếu) bằng một lệnh:

```text
"C:/ProgramData/anaconda3/python.exe" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu "pandas==2.3.3" "numpy==2.4.6" "scipy==1.16.3" "scikit-learn==1.9.0" "lightgbm==4.7.0" "statsmodels==0.14.5" "matplotlib==3.11.0" "plotly==6.3.0" "streamlit==1.51.0" "pyarrow==21.0.0" "pytest==8.4.2" "PyYAML==6.0.3" "requests==2.32.5" "beautifulsoup4==4.13.5" "lxml==5.3.0" "openpyxl==3.1.5" "torch==2.14.0+cpu" "shap==0.51.0"
```

Lưu ý trung thực: các phiên bản trên là **bản ghi môi trường đã chạy tốt trên máy gốc**; việc cài đặt trên môi trường sạch **chưa được kiểm chứng** (dự án không tạo/cài vào môi trường khác). Sau khi cài xong, hãy chạy lại bộ test (`python -m pytest tests/ -q`) và smoke test để tự kiểm chứng trước khi dùng.

### Kiểm tra nguồn dữ liệu (không cần cài gì thêm ngoài requests)

```text
"C:/ProgramData/anaconda3/python.exe" scripts/smoke_test.py
```

Tải 30 phiên gần nhất của TCB qua CAFEF theo đúng hợp đồng, in 5 dòng đầu; thoát 0 khi OK, thoát 1 kèm thông báo rõ khi lỗi mạng/API (xem [Xử lý sự cố](#xử-lý-sự-cố--faq)). *(Chạy kiểm chứng 2026-09-29: thoát 0, 30 phiên 2026-08-14 → 2026-09-29.)*

### Lần chạy đầu sau khi clone

`data/`, `models/`, `reports/` bị `.gitignore` loại khỏi git — **bản clone sạch không có bất kỳ dữ liệu nào**, mọi thứ tự sinh lại khi chạy:

1. **Tab 1 — Thị trường** (lần đầu mở): app gọi `HoseSymbolDirectory().get()` — tải danh sách mã từ nguồn công khai và **ghi cache `data/symbols.parquet`** (`cafef_source.py:540-545`); kết quả cache trong session 1 giờ (`dashboard.py:147-156`). Nguồn lỗi → trả cache cũ (nếu có) + cảnh báo; không có nguồn gì → bảng rỗng + cảnh báo, không bịa.
2. **Tab 2 — Danh mục**: `data/watchlist.json` chưa tồn tại → coi như **danh mục rỗng**, hiển thị thông báo "Danh mục trống — chuyển tab Thị trường để thêm mã…" (`dashboard.py:440-441`); file tự tạo khi bạn thêm mã đầu tiên.
3. **Tab 3 — Phân tích & dự báo** (lần đầu mở): biểu đồ tự tải lịch sử cho mã đang chọn qua `data_loader.load_history` (cache parquet + cập nhật từ nguồn nếu thiếu — spinner "Đang tải dữ liệu giá…"). Mã chưa train → caption "Mã này CHƯA được huấn luyện — bấm nút để chạy pipeline" (`dashboard.py:540`); các mục SHAP/backtest/bảng dự báo trống tới khi bạn bấm **🚀 Huấn luyện & dự báo**.
4. **Tab 4 — Lịch sử**: trống với thông báo "Chưa có dự báo nào trong DB…" (`dashboard.py:881`) cho tới khi có ít nhất một lần train thành công.

### Chạy dashboard

```text
"C:/ProgramData/anaconda3/python.exe" -m streamlit run app/dashboard.py
```

*(Streamlit 1.51.0 — kiểm chứng bằng `python -m streamlit --version`.)* Mở trình duyệt theo địa chỉ Streamlit in ra, bắt đầu ở **Tab 1 — Thị trường**.

### CLI — huấn luyện không qua giao diện

```text
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline TCB                # 1 mã
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline TCB VCB            # nhiều mã
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline --watchlist        # cả danh mục
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline TCB --refresh      # bỏ cache, tải lại toàn bộ
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline TCB --max-folds 1  # chạy nhanh (kiểm thử)
"C:/ProgramData/anaconda3/python.exe" -m src.pipeline --config khac.yaml # config khác
```

Không truyền mã → chạy cả watchlist; watchlist trống → chạy `default_ticker` trong `config.yaml` (TCB).

**Mã thoát** (`src/pipeline.py:1088-1110`): **0** = mọi mã OK; **1** = có ít nhất 1 mã lỗi (ví dụ 1/10 mã lỗi với `--watchlist` → exit 1, **9 mã còn lại vẫn chạy tới cùng** — lỗi từng mã được in rõ); **2** = lỗi trước vòng lặp (config/watchlist không đọc được).

### E2E rút gọn, test và thời gian chờ tham chiếu

```text
"C:/ProgramData/anaconda3/python.exe" scripts/run_e2e.py TCB     # trọn đường thật, rút gọn
                                                                 # (4 fold đầu, LSTM 3 epoch)
"C:/ProgramData/anaconda3/python.exe" -m pytest tests/ -q        # 249 passed, offline
```

| Tác vụ | Thời gian tham chiếu (máy gốc, CPU-only) |
|---|---|
| `pytest tests/ -q` (toàn bộ, offline) | **26.34s** (đã đo 2026-09-29) |
| `scripts/smoke_test.py` (mạng) | vài giây (30 phiên, 2 trang) |
| `scripts/run_e2e.py TCB` (mạng + 4 fold + retrain full) | **~35 giây** (ghi nhận khi phát triển) |
| Full pipeline 1 mã (62 fold × 2 horizon — bấm 🚀 hoặc CLI không `--max-folds`) | **~18 phút** (ghi nhận khi phát triển, run #3) |

Điểm nút chậm nhất là **walk-forward × 4 model** (LSTM nhiều epoch nhất), không phải tải dữ liệu — lần huấn luyện sau chỉ nhanh hơn chút nhờ cache parquet. Nghĩa là bấm "🚀 Huấn luyện & dự báo" ở Tab 3, hãy chờ `st.status` báo **"Hoàn tất"** (khoảng chục phút với mã đã có cache). Nút không có khoá: trong cùng một tab, Streamlit xử lý tương tác **tuần tự** — bấm lần nữa trong lúc chạy sẽ được xử lý sau khi run hiện tại kết thúc, tức sẽ **huấn luyện lại toàn bộ** và ghi một run mới; không có nút huỷ. Tránh bấm train đồng thời từ 2 tab/session khác nhau (không có khoá ở tầng ứng dụng; SQLite tự khoá ghi khi tranh chấp).

Yêu cầu máy: chỉ cần **CPU thường** — torch dùng bản CPU, các model rất nhỏ (file pickle đo được trên máy gốc: `lstm.pkl` 139 KB, `lightgbm_*.pkl` ~0.9 MB, `arima.pkl` 1.9 MB). RAM chính xác chưa đo; run 62 fold đã hoàn tất trên máy bàn/laptop thông thường.

## Cách dùng 4 tab

### Tab 1 — Thị trường

Bảng **390 mã HOSE** (từ nguồn chính thức, cache `data/symbols.parquet` — đã đếm lại 2026-09-29: đúng 390; nếu nguồn danh sách lỗi sẽ có cảnh báo trung thực thay vì bảng bịa). Tìm kiếm theo mã (nguồn hiện chỉ cung cấp mã, không kèm tên công ty — UI tự ghi chú), phân trang 20 dòng/trang. Bấm **"➕ Thêm vào danh mục"** trên từng dòng để thêm vào watchlist (nút chuyển thành "✓ Đã có" nếu mã đã có). Expander **"➕ Thêm mã thủ công"** cho mã không có trong bảng.

> Ứng dụng nhắm **sàn HOSE**. Ô "thêm thủ công" chỉ validate hình thức mã (`^[A-Z0-9]{2,10}$` — `watchlist.py:33`), nên một mã HNX/UPCoM cũng được lưu vào watchlist, nhưng **dữ liệu giá sẽ không tải được** (nguồn không trả phiên → Tab 3 báo "Không tải được dữ liệu giá", pipeline dừng với `InsufficientHistoryError` vì không đủ 756 phiên). Chưa kiểm chứng riêng với mã HNX/UPCoM cụ thể.

### Tab 2 — Danh mục theo dõi

Watchlist của bạn (`data/watchlist.json`): bảng tổng hợp giá cuối (đọc **chỉ từ cache local** `data/processed/{mã}.parquet` — không gọi mạng từng dòng) + thời điểm huấn luyện gần nhất (từ DB). Danh mục trống → thông báo hướng dẫn thêm mã (`dashboard.py:440-441`). Mỗi dòng có **"📈 Chi tiết"** (chuyển sang Tab 3 với mã đó) và **"🗑 Xoá"** — xoá chỉ bỏ mã khỏi watchlist, **không xoá** dữ liệu giá/DB/model của mã đó (muốn xoá sạch xem [Quản trị dữ liệu](#quản-trị-dữ-liệu)).

### Tab 3 — Phân tích & dự báo

- **Biểu đồ nến + RSI(14) + volume** (plotly): **tự nạp ngay khi mở tab, không cần bấm gì** — qua `load_price()` (cache trong session 1 giờ, TTL 3600 — `dashboard.py:127-140`), bên dưới gọi `data_loader.load_history` (đọc cache parquet và cập nhật tăng dần từ nguồn nếu thiếu phiên). Sau một lần train, cache biểu đồ được xoá để vẽ dữ liệu mới (`dashboard.py:576`). Nếu không tải được giá → lỗi đỏ "Không tải được dữ liệu giá cho {mã}" và dừng tab (`dashboard.py:580-585`).
- Nến chỉ vẽ vùng có đủ OHLC; phần lịch sử xa chỉ có close/volume (từ nguồn HOSE chính thức) sẽ vẽ **đường close** — không bịa nến, không ffill, có chú thích giải thích (`dashboard.py:602-633`).
- **KPI**: giá cuối, Δ 1 phiên, RSI, khối lượng.
- **Nút "🚀 Huấn luyện & dự báo"**: chạy `run_pipeline(ticker)` **chỉ khi bấm** (không chạy khi load trang), hiển thị 6 bước trong `st.status`. Thời gian chờ xem bảng ở mục [E2E rút gọn…](#e2e-rút-gọn-test-và-thời-gian-chờ-tham-chiếu). Lỗi được bóc riêng: thiếu lịch sử (`InsufficientHistoryError`), lỗi pipeline, lỗi không lường trước.
- **Bảng dự báo phiên kế** của 4 model × horizon: giá dự báo, hướng 3 lớp (TĂNG/GIẢM/NEUTRAL theo ±0.3%), P(tăng), giá close hiện tại. Nguồn cột **P(tăng) từng model** — xem mục [P(tăng) lấy từ model nào?](#ptăng-lấy-từ-model-nào) ngay dưới.
- **Bảng walk-forward metrics** (`reports/{ticker}/metrics.csv`): MAE model vs persistence, skill %, accuracy vs base rate, edge, ROC-AUC — kèm **cảnh báo khi không model nào thắng persistence** (trạng thái hiện tại của TCB — xem phần metrics dưới đây).
- **Biểu đồ backtest** (`backtest_h{h}.png`) và **SHAP top-15** (`shap_importance_h{h}.csv`) từ `reports/{ticker}/`.

### Tab 4 — Lịch sử dự đoán ↔ kết quả thực tế

Mọi dự báo đã ghi vào DB kèm kết quả đã đối chiếu: lọc theo mã/model/khoảng ngày `target_date`; bảng `predictions ⟕ outcomes` (lỗi tuyệt đối, đúng/sai hướng); tổng hợp theo model (số đối chiếu, tỉ lệ đúng hướng, MAE trung bình/lớn nhất) + biểu đồ MAE tích luỹ theo model. Nút **"🔄 Đối chiếu dự báo chưa settle với giá thực tế"** chạy `settle_predictions()` — **tự tải giá tăng dần từ nguồn rồi mới đối chiếu** (spinner "Đang tải giá và đối chiếu…" — `dashboard.py:900-908`), ghi `outcomes` (idempotent, an toàn khi bấm nhiều lần).

## Đọc hiểu số liệu: skill vs persistence, edge vs base rate

Hai số **quan trọng nhất** trong bảng metrics (công thức nằm ở `src/evaluation.py:262-302` và `:323-402`):

### 1. `skill vs persistence` (regression, MAE)

```
skill_vs_persistence_pct = 100 × (1 − MAE_model / MAE_persistence)
```

- **Persistence** = baseline "giá sau h phiên = giá hôm nay" (`predicted_price = close_t`) — baseline nền bắt buộc theo `config.yaml: baselines.persistence`.
- **Dương** = model thắng persistence; **0** = hoà; **âm** = thua.
- Persistence được tính trên **cùng các dòng/cùng các fold** với model (không bao giờ so MAE của model trên một tập dòng với persistence của tập dòng khác — lỗi kinh điển làm "skill" giả).
- MAPE thật của dự báo giá cổ phiếu 1 ngày ở mức ~1% (run thật TCB: ≈1.29%). Nếu bạn thấy MAPE 0.05–0.3% "đẹp mơ màng" — đó là dấu hiệu leakage (model nhìn thấy giá tương lai), không phải skill.

### 2. `edge vs base rate` (classification, hướng)

```
edge_vs_base_rate = accuracy − base_rate_majority
```

- **Base rate** = accuracy của baseline **majority-class** — luôn chọn lớp đa số của tập TRAIN, chấm trên cùng window test, không nhìn test (`config.yaml: baselines.majority_class`).
- **Không so accuracy với 50% (coin flip)**: giá có drift, base rate có thể cao hơn 50% nhiều (run thật TCB h=1: base rate 58.5%). Model accuracy 57% nghe "ổn" nhưng là **edge âm −1.5 điểm**.
- Cùng khung nhãn "TĂNG vs không-TĂNG" (nhãn neutral ±0.3% xếp về không-TĂNG) cho model và baseline.

### P(tăng) lấy từ model nào?

Cột P(tăng) trong bảng dự báo Tab 3 có **nguồn khác nhau theo model** (`src/pipeline.py:499-533`) — được báo cáo trung thực, không ép khớp nhau:

| Model | Nguồn cột P(tăng) | Nguồn cột giá dự báo |
|---|---|---|
| naive | **0.5 hằng số** — model tuyên bố KHÔNG có tín hiệu hướng (chế độ `constant` mặc định, `naive.py:14-24`) | `close_t` (persistence) |
| arima | **1.0 nếu dự báo return dương, 0.0 nếu âm** — dấu dự báo, không phải xác suất mượt (`arima_model.py:359-380`) | `close_t · exp(predicted_return)` |
| lightgbm | Xác suất sigmoid **thật** từ classifier riêng (`LightGBMClassifierModel`); giá đến từ regressor riêng (`pipeline.py:516-524`) | `close_t · exp(predicted_return)` |
| lstm | Đầu phân loại sigmoid ∈ [0, 1] của cùng mạng | `close_t · exp(predicted_return)` |

Cột **hướng 3 lớp** (TĂNG/GIẢM/NEUTRAL) thì khác: quy đổi từ `predicted_price` so với `close_t` theo vùng ±0.3% — hai phía này có thể không nhất quán với nhau và được ghi cả hai, không ép khớp (`pipeline.py:37-40`). Dự báo giá không hợp lệ (return NaN) được ghi **NULL** vào DB, không bịa.

### 3. Các số còn lại

- Regression: `reg_mae`, `reg_rmse`, `reg_mape` (đơn vị nghìn đồng, trên các dòng đã có kết quả thực tế).
- Classification: `cls_accuracy`, `cls_f1`, `cls_roc_auc` (AUC = None khi chỉ có 1 lớp trong fold — không bịa).
- Dòng dự báo "live" (chưa tới `target_date`, chưa có giá thực tế) vẫn nằm trong bảng predictions nhưng **bị loại khỏi metrics** — không được đếm như dòng đã chấm.

### Trung thực với kết quả — ví dụ thật

Run full-config gần nhất cho TCB (2056 phiên, 62 fold, 2026-09-29, đọc trực tiếp từ `data/app.db`):

| horizon | naive | arima | lightgbm | lstm |
|---|---|---|---|---|
| h=1 — skill vs persistence | 0.0% (baseline) | −0.4% | −18.5% | −71.0% |
| h=1 — edge vs base rate | −17.1 điểm | 0.0 | −4.5 | −1.5 |
| h=5 — skill vs persistence | 0.0% (baseline) | −1.1% | −26.7% | −30.7% |
| h=5 — edge vs base rate | −1.9 điểm | −0.2 | −0.3 | −5.2 |

**Hiện chưa model nào thắng persistence** — dashboard hiển thị đúng cảnh báo này thay vì khoe số. Đây là kết quả trung thực và cũng khớp tài liệu: lợi nhuận ngày của cổ phiếu gần như white noise, model thật phải chứng minh bằng edge dương **ổn định theo từng năm**, không phải một run đẹp.

Hai pattern cần biết khi đọc số (ghi nhận từ nghiên cứu các dự án tương tự):

- **Edge xấu dần đơn điệu theo horizon** (1 → 5 ngày) là dấu hiệu leakage của purge/embargo, không phải skill; model "5 ngày scoring cao hơn 1 ngày" phải bị nghi ngờ trước.
- **AUC > 0.5 nhưng accuracy không tăng**: AUC 0.52–0.59 có thể là tín hiệu ranking thật bị argmax-threshold phá trên split lệch lớp — không nhất thiết là bug của dự án.

## Walk-forward & chống leakage (ưu tiên số 1)

Cấu hình nằm ở `config.yaml: walk_forward` (engine đọc qua `WalkForwardConfig.from_mapping`, từ chối `shuffle: true`):

- **Expanding window**: train đầu 750 phiên, test block 21 phiên, retrain mỗi 21 phiên, không shuffle ở bất kỳ đâu.
- **Embargo 5 phiên + purge = horizon** giữa train và test — nhãn tại t chứa giá đến t+h, nên train luôn bị cắt h ngày cuối trước test block:

```
train [0 ………………… train_end)  [purge h]  [embargo ≥ 5]  [TEST 21 phiên)
                                ▲ các dòng này không bao giờ đến tay model/scaler
```

- **Ngưỡng phiên tối thiểu**: `initial_train_size + embargo + 1 = 756 phiên` — ít hơn thì pipeline dừng với `InsufficientHistoryError` kèm thông báo cần bao nhiêu phiên (`evaluation.make_folds`). Mã niêm yết gần đây hoặc nguồn trả ít phiên sẽ rơi vào đây.
- **Scaler fit riêng từng fold** — ràng buộc bằng cấu trúc: `model_factory` tạo model mới hoàn toàn mỗi fold, `predict()` nhận khung đã bỏ mọi cột target; có test probe chứng minh scaler mean đúng bằng mean của train (khác mean toàn chuỗi).
- **Chống leakage được chứng minh bằng test chạy được**, không phải comment: perturbation test (sửa giá tương lai → feature quá khứ bitwise không đổi), truncation probe kiểu StockIntel (cắt khung, tính lại feature, so bitwise), probe target (sửa t+3/t+6/t−1 không đổi target h=5 tại t; sửa t/t+5 thì đổi).
- **Feature không chứa giá mức**: blocklist `open/high/low/close/adj_close/volume` và tiền tố `target_/fwd_/label_` bị chặn ngay lúc assemble — fit trên levels sinh MAPE giả.
- **Không ffill ngày nghỉ lễ HOSE**: phiên nghỉ không được "vá" thành ngày return = 0 giả.

## Quy trình sử dụng định kỳ

Không có dịch vụ chạy nền — mọi thứ xảy ra khi bạn bấm nút hoặc gọi CLI. Quy trình gợi ý sau mỗi phiên giao dịch (khi nguồn dữ liệu đã có giá mới):

1. **Đối chiếu dự báo cũ (nhanh, không train lại)**: Tab 4 → bấm **"🔄 Đối chiếu…"** — tự tải giá tăng dần từ nguồn rồi ghi kết quả `outcomes` cho các dự báo chưa settle (idempotent). Chỉ cần thao tác này nếu bạn chỉ muốn cập nhật kết quả.
2. **Dự báo phiên kế**: bấm lại **🚀 Huấn luyện & dự báo** (Tab 3) hoặc chạy CLI `python -m src.pipeline --watchlist` — bước 1/6 của pipeline tự settle trước khi huấn luyện, nên một lần chạy là đủ cả hai việc.
3. **Tự động hoá theo lịch**: dự án **không có scheduler built-in** — dùng Windows Task Scheduler (hoặc cron) gọi CLI, ví dụ Action: `"C:/ProgramData/anaconda3/python.exe" -m src.pipeline --watchlist` với *Start in* = thư mục dự án. Exit code 1 = có mã lỗi (xem phần CLI).

## Quản trị dữ liệu

**Xoá 1 mã khỏi danh mục (giữ lịch sử):** Tab 2 → "🗑 Xoá". Chỉ sửa `data/watchlist.json`; dữ liệu giá, DB, model, report của mã **giữ nguyên** (vẫn hiện ở Tab 4 — cố ý, để còn đối chiếu).

**Xoá sạch 1 mã (mọi dấu vết):** không có lệnh built-in — làm tay, khi dashboard **đã đóng**:

1. Xoá file: `data/raw/{MÃ}.parquet`, `data/processed/{MÃ}.parquet`, thư mục `models/{MÃ}/`, `reports/{MÃ}/`.
2. Xoá dòng DB (SQLite bật foreign keys — `db.py:87` — nên xoá đúng thứ tự con → cha):

```sql
DELETE FROM outcomes    WHERE prediction_id IN (SELECT id FROM predictions WHERE ticker = 'VCB');
DELETE FROM predictions WHERE ticker = 'VCB';
DELETE FROM runs        WHERE ticker = 'VCB';
```

(schema: `outcomes.prediction_id → predictions.id → runs.id` — `src/db.py:47-64`).

**Reset toàn bộ dự án về trạng thái clone:** đóng app, xoá `data/`, `models/`, `reports/` (toàn bộ nằm ngoài git). Khi chạy lại: `data/symbols.parquet` tự tạo khi mở Tab 1, parquet giá tự tải khi mở Tab 3, `data/watchlist.json` tự tạo khi thêm mã, `data/app.db` tự tạo khi train đầu tiên.

**Mã ngoài HOSE:** xem ghi chú ở [Tab 1](#tab-1--thị-trường) — watchlist chấp nhận mọi mã hợp lệ về hình thức, nhưng nguồn dữ liệu chỉ phủ HOSE.

## Cảnh báo — KHÔNG phải khuyến nghị đầu tư

- Dự án phục vụ **mục đích học tập và nghiên cứu** về ML chuỗi thời gian. Mọi số liệu (dự báo giá, hướng, xác suất, metrics) **không phải khuyến nghị mua/bán** và không được dùng làm căn cứ quyết định đầu tư.
- Model có thể **thua cả baseline** (hiện đang thua — bảng ở trên). Một model từng thắng trong backtest không đảm bảo thắng trong tương lai; hiệu năng quá khứ không dự báo hiệu năng tương lai.
- Dữ liệu giá lấy từ nguồn công khai (CAFEF, sàn HOSE) **không bảo đảm chính xác/đầy đủ**; số liệu cũ từ nguồn HOSE chỉ có close + volume. Lỗi nguồn được hiển thị rõ trong app, không bịa dữ liệu thay thế.
- Chi phí giao dịch thực tế (config ghi nhận phí 0.2%/lượt) chưa được mô hình hoá thành chiến lược giao dịch — dự án **không** tối ưu lợi nhuận danh mục.
- **Nguồn dữ liệu & điều khoản sử dụng**: dự án truy vấn công khai CAFEF (`s.cafef.vn`/`cafef.vn`), sàn HOSE (`www.hsx.vn`/`api.hsx.vn`) và TCBS (dự phòng) theo cách một trình duyệt tải trang. Dự án **không liên kết, không đại diện** cho các bên này; người dùng chịu trách nhiệm tự đọc và tuân thủ điều khoản sử dụng của các trang nguồn; chỉ dùng cho mục đích cá nhân/nghiên cứu phi thương mại.
- **Quyết định đầu tư là trách nhiệm của bạn.** Người phát triển không chịu trách nhiệm về bất kỳ tổn thất nào phát sinh từ việc sử dụng phần mềm này.

## Xử lý sự cố & FAQ

| Triệu chứng | Nguyên nhân & cách xử lý |
|---|---|
| `smoke_test.py` thoát 1, "lỗi mạng sau 3 lần thử" | Script đã retry 3 lần (cách nhau 2 giây) rồi dừng — kiểm tra internet/proxy rồi chạy lại. Thoát 1 với "CAFEF không trả dữ liệu" thì mã không có dữ liệu (kiểm tra chính tả mã). |
| pip cài `torch` rất chậm / lỗi kết nối | Wheel ~124 MB tải từ `download.pytorch.org` (đã cấu hình sẵn `--extra-index-url` trong `requirements.txt:29`); proxy công ty có thể chặn host này. |
| pip cố **build `shap` từ nguồn** (đòi compiler) | Bạn đang cài bản ≠ 0.51.0 — bản 0.52.0 không có wheel cp313 win_amd64 (`requirements.txt:32-36`). Giữ đúng `shap==0.51.0`. |
| `ModuleNotFoundError: lightgbm / streamlit / plotly / …` khi chạy app | Máy bạn **không phải** Anaconda gốc đã có sẵn 17 package — dùng lệnh cài đầy đủ 19 package ở mục [Cài trên máy khác](#cài-trên-máy-khác--môi-trường-sạch). |
| Tab 3 báo "Không tải được dữ liệu giá cho {mã}" (`dashboard.py:581-585`) | Nguồn lỗi hoặc mã không tồn tại/không phải HOSE. Chạy `scripts/smoke_test.py`, kiểm tra mã rồi thử lại. |
| Pipeline dừng "Không đủ lịch sử cho walk-forward: cần >= 756 phiên" | Mã niêm yết quá muộn / nguồn trả ít phiên. Ngưỡng = 750 (train đầu) + 5 (embargo) + 1 (`evaluation.make_folds`) — không có cách "nới lỏng" an toàn; chọn mã khác hoặc chấp nhận không train mã đó. |
| CAFEF chỉ trả ~63 phiên gần nhất | Hành vi phía server (đã xác minh 29/09/2026 — `TotalCount` bị chốt, tham số ngày bị bỏ qua), **không sửa được từ client**. Loader tự ghép API chính thức HOSE cho lịch sử cũ — không cần thao tác gì. |
| Bấm "🚀 Huấn luyện & dự báo" lần nữa khi đang chạy | Thao tác được xếp hàng xử lý **sau** khi run hiện tại xong — tức sẽ huấn luyện lại toàn bộ (run mới). Không có nút huỷ; đừng bấm lại nếu không chủ đích. Xem [thời gian chờ](#e2e-rút-gọn-test-và-thời-gian-chờ-tham-chiếu). |
| Huấn luyện xong nhưng Tab 3/4 không có dự báo | Kiểm tra `st.status` đã báo **"Hoàn tất — run #…"** chưa; nếu có lỗi, lỗi hiển thị ngay trong status (kèm expander debug). Run thành công luôn ghi ít nhất dự báo của naive. |

## Tham chiếu config.yaml

Mọi tham số điều khiển được tập trung ở `config.yaml` — bảng dưới trả lời "đổi ở đâu":

| Key | Giá trị mặc định | Đọc ở đâu / tác dụng |
|---|---|---|
| `default_ticker` | `TCB` | CLI không truyền mã & watchlist trống (`pipeline.py:1053`); mã mặc định Tab 3 (`dashboard.py:229`) |
| `horizons` | `[1, 5]` | Số horizon dự báo (pipeline lặp từng horizon) — `config.get("horizons", [1, 5])` |
| `neutral_band_pct` | `0.3` (±%) | Vùng neutral 3 lớp: labeling (`build_targets`), phân loại hướng pipeline, settle DB (`dashboard.py:907`) |
| `walk_forward.initial_train_size` | `750` | Số phiên train fold đầu (`evaluation.WalkForwardConfig.from_mapping`) |
| `walk_forward.retrain_every` | `21` | Độ dài test block / chu kỳ retrain |
| `walk_forward.embargo` | `5` | Khoảng cách train↔test (engine tự nới lên `max(embargo, horizon)` khi cần purge) |
| `walk_forward.shuffle` | `false` | **Bắt buộc `false`** — `true` bị từ chối ngay khi đọc config |
| `features.feature_groups.price` | `true` | Bật `PriceFeatures` |
| `features.feature_groups.market_context` | `true` | Bật `MarketContext` (VN-Index) |
| `features.feature_groups.news_sentiment` | `false` | Chỗ cắm RAG — bật khi chưa có hiện thực sẽ raise rõ ràng (xem mục RAG) |
| `features.min_valid_ratio` | *(không có trong yaml → 0.5)* | Knob tùy chọn: bỏ feature có tỉ lệ giá trị hữu hạn < ngưỡng (LSTM cần cửa sổ đầy đủ) — `pipeline.py:393-411`; đặt `0` để tắt |
| `data.source` / `data.fallback_sources` | `cafef` / `["tcbs"]` | Mô tả nguồn; **chuỗi nguồn thực tế của loader** là `CafefSource() → HoseOfficialSource()` (`data_loader.py:55-57`), fallback TCBS nằm bên trong `CafefSource` |
| `data.start_date` | `2015-01-01` | Mốc tải lịch sử (`data_loader.DEFAULT_START`) |
| `data.cache.raw` / `data.cache.processed` | `data/raw/{ticker}.parquet` / `data/processed/{ticker}.parquet` | Đường dẫn cache giá |
| `app.watchlist_path` | `data/watchlist.json` | File danh mục |
| `database.path` | `data/app.db` | SQLite |
| `baselines.persistence` / `baselines.majority_class` | `true` / `true` | Ý nghĩa: baseline bắt buộc phải có — hiện thực là `NaiveModel` (`src/models/naive.py`) |
| `evaluation.fee_pct` | `0.2` | Phí giao dịch 0.2%/lượt — **chỉ ghi nhận cấu hình, chưa mô hình hoá** thành chi phí chiến lược |
| `models.*` | *(không có trong yaml)* | Tùy chọn: ghi đè tham số từng model (`models.lstm`, `models.lightgbm`, …); không có → dùng `DEFAULT_PARAMS` tại file model |

## Cắm module tin tức/RAG sau này

Chỗ cắm đã để sẵn, tắt trong config (`config.yaml: features.feature_groups.news_sentiment: false`). Bật flag khi chưa có hiện thực → `assemble.build_providers` raise `NotImplementedError` với thông điệp chỉ dẫn rõ (`src/features/assemble.py:80-85`) — không âm thầm bỏ qua.

### Viết provider nào

1. Tạo `src/features/news_sentiment.py` với class `NewsSentimentProvider` **kế thừa `FeatureProvider`** (`src/features/base.py`), hiện thực `provide(df) -> pd.DataFrame`:
   - index **trùng chính xác** index phiên giao dịch của mã truyền vào (hợp đồng `provide()`);
   - mỗi cột một feature, tên **không** đụng blocklist giá mức và tiền tố `target_/fwd_/label_` (vd. `news_count_3d`, `news_sent_score_5d` — phát theo cửa sổ trượt quá khứ, không phải giá trị tức thời chưa roll);
   - tự fetch/cache tin tức bên trong hoặc qua một nguồn được xác minh hợp đồng trước (nguyên tắc như `src/data_sources/`: retry, lỗi báo rõ, nguồn chết → giá trị trung tính + cảnh báo, không bịa).
2. Đăng ký trong `build_providers()` (`src/features/assemble.py`) sau nhánh `news_sentiment` — mọi provider đi qua cùng kiểm tra: trùng tên cột giữa provider là lỗi, blocklist chặn giá mức/target, sai index là lỗi.
3. **Bật flag** trong `config.yaml`:

```yaml
features:
  feature_groups:
    news_sentiment: true   # bật sau khi có NewsSentimentProvider
```

### Quy tắc chống leakage riêng của tin tức (bắt buộc, từ bài học các dự án tương tự)

- **Cửa giờ + múi giờ**: feature của phiên D chỉ được dùng bài đăng **≤ close phiên D**, tính theo múi giờ **ICT (Asia/Ho_Chi_Minh)**. Bài đăng 1 phút sau close phải rơi vào phiên D+1. Không join theo calendar date mù timezone (UTC vs ICT lệch DST/cóng). Quy tắc này phải có **unit test** (vd. test bài đăng +1 phút sau close).
- **Chính sách null**: "không có tin trong ngày" = **zero-fill** (0 có ý nghĩa: trung tính) — **tuyệt đối không `dropna()`** trên cột event/sentiment (cột này ~99.9% null, dropna là xóa sạch dataset). NaN warm-up đầu lịch sử vẫn theo chính sách chung: để NaN, không bfill.
- **Thêm feature = phải kiểm chứng**: nghiên cứu cho thấy nới từ 14 lên 86 feature làm tín hiệu rớt (t-stat 3.7 → 2.1) — chọn lọc bằng SHAP trước khi giữ feature mới (dự án đã có SHAP từ LightGBM: `reports/{ticker}/shap_importance_h{h}.csv`).

## Tham khảo — các dự án GitHub tương tự

Thiết kế chống leakage, baseline và walk-forward của dự án đối chiếu trực tiếp với các repo mã nguồn mở sau (đã đọc code + đối chiếu kết quả thật của từng repo):

| Dự án | Bài học đã áp dụng vào dự án này |
|---|---|
| [zhaymn/StockIntel](https://github.com/zhaymn/StockIntel) | Purge + embargo lấy đúng bằng label horizon `h` (`splits.py`); leakage bắt bằng **truncation probe chạy được** (`features/leakage.py`) — đã cài làm test trong `tests/test_features.py`; vùng neutral scale theo volatility thay vì băng % cố định; triệu chứng "model h ngày scoring cao hơn 1 ngày" = purge hỏng; hiệu chỉnh effective sample size cho forward return chồng lấn |
| [ArkaDutta-007/Trade-system_v1](https://github.com/ArkaDutta-007/Trade-system_v1) | Purged split (embargo 5 ngày) **cải thiện** tín hiệu thật (IC +0.0081 → +0.0101); phương pháp A/B "identical learners + eval, ONE thing changes"; `NON_FEATURES` blocklist cấm giá mức làm feature; "more ≠ better" — 86 feature làm t-stat 3.7 → 2.1; tư duy PBO/Deflated Sharpe khi đọc backtest |
| [CoderSoham/StockPredictor](https://github.com/CoderSoham/StockPredictor) | Baseline persistence + majority-class tính **trước** và trên **cùng window**; công thức `skill = 100×(1 − MAE/MAE_persistence)` và `dir_edge = dir_acc − majority`; `causal_features()` không có giá mức nào; đối chiếu results thật: 0/9 ticker có skill dương ở horizon 1 ngày, dir acc 62% < base rate 67.6% — dashboard chỉ khoe model khi thắng baseline trên cùng fold |
| [ngocminhta/stock-price-prediction](https://github.com/ngocminhta/stock-price-prediction) | Kiểm định dừng ADF (statsmodels `adfuller`) **trước khi fit** ARIMA — cài trong `src/models/arima_model.py`; CV day-forward-chaining; bug cần tránh đã né: accumulator kết quả không reset giữa các fold khiến "best params" chọn từ fold cũ |
| [Eishaan-Khatri/Reliance-Stock-Forecasting](https://github.com/Eishaan-Khatri/Reliance-Stock-Forecasting) | Mẫu walk-forward expanding + retrain mỗi 21 ngày; preprocessor fit **chỉ trên X_train** rồi transform test; điểm yếu của repo này được tránh: target là giá mức (level), không đối chiếu persistence, số published đến từ holdout tĩnh |
| [eslazarev/purged-cross-validation](https://github.com/eslazarev/purged-cross-validation) | Chuẩn thực hành purged CV: `PurgedKFold`, `WalkForwardSplit`, `audit_splitter()`; metrics chống overfitting backtest (Deflated Sharpe, PBO, effective n trials) — tư duy tham chiếu khi đọc kết quả walk-forward của dự án |

Cấu trúc dự án (interface `PriceDataSource` / `FeatureProvider` / `WalkForwardModel`, pipeline thuần tầng, config trung tâm) tự thiết kế theo đặc tả riêng; các repo trên là nguồn bài học phương pháp luận, không phải code được copy.
