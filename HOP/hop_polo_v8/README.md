# HOP PoLo V8

Bản rút gọn của HOP PoLo V7 Setting B ([../hop_polo_v7_final/](../hop_polo_v7_final/), mốc git `hop-v7`),
chạy được trên ba bộ RiTeK: **PrimeKG**, **PharmKG**, **ADint**. Thuật toán giữ nguyên V7; test
`tests/test_equivalence_v7.py` chạy chính code V7 cạnh code V8 trên cùng đồ thị, cùng trọng số và so
từng phép tính.

## Khác V7

| Thay đổi | Lý do | Ảnh hưởng kết quả |
|---|---|---|
| Bỏ codebook cấu trúc G, mã G, các mode executor khác `semantic` | Setting B không dùng G ở bước nào | Không (RNG từng stage nay được seed tường minh) |
| Bỏ cột old-score và cổng old-score của ranker | Luôn bằng 0 trong V7 | Không |
| Cạnh ngược chỉ thêm cho triple chưa có chiều ngược (`graph.inverse = "missing"`) | Prime lưu sẵn cả hai chiều cho 100% triple; V7 nhân đôi 8,1M cạnh thành 16,2M | Có: bớt cạnh trùng (Prime ~½, PharmKG −16%) |
| Epoch 0 của QA-SFT, warm-up, Setting B được chép từ dòng đã chọn của pha trước | Trùng hệt (cùng trọng số, cùng chương trình val) | Không; bớt 3 lượt chấm val |
| Chỉ giữ checkpoint `best` + `last` mỗi pha, hash mọi epoch nằm trong lịch sử | Flan 300 MB × tới 50 epoch vượt 20 GB output Kaggle | Không |
| Vector hoá chấm cạnh, đếm action, tiền tố S; phần tĩnh của MLP dùng chung giữa các nhánh Setting B | Prime: bước 2 của Setting B tới ~400k cạnh/nhánh/đáp án | Không (sai số float ≤ 2e-5) |
| Bundle đầu vào chuẩn hoá; đáp án test ở file riêng | Một loader cho ba bộ; hoãn đọc đáp án test thành ràng buộc cấu trúc | Không |
| Nối phiên: stage `done.json`, trạng thái đầy đủ mỗi epoch, lưu từng phần các bước dài | Prime cần nhiều phiên Kaggle 12 h | Không (test nối phiên = chạy liền) |

Audit giữ nguyên như V7: nhãn QA, phân bố chương trình, lệch chương trình–đường vàng, hard audit pool,
coverage, soft–hard gap (mode C, τ), khoá checkpoint, đếm lại độc lập, hash input/source/model.

## Dữ liệu

`tools/prep_inputs.py --dataset prime|pharmkg|adint` dựng `inputs/<bộ>/` (gitignore) từ `data/ritek`:

| Bộ | Train | Val | Test | Ghi chú |
|---|---:|---:|---:|---|
| prime | 4.522 | 503 | 2.801 | split `ritek/splits`; 18 câu test không có topic: dòng rỗng, chỉ backfill khi chấm |
| pharmkg | 8.186 | 1.024 | 1.023 | dev chính thức làm val; bỏ 2 câu train trùng văn bản test |
| adint | 3.457 | 384 | 482 | val = 10% train theo quy tắc V7 (dev RiTeK không có topic); topic dạng tên được tra như V7 |

Văn bản nút Prime theo D5(b): `no_rel` của STaRK bỏ hai dòng tên/loại.

## Môi trường

Torch, NumPy, SciPy, scikit-learn, pandas lấy từ `.venv` của repo (trên Kaggle: bản có sẵn của image).
Bộ thư viện text ghim nằm trong `requirements.txt`; kernel Kaggle cài đúng các bản này. Ở local, cài riêng
vào một thư mục để không đụng `.venv` (repo dùng transformers khác bản):

```bash
.venv/Scripts/python.exe -m pip install --no-deps --target .cache/hop_v8_pylib -r HOP/hop_polo_v8/requirements.txt
```

Rồi chạy với `PYTHONPATH=.cache/hop_v8_pylib` (thư mục đứng trước site-packages của `.venv`).

## Chạy

Local (CPU):

```bash
PYTHONUTF8=1 PYTHONPATH=.cache/hop_v8_pylib .venv/Scripts/python.exe HOP/hop_polo_v8/src/run.py \
  --input HOP/hop_polo_v8/inputs/pharmkg --config HOP/hop_polo_v8/configs/smoke.json \
  --output <thư mục rỗng> --flan-dir <flan-t5-small> --embedding-dir <all-mpnet-base-v2> --device cpu
```

Kaggle (T4): `tools/upload_kaggle.py code|data` tạo dataset `ritek-hop-v8-code` và `ritek-hop-v8-<bộ>`
(in `CODE_SHA`), rồi đẩy job bằng `scripts/kaggle_run.py --jobs HOP/hop_polo_v8/kaggle/jobs/<job>.json
--outdir ritek/outputs/kaggle`. Job dùng `code: HOP/hop_polo_v8/kaggle/hop_v8.py` và biến môi trường
`DATASET`, `CONFIG`, `CODE_SHA`, `DEADLINE_H`, `RESUME=auto` (phiên nối: thêm kernel phiên trước vào `kernels`).

Output: `codebook/ flan/ programs/ warmup/ setting_b/ ranker/ test/`, `audit.json` (trạng thái RUNNING /
PAUSED / COMPLETE). Test của Prime được chấm lại qua harness repo bằng `tools/eval_prime.py`
(F2 trên tập `selected`, xếp hạng = selected → phần còn lại của pool → backfill).

## Kiểm tra

```bash
PYTHONUTF8=1 .venv/Scripts/python.exe -m unittest HOP/hop_polo_v8/tests/test_contract.py
PYTHONUTF8=1 PYTHONPATH=.cache/hop_v8_pylib .venv/Scripts/python.exe -m unittest HOP/hop_polo_v8/tests/test_equivalence_v7.py
```
