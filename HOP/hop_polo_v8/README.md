# HOP PoLo V8

Bản rút gọn của HOP PoLo V7 Setting B ([../hop_polo_v7_final/](../hop_polo_v7_final/), mốc git `hop-v7`),
chạy được trên ba bộ RiTeK: **PrimeKG**, **PharmKG**, **ADint**. Với cấu hình PharmKG (`configs/default.json`)
thuật toán giữ nguyên V7, trừ `lm_head` của Flan, trần epoch và bộ token action (bảng dưới); test
`tests/test_equivalence_v7.py` chạy chính code V7 cạnh code V8 (action `schema` như V7) trên cùng đồ thị, cùng trọng
số và so từng phép tính. Prime và ADint dùng siêu
tham số chọn theo EDA ([docs/HYPERPARAMS.md](docs/HYPERPARAMS.md)).

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
| Giữ `lm_head` riêng của Flan-T5 sau khi thêm token (`flan.untie_lm_head = true`) | transformers 5.x buộc `lm_head` vào embedding khi resize | Có: Hits@1 val stage Flan PharmKG +1,95pp, p = 0,008 (mục dưới) |
| Trần epoch 24 → 60 cho mọi pha, patience giữ 3 | QA-SFT của PharmKG chạm trần 24 khi Hits@1 val còn tăng; resume không cho đổi config | Có khi pha còn tăng sau epoch 24 |
| Teacher bỏ qua đáp án không có đường ≤ `max_hops` (đếm ở `labels/teacher_depths.json`, `warmup/done.json`) | V7 dừng hẳn: Prime có 23 đáp án train (13 câu) xa hơn 3 hop | Không trên PharmKG (mọi đáp án ≤ 3 hop) |
| `max_hops` theo config (PharmKG, Prime 3; ADint 4), khai thác nhãn đường 4 hop (2 hop xuôi gặp 2 hop ngược) | ADint: ≤ 3 hop chỉ phủ 90,3% đáp án, ≤ 4 hop 97,9% | Có (ADint) |
| Pool lấy cả con top-`local_top_k` ở hop cuối (`pool_last_hop`) | V7 chỉ lấy hop 1–2, nút ở hop cuối chỉ vào qua ≤ `path_beam` đường; Prime có 26% đáp án ở 3 hop | Có (Prime, ADint) |
| Quota cắt pool theo độ sâu vào config (`pool_depth_quotas`); chỗ thừa của độ sâu ít ứng viên chia tiếp theo cùng tỉ lệ | Prime: hàng nghìn ứng viên ở cả độ sâu 2 và 3; V7 để chỗ thừa cho điểm toàn cục (nghiêng về nút nông) | Không trên PharmKG (test so với bản sao V7) |
| Gradient checkpointing log-softmax bước hai của Setting B khi huấn luyện | Prime: hub ~15k cạnh × 6 nhánh × tới 256 route vượt 14 GB GPU (run 2026-10-28 hết bộ nhớ) | Không: cùng phép toán (test so khối lượng và gradient) |
| Hoà Hits@1 val phá bằng Recall@20 khi chọn checkpoint và early stopping (`selection.tie_break`), patience 5 (Prime, ADint) | Val 503 / 384 câu, Hits@1 thấp nên hay hoà; run 2026-10-28 dừng ở epoch 6–7 khi R@20 còn tăng | Có (Prime, ADint); PharmKG giữ luật V7 |
| Action (quan hệ, chiều, loại đích) thay schema (`graph.action = "target"`), chung cho cả ba bộ (Prime, PharmKG từ 2026-10-29) | ADint: 16.732 schema, 51% có ≤ 5 cạnh. Topic cùng loại thì hai cách chọn đúng cùng tập cạnh: mọi câu Prime, PharmKG có một topic; 2,9% câu train ADint có topic khác loại (hop đầu lấy tập cạnh rộng hơn) | Có: token action PharmKG 552 → 182, Prime 56 → 45, ADint 16.732 → 1.395; Flan và embedding action của policy dùng chung token giữa các loại nguồn |

Audit giữ nguyên như V7: nhãn QA, phân bố chương trình, lệch chương trình–đường vàng, hard audit pool,
coverage, soft–hard gap (mode C, τ), khoá checkpoint, đếm lại độc lập, hash input/source/model.

## Config

| File | Dùng cho |
|---|---|
| `configs/default.json` | PharmKG: siêu tham số V7 + `flan.untie_lm_head=true` + trần 60 epoch (2026-10-28) + action (quan hệ, chiều, loại đích) (2026-10-29); các config khác kế thừa action này |
| `configs/prime.json` | Prime: trần nhãn 500; `local_top_k` 250, `path_beam` 60, pool 3.000 gồm cả hop cuối, quota 43/31/26%; hoà Hits@1 phá bằng R@20, patience 5 |
| `configs/adint.json` | ADint: 4 hop, pool gồm cả hop cuối, quota 48/34/10/8%; hoà Hits@1 phá bằng R@20, patience 5 |
| `configs/tied.json` | `lm_head` buộc như V7 trên transformers 5.x: nhánh đối chứng của ablation |
| `configs/smoke.json`, `smoke_prime.json`, `smoke_adint.json` | Smoke của từng bộ: 48/24/24 câu, mỗi pha 1 epoch |

`--stop-after <stage>` (kernel: `STOP_AFTER`) dừng sau một stage, vd chỉ chạy tới hết Flan cho ablation;
chạy tiếp bằng `--resume` không kèm `--stop-after`.

## lm_head của Flan-T5

Flan-T5 (T5 v1.1) có `lm_head` huấn luyện riêng, khác ma trận embedding `shared`. transformers 5.16.1
ép `tie_word_embeddings=True` cho mọi T5 (`configuration_t5.py`), nên `resize_token_embeddings` (PharmKG khi
còn action schema: thêm 753 token chương trình, nay 383) thay `lm_head` bằng `shared`: logit của cùng một câu lệch tới 723 so với Flan gốc (độ
lệch chuẩn logit gốc 3.5), và gradient phía đầu ra (CE trên toàn vocab) cập nhật embedding mà encoder dùng
để đọc câu hỏi. Image Kaggle có sẵn đúng transformers 5.16.1 (kernel probe 2026-10-27), nên V7 và checkpoint
Setting B cũ (H@1 0.563) đều chạy với `lm_head` bị buộc; bản replay V7 ở local dùng transformers 4.53.2 thì
không buộc. `untie_lm_head=true` dựng lại `lm_head` gốc sau resize (`tests/test_flan_head.py`: logit vocab cũ
trùng Flan gốc, còn nguyên sau save → load; log Kaggle xác nhận 5.16.1 không buộc lại khi nạp checkpoint).

Ablation stage Flan trên PharmKG (2026-10-28, T4, CODE_SHA `407e490…`, kernel
`ritek-hop-v8-pharmkg-flan-{tied,untied}`, một seed mỗi nhánh, chọn epoch trên val 1.024 câu):

| | tied (V7) | untied (V8) |
|---|---:|---:|
| Hits@1 val | 0,3447 | **0,3643** |
| Recall@20 val | 0,6567 | 0,6725 |
| Epoch chọn: synthetic / QA-SFT | 5 / 23 | 2 / 23 |
| Thời gian stage Flan | 3,7 h | 3,1 h |

McNemar exact trên Hits@1 từng câu: 16 câu chỉ tied đúng, 36 câu chỉ untied đúng, p = 0,0078, nên V8 chọn
untied. Untied dẫn ở mọi epoch QA-SFT; khoảng cách hẹp dần từ 3–8pp (epoch 4–9) xuống ~2pp (epoch 23).

## Siêu tham số theo bộ dữ liệu

Giá trị V7 được chỉnh trên PharmKG. Các giá trị phụ thuộc dữ liệu đã được chọn lại cho Prime và ADint theo EDA
(quyết định 2026-10-28); bảng quyết định, số liệu căn cứ và các giá trị còn giữ của V7 (cần test sau) nằm ở
[docs/HYPERPARAMS.md](docs/HYPERPARAMS.md), sinh lại bằng `tools/hparam_stats.py` và `tools/hub_groups.py`.
Setting B vẫn chỉ học route ≤ 2 hop như V7 (đáp án 3–4 hop chỉ có teacher CE).

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
