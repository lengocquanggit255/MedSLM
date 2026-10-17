# Kế hoạch: chạy lại benchmark trên STaRK-Prime bản RiTeK

Trạng thái (2026-10-17): **P0, P1 xong.** P2, P3, P5 xong. Đang chạy trên Kaggle: P4 GNN-RAG A+ phiên 3 (`ritek-gnnrag-rearev-s3`, minhduonngg); phiên 4–6 tự đẩy. Chưa chạy: P6 ARK (chờ duyệt).
Pha GPU P2–P5 được duyệt trước: smoke đạt thì tự đẩy bản full; smoke có vấn đề thì chờ duyệt (2026-10-16).
So sánh dữ liệu: [docs/ritek_prime_compare.json](docs/ritek_prime_compare.json) (`scripts/eda_ritek_prime.py`).

Mọi thứ của lần chạy này nằm trong `ritek/`:

| Đường dẫn | Nội dung |
|---|---|
| `ritek/PLAN.md` | file này |
| `ritek/scripts/` | script riêng của lần chạy |
| `ritek/splits/` | file split |
| `ritek/results/<method>/<variant>/<split>/` | `metrics.json` · `manifest.json` · `per_query.parquet` |
| `ritek/outputs/` | output lớn từ Kaggle, đã gitignore |

Harness dùng chung: `src/starkbench/`. Script dùng chung có cờ `--dataset ritek` (ghi vào `ritek/results`): `run_lexical`, `eval_dense`, `eval_dotprod`, `eval_mor_rerank`.

## 1. Dữ liệu RiTeK khác STaRK gốc ở đâu

- Đồ thị, text node, câu hỏi, đáp án: **giống hệt**.
- Tập test = STaRK test, đủ 2,801 query; `test-0.1` (280) nằm trọn trong đó.
- RiTeK train = 5,052 query, là tập con của STaRK train (6,162). RiTeK **không có val**.
- RiTeK thêm `topic_entity`: đúng 1 topic mỗi query. 27 query train và 18 query test không có topic.

⇒ Phương pháp **không** dùng train và **không** dùng topic cho số giống hệt trên test. Đã kiểm: BM25 cho Δ = 0 trên mọi metric.

## 2. Quyết định đã chốt

| # | Quyết định | Ngày |
|---|---|---|
| Q1 | **Dùng lại checkpoint gốc** (planner + reranker MoR, Contriever-FT). Chúng train trên STaRK train, là tập chứa RiTeK train, và không chạm test (planner: 1,081 query, cả 1,081 thuộc STaRK train). | 10-15 |
| Q2 | **GNN-RAG = A+**: kiến trúc và vòng train ReaRev **đúng code gốc**, chạy trên **toàn KG**. Chỉ viết lại `reason_layer`/`reason_layer_inv` sang dạng gộp theo loại quan hệ (tương đương đại số). Trước khi train phải kiểm chứng trùng số với code gốc trên subgraph nhỏ. Cấu hình CWQ của repo (num_ins 3, num_iter 2, num_gnn 3, entity_dim 50), LM `sbert`. **Không ép số epoch nếu một phiên vẫn dưới 12h.** | 10-16 |
| Q3 | Để **ARK** (cần LLM) chạy **cuối cùng**. MoR và MoR+BGE-M3 không gọi LLM vì planner dùng cache. | 10-15 |
| Q4 | Tạm thời chỉ dùng **Kaggle**. Colab để sau. | 10-15 |
| Q5 | BGE-M3: **chỉ dense** (CLS chuẩn hoá), cửa sổ **8192 token**, chạy cả **no_rel** và **add_rel**. Doc dài hơn 8192 token thì cắt chunk 8192 rồi gộp điểm bằng max và top3_mean. Tỉ lệ doc dài hơn: no_rel 0.14%, add_rel 0.68%. | 10-15 |
| Q6 | MoR+BGE-M3: BGE-M3 thay Contriever ở **cả hai vai** (text retriever và scorer), **train lại reranker**. Thêm **run đối chứng MoR-Contriever có reranker train lại trên đúng split RiTeK**, để ablation chỉ khác encoder. | 10-16 |
| Q7 | Val = **10% RiTeK train** (seed 42), tức 4,547 train / 505 val (`ritek/splits/`). | 10-15 |
| Q8 | Query không có topic: GNN-RAG/SubgraphRAG **chỉ dùng điểm backfill**, n_native = 0. Mẫu số vẫn là toàn bộ split. | 10-15 |
| — | SubgraphRAG dùng encoder **`gte-large-en-v1.5`** như paper (lần chạy trên STaRK đã dùng MedCPT, là một sai lệch). | 10-16 |
| — | **Full test là kết quả chính.** Số trên `test-0.1` cắt từ `per_query.parquet` của run full test. | 10-16 |
| — | **Git**: commit local các file của lần chạy này (harness `src/starkbench/`, script dùng chung, `ritek/`). Remote: nhánh `stark-ritek` của MedSLM = `origin/main` + `git subtree add --prefix=stark_ritek master`; cập nhật bằng `git subtree merge --prefix=stark_ritek master` trong một worktree. Không đụng `main`, `ming`. | 10-16 |
| — | SubgraphRAG: text đưa vào gte = **tên** node, tên quan hệ, câu hỏi, đúng code gốc (`text_entity_list`, `relation_list`). Lần chạy STaRK dùng embedding doc MedCPT add_rel. Subgraph PPR-2000 (alpha 0.85, 20 vòng) như lần STaRK, seed = topic RiTeK. | 10-16 |
| — | SubgraphRAG **khớp code gốc** (đối chiếu Graph-COM/SubgraphRAG @ 48eb05e, user yêu cầu sửa cả 4 điểm lệch): (1) nhãn = đường ngắn nhất **hai chiều** topic<->đáp án theo từng cặp, không giới hạn bước, DiGraph gộp cạnh song song — bản cũ một chiều thiếu đúng 1/2 nhãn (RiTeK: 2.00× triple dương, cùng 3,243/357/1,971 query có đường); (2) train trên **toàn bộ** cạnh subgraph (backward theo khúc), bỏ lấy mẫu 100k; (3) mô hình = `Retriever` gốc (one-hot 2 chiều, DDE 2 kênh, một Linear đầu); (4) vòng train = train.py gốc (seed 42, val trước train, patience 10). Kiểm: `test_sgrag_equivalence.py` (chạy chính code gốc) + `test_sgrag_resume.py`. Chạy lại đủ **sau khi GNN xong**; số `mlp_sqrt`/`mlp_none` hiện có là bản cũ. | 10-17 |
| — | **MoR reranker train lại — adapted:** `batch_evaluator` gốc gán điểm 0 cho node ngoài candidate. Điểm reranker train lại có thể âm hết (loss cross-entropy không phụ thuộc dấu), khi đó val hit@1 = 0 mọi epoch và checkpoint "best" kẹt ở epoch 1. Sửa: node ngoài candidate = −inf (`ritek/kaggle/mor_ritek`). Run `abe23ee` bị lỗi này, không dùng. | 10-16 |
| — | MoR-prime chỉ dùng Contriever-FT ở **một** chỗ: `scorer` (dot product embedding tính sẵn). Điểm này vừa chọn top-100 candidate vừa là một cột feature của reranker. Text retriever của MoR-prime là BM25. Nên MoR+BGE-M3 = thay hai dict embedding của scorer (doc no_rel, như Contriever), BM25 giữ nguyên. | 10-16 |

## 3. Metric — một định nghĩa cho mọi phương pháp

Bộ metric (`eval.METRICS_REPORT`): **F2@5 · Hit@1 · Hit@5 · Recall@10 · Recall@20 · MRR**, macro-average theo query, bản `sanitized`.

- `F2@5 = 5·tp / (4·G + k)`
  - **k = min(5, m)**, với m = số node do **chính phương pháp** trả về (`n_native`, chưa tính backfill).
  - G = số đáp án (ID node, đã bỏ trùng).
  - tp = số đáp án trong k node đó.
  - Phương pháp chấm mọi node có m = 129,375 nên k = 5. Query mà phương pháp không trả gì có m = 0 nên F2 = 0.
  - Không dùng k = m cho phương pháp xếp hạng: m của chúng là 100 (MoR), 2,000 (SubgraphRAG) hay 129,375 (BM25), nên F2 sẽ đo độ dài danh sách chứ không đo chất lượng.
  - Với phương pháp trả về **tập đáp án** (ARK) có m > 5: quyết định khi tới ARK. Cả hai cách tính được từ output đã lưu.
- `Hit@k`, `Recall@k` (= |A ∩ top-k| / |A|), `MRR`: tính trên **toàn bộ xếp hạng** 129,375 node, gồm cả backfill.
- `native@k`: tỉ lệ vị trí trong top-k do chính phương pháp đưa ra.

Kiểm chứng (`scripts/test_parity.py`, 2026-10-16):
- 296 phép thử × 7 metric khớp `stark_qa.Evaluator` (có recall@10).
- F2 khớp công thức ở 300 phép thử và 4 ca viết tay (m = 0, 1, 3, 10).
- Parity chỉ so ở vùng mà mọi ngưỡng cắt nằm trong phần đã chấm điểm (≥ 20 node). Ngoài vùng đó là dải sàn hoà nhau: `torch.topk` chọn tuỳ ý còn ta chọn ID nhỏ trước. Benchmark luôn backfill nên không chấm ở vùng này.

Kết quả rà soát (2026-10-15):

| Điểm | Phát hiện | Xử lý |
|---|---|---|
| ID đáp án trùng | 6 query STaRK test có `answer_ids` lặp. RiTeK dùng dict nên không lặp. | G luôn đếm sau khi bỏ trùng. |
| Khớp theo tên | 31 node đáp án test có tên trùng node khác, ảnh hưởng 63 query **nếu** khớp theo tên. | Mọi phương pháp xuất **ID node**, nên khớp theo ID. |
| MRR từ top-1000 | Thấp hơn bản full tối đa 0.007pp. | Số chính thức lấy từ `metrics.json` và `per_query.parquet`. |
| Bộ chấm riêng của từng repo | MoR gán 0 cho node không chấm. ARK chỉ xét top-20. SubgraphRAG chấm triple. stark_qa BM25 chỉ top-100. | **Không dùng** cho bảng. Mọi phương pháp đi qua `runner.evaluate_and_save`. |
| Query không có dự đoán | `eval_mor_rerank.py` từng **lặng lẽ bỏ** các query này, làm mẫu số co lại. | Đã sửa: giữ lại, chấm bằng backfill, n_native = 0. |
| Điểm hoà | `evaluate_one` phá hoà theo ID nhỏ trước. | Adapter phải đổi thứ tự riêng của phương pháp thành điểm **giảm ngặt**. |
| Dấu điểm | torchmetrics loại đáp án có điểm ≤ 0 khỏi recall/MRR. | Báo cáo bản `sanitized`. |

## 4. Phương pháp và việc phải làm

| Nhóm | Phương pháp | Với RiTeK | GPU |
|---|---|---|---|
| c | BM25 no_rel/add_rel | chấm lại (số không đổi) | CPU |
| c | MedCPT (no_rel/add_rel × max/top3) | chấm lại từ `results/kaggle/stark-enc-medcpt-*-full/scores_*.npy` | CPU |
| c | Contriever-FT | chấm lại từ `results/kaggle/stark-mor-emb` | CPU |
| c | BGE-M3 (no_rel/add_rel) | encode mới | Kaggle |
| a | MoR (đủ, traversal, w/o RS, w/o RT) | chấm lại từ dự đoán đã có | CPU |
| a | MoR-Contriever, reranker train lại (đối chứng) | traversal train/val bằng Contriever-FT, train reranker, rerank test | Kaggle + CPU |
| a | MoR + BGE-M3 | traversal train/val/test bằng BGE-M3, train reranker, rerank | Kaggle + CPU |
| a | ARK | **cuối cùng** | — |
| b | SubgraphRAG `none` + `sqrt` | gte-large, seed = topic RiTeK, PPR-2000, train trên 4,547 | Kaggle |
| b | GNN-RAG (A+) | ReaRev gốc trên toàn KG, seed = topic RiTeK | Kaggle |

## 5. Thứ tự và chi phí (số đo thật từ log Kaggle T4x2 của các lần chạy STaRK)

| Pha | Việc | Đã đo trên STaRK | Ghi chú |
|---|---|---|---|
| P0 | harness | — | ✅ xong 2026-10-16 |
| P1 | chấm lại BM25, MedCPT, Contriever-FT, MoR | rerank MoR trên CPU 6.6 phút | ✅ 11 run, metric cũ Δ = 0 so với STaRK |
| P2 | BGE-M3 | MedCPT 7–15 phút/biến thể | ✅ encode no_rel 18 phút, add_rel 47 phút (T4x2). Test: add_rel_max H@1 12.07 · R@20 31.32 · MRR 19.59; no_rel H@1 11.21 · R@20 29.05 · MRR 18.20. 4 run trong `ritek/results/cosine_bgem3/` |
| P3 | SubgraphRAG (+ encode gte-large) | `none` 2.6h · `sqrt` 8.2h | Code 9dcbbde (đã sửa rò nhãn val/test). `sqrt` ✅: H@1 6.89 · H@5 25.88 · R@20 36.00 · MRR 15.98, best epoch 16, val triple_recall@100 72.3%. `none` ✅: H@1 6.68 · H@5 24.67 · R@20 32.77 · MRR 15.10, best epoch 36 (dừng sớm ở 46), val triple_recall@100 67.1% -> `sqrt` là bản chính. **Bản cũ, lệch code gốc** (xem quyết định 10-17): code mới d906dba/2feeaee, smoke `ritek-sgrag-smoke2` ✅ (tienthanhahihi, 16 phút): nạp đúng nhãn mới (train 2,659,856 triple dương), VRAM đỉnh 2.45 GB ở subgraph 1.3M cạnh, 1.04 s / 1M cạnh fwd+bwd -> ~0.54h train + ~1 phút val mỗi epoch đủ (3,243 query × ~580k cạnh); encode gte 14 phút/phiên. Bản cũ dừng ở epoch 27 (`sqrt`) / 46 (`none`) -> ước 15–26h GPU mỗi biến thể, 2–3 phiên. Chạy đủ sau GNN |
| P4 | GNN-RAG A+ | 36–80 phút trên PPR-2000 | ✅ trùng số với code gốc (`test_rearev_equivalence.py`). Phiên 1: 11 epoch / 10.1h (~54 phút/epoch), dev H1 14.3→22.5 vẫn tăng; test (best epoch 10) H@1 20.31 · R@20 46.16 · MRR 28.10 (`gnnrag/rearev_fullkg_s1`). **Không giảm epoch**: chạy tiếp nhiều phiên, state đầy đủ (`test_rearev_resume.py`: Δ = 0); lần nối đầu Adam khởi tạo lại. Còn 89 epoch ≈ 81h GPU ở 1 GPU. Từ phiên 3: **2 GPU** (micro-batch song song, cùng gradient: `test_rearev_dual.py` Δ = 0; smoke T4x2 lệch 0.0, 5.48 → 2.78 giây/batch), state lưu mỗi epoch. Phiên 2 ✅ (1 GPU, epoch 11→22, dừng theo hạn): dev H1 tốt nhất 24.3 (epoch 20); test H@1 22.92 · H@5 42.02 · R@20 49.13 · MRR 31.70 (`gnnrag/rearev_fullkg_s2`). Phiên 3 ✅ (epoch 22→34, nối state đầy đủ): kiểm 2 GPU báo lệch tương đối 0.52 > 1e-4 -> chạy 1 GPU; dev H1 tốt nhất 25.8; test H@1 23.13 · H@5 42.16 · R@20 49.73 · MRR 31.90 (`gnnrag/rearev_fullkg_s3`). Chẩn đoán (`ritek-gnnrag-dual-diag`, DUAL_DIAG=1, state epoch 34): trọng số và loss trùng tuyệt đối; lệch do nhiễu cộng của GPU — hai lần chạy 1 GPU y hệt đã lệch tới 80% trên 2 bias có gradient ~1e-8; lệch L2 toàn vector của đường 2 GPU 1.1–1.8e-7 = nhiễu nền 1 GPU 1.1–2.1e-7. Code 5eac6e4: tiêu chí = lệch tương đối L2 toàn vector ≤ 1e-5 (bản sao lệch cỡ 1 bước Adam cho 0.55). Phiên 4 bản 1 GPU (`-s4`) đã xoá sau ~40 phút; 🔄 `ritek-gnnrag-rearev-s4b` (minhlon192, từ epoch 34, 2 GPU) → s5 minhlon192 → s6 tienthanhahihi, ~22 epoch/phiên. Các phiên sau tự đẩy bằng `ritek/scripts/gnn_next_session.py`, hạn 11.5h/phiên (Kaggle cho phiên đã bắt đầu chạy hết 12h dù quota cạn): s3 minhduonngg → s4, s5 minhlon192 → s6 tienthanhahihi; ~21–22 epoch/phiên, dự kiến xong ~18/10 |
| P5 | MoR + BGE-M3 và đối chứng | traversal test ~11 phút | MoR+BGE-M3 (code 53b8416): rerank H@1 25.21 · R@20 46.59 · MRR 34.25; traversal H@1 12.89 · R@20 40.96. Val hit@1 reranker 14.5→30.5. Đối chứng Contriever (reranker train lại) ✅: H@1 35.42 · R@20 63.08 · MRR 46.12 (checkpoint tác giả: 36.09 · 63.05 · 46.76); traversal chạy lại trùng P1 từng query (Δ = 0). ⇒ thay Contriever-FT bằng BGE-M3: H@1 −10.2, R@20 −16.5. |
| P6 | ARK | — | để sau |

Mỗi pha cần GPU: smoke test → probe → bảng ước lượng đo thật → chờ duyệt → chạy.
