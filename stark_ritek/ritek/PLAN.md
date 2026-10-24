# Kế hoạch: chạy lại benchmark trên STaRK-Prime bản RiTeK

Trạng thái (2026-10-20): **P0–P5 xong** (bảng §6). Ablation: GNN-RAG + BGE-M3 (P4b) dừng ở epoch 42 (20/10); SubgraphRAG `sqrt` dừng ở epoch 38 (24/10). Không còn job Kaggle nào chạy. Chưa chạy: P6 ARK (chờ duyệt).
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
| Q9 | **Bản chính của retriever dense/lexical: corpus `add_rel` (n_rel=-1) + gộp chunk `top3_mean`** — đúng stark_qa `MultiVSS` (`get_doc_info(add_rel=True)`, cắt chunk, `top3_avg`); `add_rel` chứa trọn `no_rel` và cao hơn `no_rel` ở mọi encoder; `top3_mean` giảm thiên vị node hub nhiều chunk. `no_rel` và gộp `max` vào phụ lục (độ nhạy). Backfill của GNN-RAG/SubgraphRAG giữ MedCPT add_rel max như các lần chạy trước. Bảng: `ritek/scripts/main_table.py`. | 10-20 |
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
| P2 | BGE-M3 | MedCPT 7–15 phút/biến thể | ✅ encode no_rel 18 phút, add_rel 47 phút (T4x2). Test (bản chính Q9) add_rel H@1 12.10 · R@20 30.94 · MRR 19.51; phụ lục: add_rel_max H@1 12.07 · R@20 31.32 · MRR 19.59, no_rel H@1 11.21 · R@20 29.05 · MRR 18.20. 4 run trong `ritek/results/cosine_bgem3/` |
| P3 | SubgraphRAG (+ encode gte-large) | `none` 2.6h · `sqrt` 8.2h | Code 9dcbbde (đã sửa rò nhãn val/test). `sqrt` ✅: H@1 6.89 · H@5 25.88 · R@20 36.00 · MRR 15.98, best epoch 16, val triple_recall@100 72.3%. `none` ✅: H@1 6.68 · H@5 24.67 · R@20 32.77 · MRR 15.10, best epoch 36 (dừng sớm ở 46), val triple_recall@100 67.1% -> `sqrt` là bản chính. **Bản cũ, lệch code gốc** (xem quyết định 10-17): code mới d906dba/2feeaee, smoke `ritek-sgrag-smoke2` ✅ (tienthanhahihi, 16 phút): nạp đúng nhãn mới (train 2,659,856 triple dương), VRAM đỉnh 2.45 GB ở subgraph 1.3M cạnh, 1.04 s / 1M cạnh fwd+bwd -> ~0.54h train + ~1 phút val mỗi epoch đủ (3,243 query × ~580k cạnh); encode gte 14 phút/phiên. Bản cũ dừng ở epoch 27 (`sqrt`) / 46 (`none`) -> ước 15–26h GPU mỗi biến thể, 2–3 phiên. 🔄 `none` chạy đủ trên minhduonngg từ 17/10 (`ritek-sgrag2-none-s<k>`, code 2feeaee, tự nối phiên bằng `ritek/scripts/sgrag_chain.py`). Phiên 1 ✅ (19 epoch, 10.8h, ~32 phút train/epoch, VRAM đỉnh 2.46 GB): val triple_recall@100 tốt nhất 58.68% (epoch 18), còn tăng, patience 0/10; test tạm (checkpoint epoch 18) H@1 6.10 · H@5 20.96 · R@20 29.68 · MRR 13.49 (`subgraphrag/orig_none_s1`, chưa hội tụ). Phiên 2 ✅ (epoch 19→38). Phiên 3 ✅ (1 epoch) → **`none` XONG**: dừng sớm sau 39 epoch (patience 10), tốt nhất epoch 28 val triple_recall@100 60.56%; test **F2@5 11.22 · H@1 6.32 · H@5 22.21 · R@10 24.68 · R@20 30.63 · MRR 14.06** (`subgraphrag/orig_none`). So bản cũ lệch code gốc `mlp_none` (`scripts/compare.py --results-root ritek/results --dataset ritek`, n = 2,801): H@1 −0.36 (p 0.40, không kết luận được); H@5 −2.46, R@20 −2.14, MRR −1.04 (p ≤ 0.003) — bản đúng code gốc xếp hạng node KÉM hơn có ý nghĩa. Tổng GPU `none` ~22.6h / 3 phiên. 🔄 `sqrt` (`ritek-sgrag2-sqrt-s<k>`, tienthanhahihi) từ 18/10 sau khi GNN xong: phiên 1 ✅ (epoch 0→19, tốt nhất epoch 15 val 66.05%), phiên 2 ✅ (epoch 19→38, dừng theo hạn; tốt nhất epoch 35 val **67.75%**, patience 2/10); test tạm (checkpoint epoch 35) F2@5 14.33 · H@1 7.25 · H@5 27.95 · R@10 31.03 · R@20 37.31 · MRR 16.88 (`subgraphrag/orig_sqrt_s2`). **Dừng ở epoch 38 theo yêu cầu 24/10** (phiên 3 đã chạy ~1.5h thì huỷ, xoá kernel `ritek-sgrag2-sqrt-s3`; chưa chạm patience 10): kết quả ablation = bản tạm `subgraphrag/orig_sqrt_s2` (checkpoint epoch 35), chưa hội tụ. Tải output phiên 2 gặp 502/IncompleteRead (file thành trang lỗi HTML) -> `kaggle_run.download_output` kiểm và tải lại |
| P4 | GNN-RAG A+ | 36–80 phút trên PPR-2000 | ✅ trùng số với code gốc (`test_rearev_equivalence.py`). Phiên 1: 11 epoch / 10.1h (~54 phút/epoch), dev H1 14.3→22.5 vẫn tăng; test (best epoch 10) H@1 20.31 · R@20 46.16 · MRR 28.10 (`gnnrag/rearev_fullkg_s1`). **Không giảm epoch**: chạy tiếp nhiều phiên, state đầy đủ (`test_rearev_resume.py`: Δ = 0); lần nối đầu Adam khởi tạo lại. Còn 89 epoch ≈ 81h GPU ở 1 GPU. Từ phiên 3: **2 GPU** (micro-batch song song, cùng gradient: `test_rearev_dual.py` Δ = 0; smoke T4x2 lệch 0.0, 5.48 → 2.78 giây/batch), state lưu mỗi epoch. Phiên 2 ✅ (1 GPU, epoch 11→22, dừng theo hạn): dev H1 tốt nhất 24.3 (epoch 20); test H@1 22.92 · H@5 42.02 · R@20 49.13 · MRR 31.70 (`gnnrag/rearev_fullkg_s2`). Phiên 3 ✅ (epoch 22→34, nối state đầy đủ): kiểm 2 GPU báo lệch tương đối 0.52 > 1e-4 -> chạy 1 GPU; dev H1 tốt nhất 25.8; test H@1 23.13 · H@5 42.16 · R@20 49.73 · MRR 31.90 (`gnnrag/rearev_fullkg_s3`). Chẩn đoán (`ritek-gnnrag-dual-diag`, DUAL_DIAG=1, state epoch 34): trọng số và loss trùng tuyệt đối; lệch do nhiễu cộng của GPU — hai lần chạy 1 GPU y hệt đã lệch tới 80% trên 2 bias có gradient ~1e-8; lệch L2 toàn vector của đường 2 GPU 1.1–1.8e-7 = nhiễu nền 1 GPU 1.1–2.1e-7. Code 5eac6e4: tiêu chí = lệch tương đối L2 toàn vector ≤ 1e-5 (bản sao lệch cỡ 1 bước Adam cho 0.55). Phiên 4 bản 1 GPU (`-s4`) đã xoá sau ~40 phút. Phiên 4b ✅ (`ritek-gnnrag-rearev-s4b`, minhlon192, epoch 34→57, **2 GPU bật**: lệch L2 1.17e-7 = nhiễu nền 1 GPU 1.08e-7; 3,158 → 1,636 giây/epoch): dev H1 tốt nhất 26.04 (10 epoch cuối 25.6–26.0), train H1 21.2→23.8; test H@1 23.60 · H@5 42.63 · R@10 44.53 · R@20 51.89 · MRR 32.61 · F2@5 23.28 (`gnnrag/rearev_fullkg_s4b`). Phiên 5 ✅ (minhlon192, epoch 57→81, 2 GPU lệch L2 7.3e-7): dev H1 tốt nhất 26.44; test H@1 25.10 · H@5 43.56 · R@20 52.18 · MRR 33.76 (`gnnrag/rearev_fullkg_s5`). **Phiên 6 ✅ — XONG 100 epoch** (tienthanhahihi, epoch 81→100, 8.7h, 2 GPU lệch L2 2.2e-7 / nhiễu nền 2.6e-7): dev H1 tốt nhất **27.24** (epoch 98, checkpoint dùng cho test); test **F2@5 24.27 · H@1 25.92 · H@5 44.91 · R@10 46.03 · R@20 52.90 · MRR 34.73** (`gnnrag/rearev_fullkg`, kết quả chính). Xếp hàng GPU Kaggle tối 17/10: s5 chờ 3h32m. Các phiên sau tự đẩy bằng `ritek/scripts/gnn_next_session.py`, hạn 11.5h/phiên (Kaggle cho phiên đã bắt đầu chạy hết 12h dù quota cạn): s3 minhduonngg → s4, s5 minhlon192 → s6 tienthanhahihi; ~21–22 epoch/phiên, dự kiến xong ~18/10 |
| P4b | GNN-RAG A+ với LM BGE-M3 (ablation, adapted) | như P4 | Đổi duy nhất LM mã hoá câu hỏi + tên quan hệ: MiniLM-L6 (`--lm sbert` gốc) → `BAAI/bge-m3` (LM=bgem3, `add_lm_choice` chèn nhánh y như các nhánh gốc của BERTInstruction; code 14b5edd). `test_rearev_equivalence.py` với LM=bgem3 PASS (loss Δ 0, pred_dist 1.9e-9). Câu bị cắt token (giới hạn số từ + 2 của gốc), test: MiniLM 32/2,783 · BGE-M3 58/2,783. Smoke ✅ (`ritek-gnnrag-bgem3-smoke`, 64 câu/split, 2 epoch): VRAM đỉnh 11.54 GB (MiniLM 9.51) ở micro-batch 4; 2 GPU lệch L2 2.3e-7 = nhiễu nền; 2.86 giây/batch (MiniLM 2.75, +4%) ⇒ ~28.4 phút/epoch, ~23 epoch/phiên, 100 epoch ≈ 49h / 5 phiên. Duyệt 19/10, chạy song song với SubgraphRAG `sqrt`. Phiên 1 ✅ (minhlon192, epoch 0→21, 29.3 phút/epoch): dev H1 tốt nhất 23.86 (MiniLM cùng epoch 24.3). Phiên 2 ✅ (minhduonngg, epoch 21→42, state đầy đủ): dev H1 tốt nhất 24.06 (MiniLM tới epoch 42: 26.0); test tạm F2@5 21.77 · H@1 22.21 · H@5 40.06 · R@20 48.41 · MRR 30.71 (`gnnrag/rearev_fullkg_bgem3_s2`; MiniLM epoch 34: H@1 23.13 · R@20 49.73 · MRR 31.90). Checkpoint chứa cả trọng số LM: 2.27 GB/file, output ~10.5 GB/phiên. **Dừng ở epoch 42/100 theo quyết định 20/10** (không chạy s3–s5): kết quả ablation = bản tạm `gnnrag/rearev_fullkg_bgem3_s2`, chưa đủ số epoch của cấu hình gốc nên chỉ là tham khảo |
| P5 | MoR + BGE-M3 và đối chứng | traversal test ~11 phút | MoR+BGE-M3 (code 53b8416): rerank H@1 25.21 · R@20 46.59 · MRR 34.25; traversal H@1 12.89 · R@20 40.96. Val hit@1 reranker 14.5→30.5. Đối chứng Contriever (reranker train lại) ✅: H@1 35.42 · R@20 63.08 · MRR 46.12 (checkpoint tác giả: 36.09 · 63.05 · 46.76); traversal chạy lại trùng P1 từng query (Δ = 0). ⇒ thay Contriever-FT bằng BGE-M3: H@1 −10.2, R@20 −16.5. |
| P6 | ARK | — | để sau |

Mỗi pha cần GPU: smoke test → probe → bảng ước lượng đo thật → chờ duyệt → chạy.

## 6. Kết quả chính (test, 2,801 query, `sanitized`)

Sinh bằng `python ritek/scripts/main_table.py` (quy ước bản chính: Q9). Chưa có kiểm định cặp giữa các phương pháp.

**Bản chính**

| Phương pháp | Biến thể | f2@5 | hit@1 | hit@5 | recall@10 | recall@20 | mrr |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 | `bm25/add_rel` | 17.93 | 16.39 | 35.13 | 35.00 | 40.42 | 25.10 |
| MedCPT | `cosine_medcpt/add_rel` | 7.57 | 8.78 | 16.17 | 14.86 | 18.58 | 12.51 |
| BGE-M3 | `cosine_bgem3/add_rel` | 12.89 | 12.10 | 27.63 | 25.37 | 30.94 | 19.51 |
| Contriever-FT | `contriever_ft/dotprod` | 22.43 | 24.99 | 45.38 | 43.07 | 50.98 | 34.68 |
| MoR | `mor/full_rerank` | 31.36 | 36.09 | 59.62 | 56.83 | 63.05 | 46.76 |
| GNN-RAG (adapted: toàn KG) | `gnnrag/rearev_fullkg` | 24.27 | 25.92 | 44.91 | 46.03 | 52.90 | 34.73 |
| SubgraphRAG | `subgraphrag/orig_none` | 11.22 | 6.32 | 22.21 | 24.68 | 30.63 | 14.06 |
| ARK | `ark/main` | chưa có | — | — | — | — | — |

**Ablation / đối chứng**

| Phương pháp | Biến thể | f2@5 | hit@1 | hit@5 | recall@10 | recall@20 | mrr |
|---|---|---:|---:|---:|---:|---:|---:|
| MoR traversal, không rerank | `mor/traversal_no_rerank` | 27.25 | 31.13 | 53.34 | 51.49 | 60.14 | 41.53 |
| MoR w/o RT | `mor/wo_RT` | 27.33 | 30.56 | 53.37 | 51.18 | 60.07 | 41.13 |
| MoR w/o RS | `mor/wo_RS` | 18.87 | 22.31 | 38.49 | 34.98 | 39.21 | 29.75 |
| MoR, reranker train lại (đối chứng) | `mor/contriever_retrained_rerank` | 31.05 | 35.42 | 59.23 | 56.89 | 63.08 | 46.12 |
| MoR + BGE-M3 | `mor/bgem3_rerank` | 23.24 | 25.21 | 45.20 | 41.75 | 46.59 | 34.25 |
| GNN-RAG + BGE-M3 (dừng ở epoch 42/100) | `gnnrag/rearev_fullkg_bgem3_s2` | 21.77 | 22.21 | 40.06 | 41.67 | 48.41 | 30.71 |
| SubgraphRAG + pos_weight sqrt (dừng ở epoch 38) | `subgraphrag/orig_sqrt_s2` | 14.33 | 7.25 | 27.95 | 31.03 | 37.31 | 16.88 |

**Phụ lục: độ nhạy corpus và cách gộp chunk**

| Phương pháp | Biến thể | f2@5 | hit@1 | hit@5 | recall@10 | recall@20 | mrr |
|---|---|---:|---:|---:|---:|---:|---:|
| BM25 no_rel | `bm25/no_rel` | 13.44 | 12.78 | 27.95 | 26.01 | 31.21 | 19.89 |
| MedCPT add_rel, gộp max | `cosine_medcpt/add_rel_max` | 9.02 | 9.21 | 19.21 | 17.66 | 22.78 | 14.26 |
| MedCPT no_rel | `cosine_medcpt/no_rel` | 5.49 | 5.46 | 12.21 | 11.06 | 14.69 | 8.94 |
| MedCPT no_rel, gộp max | `cosine_medcpt/no_rel_max` | 6.88 | 6.75 | 15.10 | 13.88 | 17.31 | 10.87 |
| BGE-M3 add_rel, gộp max | `cosine_bgem3/add_rel_max` | 13.00 | 12.07 | 27.92 | 25.63 | 31.32 | 19.59 |
| BGE-M3 no_rel | `cosine_bgem3/no_rel` | 12.04 | 11.21 | 25.71 | 24.17 | 29.05 | 18.20 |
| BGE-M3 no_rel, gộp max | `cosine_bgem3/no_rel_max` | 12.07 | 11.21 | 25.71 | 24.12 | 29.12 | 18.17 |

**Subgraph (test)** — `python ritek/scripts/subgraph_stats.py`. avg answer recall là TRẦN của phương pháp (không xếp hạng được node chưa thấy); đặt cạnh R@20.

| Phương pháp | Q_valid | avg node | avg edge | avg answer recall | full recall rate | answer recall /2,801 | full recall /2,801 | R@20 thật |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SubgraphRAG (PPR-2000 từ topic) | 2,783 | 2,000 | 293,469 | 68.50% | 65.15% | 68.06% | 64.73% | 30.63 |
| GNN-RAG — đầu vào: toàn KG | 2,783 | 129,375 | 4,049,642 | 100.00% | 100.00% | 99.36% | 99.36% | 52.90 |
| GNN-RAG — node có pred_dist ≠ 0 | 2,783 | 123,332 | 1,878,834 | 100.00% | 100.00% | 99.36% | 99.36% | 52.90 |

Q_valid bỏ 18 query test không có topic (không có subgraph); hai cột "/2,801" tính chúng là 0. avg edge = cạnh có cả hai đầu trong subgraph trên đồ thị **vô hướng đã gộp** (4,049,642 cặp node, từ 8,100,498 triple có hướng) — đúng định nghĩa notebook QAFD tham chiếu (`ref_qafd/qafd-rag.ipynb` §8), quyết định 24/10 để QAFD vào cùng bảng. (SubgraphRAG thực tế chấm 586,981 triple có hướng mỗi query.)
