# Siêu tham số HOP: giá trị hiện tại (chỉnh trên PharmKG) và thống kê dữ liệu để chọn lại

Siêu tham số V7 được chỉnh trên PharmKG. Mục đầu là các giá trị đã chọn lại cho PrimeKG và ADint theo EDA; các
bảng sau đặt từng giá trị V7 cạnh đại lượng dữ liệu mà nó phụ thuộc, trên cả ba bộ. Giá trị mới vẫn cần đo lại
sau run đầu tiên và so cặp như quy ước benchmark.

Nguồn số: `tools/hparam_stats.py --sample 500` (seed 42; chỉ train/val, không đụng đáp án test) →
[`hparam_stats.json`](hparam_stats.json). Khoảng cách hop chính xác tính trên mọi đáp án train + val (PharmKG
10.177, Prime 13.066, ADint 4.021); các đại lượng theo đường vàng tính trên 500 câu train mỗi bộ. Đồ thị là đồ
thị V8 (`inverse = missing`). Ngày: 2026-10-27.

## Quyết định theo EDA (2026-10-28)

In đậm là giá trị khác V7. Số căn cứ lấy từ các bảng thống kê bên dưới và từ [docs/EDA.md](../../../docs/EDA.md),
[docs/EDA_ADINT.md](../../../docs/EDA_ADINT.md).

| Siêu tham số | PharmKG | PrimeKG | ADint | Căn cứ |
|---|---|---|---|---|
| `max_hops` | 3 | 3 | **4** | ≤ 3 hop phủ 100% / 99,8% / 90,3% đáp án; ADint ≤ 4 hop phủ 97,9% |
| `label_edge_cap` | 200 | **500** | 200 | Nhóm hop đầu ≤ trần: 98,7% / 83,6% / 100% |
| Trần epoch mọi pha | **60** | **60** | **60** | QA-SFT PharmKG chạm trần 24 khi còn tăng; Prime, ADint ít bước mỗi epoch hơn (283, 217 so với 512) |
| Chọn checkpoint, early stopping | Hits@1, hoà → epoch sớm; patience 3 (V7) | **Hits@1, hoà → Recall@20; patience 5** | **như Prime** | Val 1.024 / 503 / 384 câu: một câu = 0,10 / 0,20 / 0,26pp. Run đủ 2026-10-28: QA-SFT Prime e4 / e6 / e7 cùng Hits@1 0,066 trong khi R@20 0,309 → 0,321 → 0,328, nên dừng ở e7 và chọn e4; ADint dừng QA-SFT ở e7, Setting B ở e6 khi R@20 còn tăng (quyết định 2026-10-28) |
| `local_top_k` | 60 | **250** | 60 | Nhóm cạnh tranh ở nút cha (bảng ngay dưới): Prime phải đúng tới S2 mới ≤ 60 cho 82% đáp án 2 hop, ≤ 250 cho 98% |
| `path_beam` | 30 | **60** | 30 | Đáp án mỗi câu mean / p90: 1,1 / 1 · 2,6 / 7 · 1,0 / 1 |
| `max_pool_nodes` | 1.200 | **3.000** | 1.200 | Prime: 250 con mỗi trạng thái, hàng nghìn ứng viên ở độ sâu 2 và 3 |
| `pool_last_hop` | false (V7) | **true** | **true** | Đáp án ở hop cuối: 0,02% / 26,2% / 7,7% (ADint: hop 4); không bật thì chúng chỉ vào pool qua ≤ `path_beam` đường |
| `pool_depth_quotas` | 48/47/5 (V7) | **43/31/26** | **48/34/10/8** | Tỉ lệ đáp án ở độ sâu 1..`max_hops`, chuẩn hoá: Prime 42,1/30,5/26,2; ADint 46,7/33,1/9,7/7,7. Chỗ thừa của độ sâu ít ứng viên chia tiếp theo cùng tỉ lệ |
| `graph.action` | **target: 182** (schema 552) | **target: 45** (schema 56) | **target: 1.395** (schema 16.732) | Topic cùng loại: hai cách chọn đúng cùng tập cạnh; mọi câu Prime, PharmKG có một topic, 2,9% câu train ADint có topic khác loại (hop đầu lấy tập cạnh rộng hơn). 51% schema ADint có ≤ 5 cạnh trong cả đồ thị. Prime, PharmKG đổi theo ADint ngày 2026-10-29 để ba bộ cùng một định nghĩa action |

Giữ giá trị V7 vì EDA cho thấy đã phù hợp: `route_cap_per_answer` 256, codebook 3 tầng × 64 mã, `synthetic`
10.000 / 10.000 (Prime ~156 ví dụ mỗi mã S1, PharmKG ~113), top-20, rescue 6 slot. EDA không quyết được, giữ V7 và
test sau: learning rate, batch, trọng số loss, prior A/S, `program_beam`, length penalty, trọng số điểm lạnh. Riêng
ADint: tín hiệu từ vựng nằm ở description của đáp án (trùng 25,8% với câu hỏi, tên chỉ 3,6%) trong khi điểm lạnh
dùng để chọn checkpoint đặt trọng số tên 1,8 và text 0,8; so điểm lạnh với ranker trên val sau run đầu tiên.

Sau mỗi run, `pool_cap_drop_by_depth` trong chẩn đoán val (số đáp án bị trần pool loại, theo độ sâu) cho biết có
cần chỉnh quota bằng số đo hay không.

### Nhóm cạnh cạnh tranh ở nút cha của đáp án (căn cứ `local_top_k`)

Khi chương trình Flan đúng, cạnh vàng chỉ còn cạnh tranh với các cạnh ra của nút cha có cùng mức prior: cùng
action, và cùng tiền tố mã S nếu chương trình có S. Tỉ lệ đáp án có nhóm đó ≤ 60 cạnh
(`tools/hub_groups.py` → [`hub_groups.json`](hub_groups.json); 500 câu train mỗi bộ, đường teacher; mã S của Prime fit
lại từ embedding của smoke đúng cấu hình run thật; ADint chưa có codebook nên chỉ có số theo action):

| Nút cha | PharmKG | PrimeKG | ADint |
|---|---:|---:|---:|
| Đáp án 1 hop: bậc topic p50 / p90 | 188 / 812 | 162 / 1.270 | 29 / 230 |
| ↳ cùng action | 87% | 74% | 100% |
| ↳ cùng action + S1S2 | 100% | 97% | — |
| Đáp án 2 hop: bậc nút giữa p50 / p90 | 97 / 312 | 353 / 15.082 | 75 / 13.113 |
| ↳ cùng action | 95% | 52% | 90% |
| ↳ cùng action + S1 | 100% | 55% | — |
| ↳ cùng action + S1S2 (≤ 60 / ≤ 250) | 100% / 100% | 82% / 98% | — |
| ↳ cùng action + S1S2S3 | 100% | 99% | — |

Hub của Prime: 89 nút có hơn 10k cạnh ra, 88 trong số đó là anatomy với gần như toàn bộ cạnh "expression present →
gene/protein", tức cùng một schema; cụm S1 của Prime chứa ~2.000 nút (PharmKG ~113), nên S1 gần như không tách được.

## Quy mô dữ liệu

| | PharmKG | PrimeKG | ADint |
|---|---:|---:|---:|
| Nút | 7.262 | 129.375 | 162.006 |
| Cạnh có hướng (V8) | 807.545 | 8.100.498 | 1.571.054 |
| Cạnh ngược V8 phải thêm | 327.643 | 0 | 489.649 |
| Quan hệ / loại nút | 28 / 7 | 18 / 10 | 15 / 112 |
| Schema / action V8 (số token `<A_i>`; action = quan hệ, chiều, loại đích) | 552 / **182** | 56 / **45** | 16.732 / **1.395** |
| Bậc ra p50 / p90 / p99 / max | 60 / 246 / 915 / 5.913 | 4 / 136 / 990 / 17.355 | 3 / 24 / 74 / 13.113 |
| Train / val / test | 8.186 / 1.024 / 1.023 | 4.522 / 503 / 2.801 | 3.457 / 384 / 482 |
| Bước cập nhật mỗi epoch (batch 16) | 512 | 283 | 217 |

## Siêu tham số và đại lượng liên quan

| Siêu tham số (giá trị hiện tại) | Đại lượng | PharmKG | PrimeKG | ADint | Ghi chú cho Prime |
|---|---|---|---|---|---|
| `max_hops` (V7: 3) | Khoảng cách ngắn nhất thật (BFS toàn đồ thị, mọi đáp án train + val): 0 / 1 / 2 / 3 / > 3 hop | 1,6 / 77,2 / 21,1 / 0,02 / 0%; xa nhất 3 hop | 1,0 / 42,1 / 30,5 / 26,2 / 0,17%; **xa nhất 5 hop** (4 hop: 16 đáp án, 5 hop: 7) | 0,8 / 46,7 / 33,1 / 9,7 / **9,4%** (4: 7,7, 5: 1,6, 6: 0,2) + 0,3% không nối tới; xa nhất 6 hop | Prime: 3 hop phủ 99,8% đáp án; ADint: 3 hop chỉ phủ 90,3%, 4 hop phủ 97,9% |
| Route Setting B ≤ 2 hop | Tỉ lệ đáp án được loss đường phủ | 98,4% | **72,6%** | 79,8% | 26,2% đáp án Prime chỉ được teacher CE phủ |
| `pool_depth_quotas` (V7: 48 / 47 / 5%) | Tỉ lệ đáp án ở độ sâu 1 / 2 / 3 | 77 / 21 / 0% | 42 / 31 / 26% | 47 / 33 / 10% | Độ sâu 3 chỉ vào pool qua 30 đường của beam cuối |
| `label_edge_cap` = 200 (Prime 500) | Số cạnh cùng schema ở hop đầu của đường vàng p50 / p90 / p99; tỉ lệ ≤ 200 / 500 / 1.000 | 9 / 70 / 226; 98,7 / 100 / 100% | 14 / 1.171 / 2.130; 83,0 / 83,6 / 88,4% | 1 / 6 / 137; 100% | Chi phí tính trước khi lọc S, nên trần chặn ở topic hub; đo nhãn thật (60 câu): 76,7 / 80,7 / 81,2% đáp án có nhãn |
| `path_beam` = 30, `local_top_k` = 60 | Bậc ra của topic p50 / p90 / p99 / max | 197 / 822 / 2.316 / 5.913 | 31 / 739 / 1.844 / 16.420 | 24 / 241 / 500 / 13.113 | |
| | Bậc ra của nút giữa trên đường vàng 2 hop p50 / p90 / p99 | 97 / 312 / 1.009 | 353 / **15.082** / 16.513 | 75 / 13.113 / 13.113 | Prime: chọn top-60 trong ~15k cạnh ở hub (PharmKG: 60 trong ~300) |
| `route_cap_per_answer` = 256 | Số route 2 hop mỗi đáp án p50 / p90 / p99 / max; tỉ lệ ≤ 256 | 39 / 151 / 391 / 563; 94,8% | 4 / 53 / 502 / 821; 94,8% | 3 / 16 / 70 / 150; 100% | Đủ cho cả ba bộ |
| `max_pool_nodes` = 1.200, top-k = 20 | Số đáp án mỗi câu mean / p90 / p99 / max | 1,1 / 1 / 3 / 3 | 2,6 / 7 / 17 / 20 | 1,0 / 1 / 2 / 3 | Không câu nào > 20 đáp án; kích thước pool thật xem hard audit của run full |
| Codebook S 3 tầng × 64 mã (262.144 mã đầy đủ) | Nút / mã nếu phân đều | 0,03 | 0,49 | 0,62 | Smoke Prime (codebook smoke 3k mẫu): 9.135 mã đầy đủ được dùng, ~14 nút/mã, nên tiền tố S kém phân biệt hơn PharmKG |
| Action (V7: schema = loại nguồn, quan hệ, chiều, loại đích; V8: bỏ loại nguồn) | Số action schema / target | 552 / 182 | 56 / 45 | 16.732 / 1.395 | ADint: schema thêm 16,7k token vào vocab Flan; V7 từng có mode `relation` (quan hệ × chiều, 30 action) mà V8 đã bỏ |
| `synthetic.nodes` / `synthetic.edges` = 10.000 / 10.000 | Tỉ lệ nút / cạnh được phủ | 100% / 1,24% | 7,7% / 0,12% | 6,2% / 0,64% | ADint: riêng đại diện mỗi schema đã là 16.732 ví dụ cạnh |
| epochs 24 / patience 3 / batch 16 | Bước cập nhật mỗi epoch | 512 | 283 | 217 | Prime ~½ số bước mỗi epoch của PharmKG, nên patience 3 epoch ứng với ít bước hơn. Ngay trên PharmKG, QA-SFT chạm trần: cả hai nhánh ablation `lm_head` (2026-10-28) tốt nhất ở epoch 23/24, Hits@1 val còn tăng 0,5–0,75pp/epoch ở epoch 19–23 |
| `program_beam` 5, `decoder_length_penalty` 2.0 | Độ dài chương trình nhãn | — | — | — | Lấy từ `flan/labels/qa_label_audit.json` của run full |

## Hằng số trong code (chưa từng chỉnh ngoài PharmKG)

| Hằng số | Giá trị | File |
|---|---|---|
| Prior A/S cộng vào logit cạnh | +1,2·khớp action, +1,05·khớp S, −2,1·lùi S, −1,3·lệch action | `src/hop/policy.py` |
| Điểm lạnh (proxy chọn checkpoint, nền ranker) | 1,8 tên + 0,8 văn bản + 0,8 quan hệ + 0,25 điểm đường + 0,15 khớp action − 0,10 độ sâu + 0,10 topic | `src/hop/ranker.py` |
| Loss ranker | 0,40 top-1 + 0,45 top-20 (biên 0,2) + 0,15 nhiễu khó (biên 0,15); 32 + 32 nhiễu khó | `src/hop/ranker.py` |
| Loss Setting B | CE 1,0 · rank 0,15 · đường 0,2 | `configs/default.json` |
| Cắt beam | trùng tiền tố tối đa 5 (topic) / 2 / 1 (độ sâu cuối); tối thiểu 3 slot mỗi beam có hop; slot phụ theo √(xác suất + 0,01); rescue 6 slot, xác suất 0,1·khối lượng terminal | `src/hop/search.py` |
| Route trong pool | 3 route tốt nhất mỗi nút | `src/hop/ranker.py` |
