"""HOP PoLo V8: chương trình nhảy (Flan-T5) + policy cạnh toàn láng giềng + pool ranker.

Bản V8 giữ nguyên thuật toán của V7 (Setting B), bỏ các bước không dùng tới (mã cấu trúc G,
điểm old-score bị gán 0, các mode executor khác `semantic`) và chạy được trên PharmKG, ADint,
PrimeKG (RiTeK) từ cùng một định dạng bundle đầu vào (xem `tools/prep_inputs.py`).
"""
