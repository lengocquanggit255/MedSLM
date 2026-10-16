"""P5 (ritek/PLAN.md) — MoR train lại reranker trên split RiTeK, scorer Contriever-FT hoặc BGE-M3.

SCORER=contriever : run đối chứng — giống MoR gốc, chỉ khác reranker train lại trên RiTeK train/val.
SCORER=bgem3      : thay embedding của scorer bằng BGE-M3 dense (doc no_rel, cửa sổ 8192).

Trong MoR-prime, Contriever-FT chỉ có MỘT chỗ dùng: `scorer` (Reasoning/text_retrievers/contriever.py
`score()` = dot product giữa query_emb_dict và candidate_emb_dict tính sẵn). Điểm đó vừa chọn
top-100 candidate (`check_topk` / `check_negtopk`) vừa là cột cuối của score_vector mà reranker
học. Text retriever của MoR-prime là BM25 (mặc định `--text_retriever_name bm25`), không phải
Contriever. Vì vậy "thay Contriever bằng BGE-M3" = thay hai dict embedding, không sửa code MoR.

Luồng (mỗi bước gọi script gốc của MoR, như kaggle/mor_run/mor_run.py):
  1. eval.py --mod train  -> prime_train.pkl  (RiTeK train 4,547; check_negtopk như code gốc)
  2. eval.py --mod val    -> prime_val.pkl    (RiTeK val 505; check_topk)
  3. eval.py --mod test   -> prime_test.pkl   (2,801)
  4. Reranking/train_eval_path_prime.py — 100 epoch, chọn theo val hit@1 (code gốc)
  5. Reranking/rerank.py với checkpoint vừa train -> mor_rerank_pred.pkl
Kế hoạch của planner lấy từ cache có sẵn của repo (không gọi LLM). RiTeK train và val đều là tập
con của STaRK train, nên cả hai lấy từ `cache/prime/path/train_20250222.pkl`, lọc theo id.

Bản vá vào repo gốc: giống mor_run.py (import Planner lười, load_skb có root, stub module không
dùng, assert khớp vị trí cache kế hoạch) + lọc split RiTeK + wandb no-op + num_workers=2.

Dataset: mor-package (code MoR) · ritek-mor-data (ritek/scripts/prep_mor_data.py).
SMOKE=1: LIMIT query mỗi split, 2 epoch.
"""
import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys

SCORER = os.environ.get("SCORER", "contriever")
SMOKE = os.environ.get("SMOKE", "0") == "1"
LIMIT = int(os.environ.get("LIMIT", "40" if SMOKE else "0"))
EPOCHS = int(os.environ.get("EPOCHS", "2" if SMOKE else "100"))
# batch train mặc định của code gốc là 256 và drop_last=True: smoke vài chục query sẽ không có batch nào.
TRAIN_BS = int(os.environ.get("TRAIN_BS", "16" if SMOKE else "256"))
assert SCORER in ("contriever", "bgem3"), SCORER
OUT = "/kaggle/working"
WORK = "/kaggle/temp/MoR"            # không để trong /kaggle/working: khỏi tải về cả repo + SKB
DATA_ROOT = "/kaggle/temp/starkdata"


def log(*a):
    print(*a, flush=True)


def run(cmd, cwd, env):
    log(f"\n===== {' '.join(cmd)} (cwd={cwd}) =====")
    r = subprocess.run(cmd, cwd=cwd, env=env)
    if r.returncode != 0:
        sys.exit(f"lỗi returncode={r.returncode}: {cmd}")


_m = [os.path.dirname(os.path.dirname(p))
      for p in sorted(glob.glob("/kaggle/input/**/Reasoning/mor4path.py", recursive=True))]
_m = [d for d in _m if os.path.exists(f"{d}/eval.py")]
assert _m, "không thấy code MoR (mor-package) trong /kaggle/input"
MOR_SRC = _m[0]
_d = sorted(glob.glob("/kaggle/input/**/ritek_mor_data.txt", recursive=True))
assert _d, "không thấy dataset ritek-mor-data"
DATA = os.path.dirname(_d[0])
log(f"MoR={MOR_SRC}\ndata={DATA}\nSCORER={SCORER} SMOKE={SMOKE} LIMIT={LIMIT} EPOCHS={EPOCHS} TRAIN_BS={TRAIN_BS}")

# Checkpoint Contriever-FT: get_contriever() nạp nó trong __init__ của scorer dù score() chỉ dùng dict.
CKPT_SHA = "5604e310687c5b2a98485072b0a08fc5d0246ba88ddcc86295794daf7f806300"
from huggingface_hub import hf_hub_download  # noqa: E402
CKPT = hf_hub_download("GagaLey/MoR", "Reasoning/text_retrievers/model_checkpoint/contriever/prime/"
                       "best_20250204.pth", local_dir="/kaggle/temp/hf")
_h = hashlib.sha256()
with open(CKPT, "rb") as _f:
    for _b in iter(lambda: _f.read(1 << 22), b""):
        _h.update(_b)
assert _h.hexdigest() == CKPT_SHA, f"checkpoint sai sha256: {_h.hexdigest()}"

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bm25s", "torch_geometric"], check=False)
import importlib.util  # noqa: E402
for _n in ["torch_geometric", "bm25s"]:
    assert importlib.util.find_spec(_n) is not None, f"thiếu {_n}"

shutil.copytree(MOR_SRC, WORK, dirs_exist_ok=True)
os.makedirs(f"{WORK}/Reasoning/text_retrievers/model_checkpoint/contriever/prime", exist_ok=True)
os.symlink(CKPT, f"{WORK}/Reasoning/text_retrievers/model_checkpoint/contriever/prime/best_20250204.pth")
os.makedirs(DATA_ROOT, exist_ok=True)

# ---------------------------------------------------------------- embedding của scorer
# Chạy ở tiến trình con: không tạo CUDA context ở tiến trình cha (đã gây OOM ở GĐ4).
EMB = "/kaggle/temp/emb"
os.makedirs(EMB, exist_ok=True)
MK = r'''
import sys, numpy as np, torch
data, scorer, out = sys.argv[1:4]
if scorer == "contriever":
    D = np.load(f"{data}/contriever/candidate_emb.npy").astype(np.float32)
    Q = torch.load(f"{data}/contriever/query_emb_dict.pt", map_location="cpu", weights_only=False)
else:
    # BGE-M3 dense: vector node = trung bình chunk đã L2-normalize rồi normalize lại
    # (ritek/scripts/prep_mor_data.py). fp32 như dict Contriever của MoR.
    D = np.load(f"{data}/bgem3/nodeemb_no_rel.npy").astype(np.float32)
    qe = np.load(f"{data}/bgem3/qemb.npy").astype(np.float32)
    qi = np.load(f"{data}/bgem3/qids_all.npy")
    Q = {int(q): torch.from_numpy(qe[i]) for i, q in enumerate(qi)}
assert D.shape[0] == 129375 and np.isfinite(D).all()
torch.save({i: torch.from_numpy(D[i]) for i in range(len(D))}, f"{out}/candidate_emb_dict.pt")
torch.save({int(k): v for k, v in Q.items()}, f"{out}/query_emb_dict.pt")
print(f"scorer {scorer}: candidate {D.shape}, query {len(Q)}", flush=True)
'''
open("/kaggle/temp/mk_emb.py", "w").write(MK)
run([sys.executable, "-u", "/kaggle/temp/mk_emb.py", DATA, SCORER, EMB], "/kaggle/temp", dict(os.environ))
for sub, fn in [("doc_no_rel_no_compact", "candidate_emb_dict.pt"), ("query_no_rel_no_compact", "query_emb_dict.pt")]:
    d = f"{WORK}/Reasoning/data/emb/prime/contriever/{sub}"
    os.makedirs(d, exist_ok=True)
    os.symlink(f"{EMB}/{fn}", f"{d}/{fn}")

# ------------------------------------------------------------------------- vá eval.py
p = f"{WORK}/eval.py"
src = open(p, encoding="utf-8").read()


def sub1(old, new):
    global src
    assert src.count(old) == 1, f"eval.py: kỳ vọng đúng 1 chỗ '{old[:60]}', thấy {src.count(old)}"
    src = src.replace(old, new)


sub1("from Planning.model import Planner\n", "")
sub1("    reasoner = Planner(dataset_name)\n", "    reasoner = None  # planner bỏ qua: dùng cache kế hoạch có sẵn\n")
sub1("    skb = load_skb(dataset_name)", f"    skb = load_skb(dataset_name, root='{DATA_ROOT}', download_processed=True)")
sub1("    qa = load_qa(dataset_name, human_generated_eval=False)", f"    qa = load_qa(dataset_name, root='{DATA_ROOT}')")
# RiTeK train/val ⊂ STaRK train: dùng index + cache kế hoạch của STaRK train rồi lọc theo id.
sub1("    all_indices = split_idx[mod].tolist()",
     "    _src_mod = 'train' if mod in ('train', 'val') else mod\n"
     "    all_indices = split_idx[_src_mod].tolist()")
assert src.count('plan_cache_path = f"./cache/{dataset_name}/path/{mod}_20250222.pkl"') == 2
src = src.replace('plan_cache_path = f"./cache/{dataset_name}/path/{mod}_20250222.pkl"',
                  'plan_cache_path = f"./cache/{dataset_name}/path/{_src_mod}_20250222.pkl"', 1)
sub1("    # ***** Reasoning *****", f"""    # [VÁ] cache kế hoạch phải khớp all_indices THEO VỊ TRÍ (eval.py giả định vậy).
    assert len(plan_output_list) == len(all_indices), (len(plan_output_list), len(all_indices))
    for _k, _i in enumerate(all_indices):
        assert plan_output_list[_k]['q_id'] == qa[_i][1], f"LỆCH VỊ TRÍ tại {{_k}}"
    # [VÁ] lọc theo split RiTeK, giữ nguyên thứ tự và vị trí song song
    _ids = [int(x) for x in open('{DATA}/splits/' + mod + '.index').read().split()]
    _keep = set(_ids)
    _sel = [_k for _k, _i in enumerate(all_indices) if int(qa[_i][1]) in _keep]
    assert len(_sel) == len(_keep), f"split RiTeK {{mod}}: {{len(_keep)}} id, chỉ khớp {{len(_sel)}}"
    _sel = _sel[:{LIMIT}] if {LIMIT} else _sel
    all_indices = [all_indices[_k] for _k in _sel]
    plan_output_list = [plan_output_list[_k] for _k in _sel]
    print(f"[RiTeK] {{mod}}: {{len(all_indices)}} query (cache {{_src_mod}})", flush=True)

    # ***** Reasoning *****""")
sub1('''    save_path = f"{dataset_name}_{mod}.pkl"''',
     '''    if mod == "test":
        with open("/kaggle/working/mor_pred_test.pkl", "wb") as _f:
            pkl.dump([{'q_id': o['q_id'], 'pred_dict': o['pred_dict']} for o in outputs], _f)
    save_path = f"{dataset_name}_{mod}.pkl"''')
open(p, "w", encoding="utf-8").write(src)

# ------------------------------------------------------- stub + vá reranker
STUB = """import importlib, sys, types


class _M(types.ModuleType):
    def __getattr__(self, n):
        if n.startswith('__'):
            raise AttributeError(n)
        return type(n, (), {})


# Chỉ stub module VẮNG MẶT, không nằm trên đường chạy prime + bm25 + contriever (xem mor_run.py).
for _n in ['tdc', 'tdc.resource', 'ogb', 'ogb.nodeproppred', 'ogb.utils', 'ogb.utils.url',
           'gdown', 'langdetect', 'langchain', 'langchain.text_splitter',
           'anthropic', 'voyageai', 'gritlm', 'llm2vec', 'colbert', 'colbert.infra', 'colbert.data']:
    try:
        importlib.import_module(_n)
        continue
    except Exception:
        pass
    _m = _M(_n)
    sys.modules[_n] = _m
    if '.' in _n:
        _p, _c = _n.rsplit('.', 1)
        if _p in sys.modules:
            setattr(sys.modules[_p], _c, _m)
"""
# wandb: chỉ dùng để log; no-op để khỏi đăng nhập/mạng.
os.makedirs("/kaggle/temp/stubs", exist_ok=True)
open("/kaggle/temp/stubs/wandb.py", "w").write("def init(*a, **k):\n    pass\n\n\ndef log(*a, **k):\n    pass\n")

for f in ["eval.py", "Reranking/rerank.py", "Reranking/train_eval_path_prime.py"]:
    fp = f"{WORK}/{f}"
    t = STUB + open(fp, encoding="utf-8").read()
    if f.startswith("Reranking/"):
        assert t.count("skb = load_skb(dataset_name)") == 1, f
        t = t.replace("skb = load_skb(dataset_name)",
                      f"skb = load_skb(dataset_name, root='{DATA_ROOT}', download_processed=True)")
        t = re.sub(r"num_workers=\d+", "num_workers=2", t)      # 32 dễ treo trên Kaggle
    if f == "Reranking/rerank.py":
        old = '    print(f"Results: {avg_results}")'
        assert t.count(old) == 1
        t = t.replace(old, old + """
    import pickle as _pkl
    with open("/kaggle/working/mor_rerank_pred.pkl", "wb") as _f:
        _pkl.dump({"pred_ids": pred_list, "scores": scores_cand_list,
                   "ans_ids": [list(a) for a in ans_ids_list]}, _f)
    print("đã lưu /kaggle/working/mor_rerank_pred.pkl", flush=True)""")
    open(fp, "w", encoding="utf-8").write(t)
log("đã vá eval.py, rerank.py, train_eval_path_prime.py")

env = dict(os.environ, PYTHONPATH=f"/kaggle/temp/stubs:{WORK}", TOKENIZERS_PARALLELISM="false",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", CUDA_VISIBLE_DEVICES="1", WANDB_MODE="disabled")
for mod in ("train", "val", "test"):
    run([sys.executable, "-u", "eval.py", "--dataset_name", "prime", "--scorer_name", "contriever",
         "--mod", mod], WORK, env)
shutil.copy(f"{WORK}/prime_test.pkl", f"{OUT}/prime_test.pkl")

env_tr = dict(env, CUDA_VISIBLE_DEVICES="0,1")
run([sys.executable, "-u", "train_eval_path_prime.py", "--epochs", str(EPOCHS), "--train_batch_size", str(TRAIN_BS)],
    f"{WORK}/Reranking", env_tr)
cks = glob.glob(f"{WORK}/Reranking/data/checkpoints/prime/path/best_*.pth")
assert cks, "train không lưu checkpoint best_*.pth"
best = max(cks, key=lambda x: float(os.path.basename(x)[len("best_"):-len(".pth")]))
log(f"checkpoint tốt nhất theo val hit@1: {best}")
os.makedirs("/kaggle/temp/ckpt/prime", exist_ok=True)
shutil.copy(best, "/kaggle/temp/ckpt/prime/best.pth")
shutil.copy(best, f"{OUT}/reranker_{SCORER}_{os.path.basename(best)}")
for j in glob.glob(f"{WORK}/Reranking/data/outputs/prime/results_*.json"):
    shutil.copy(j, f"{OUT}/train_{os.path.basename(j)}")

run([sys.executable, "-u", "rerank.py", "--dataset_name", "prime", "--checkpoint_path", "/kaggle/temp/ckpt"],
    f"{WORK}/Reranking", env)
log("MOR_RITEK_DONE")
