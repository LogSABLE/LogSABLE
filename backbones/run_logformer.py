"""LogFormer: official model.py and its defaults (1 layer, adapter 64, window 20, lr 1e-5, 5 pretrain and 20 tuning epochs,
BCEWithLogits, distilbert-base-nli-mean-tokens embeddings). Pre-train on a source dataset, then tune adapters on the target."""
import os, random, sys, time
import numpy as np, torch, torch.nn as nn
B = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(B, "logformer_official")); sys.path.insert(0, B)
import model as _lf
from model import Model
# Compatibility only: newer PyTorch passes is_causal to encoder layers; the official layer predates it. Model code is unchanged.
_orig_forward = _lf.TransformerEncoderLayer.forward
_lf.TransformerEncoderLayer.forward = lambda self, src, src_mask=None, src_key_padding_mask=None, **kw: _orig_forward(self, src, src_mask, src_key_padding_mask)
from common import load, save
from sentence_transformers import SentenceTransformer
SOURCE = {"hdfs": "bgl", "bgl": "hdfs", "liberty": "bgl"}
dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(123); np.random.seed(123); random.seed(123)
enc = SentenceTransformer("distilbert-base-nli-mean-tokens", device="cpu")
def tensors(ds):
    ids, y, split, keys, vocab = load(ds)
    emb = np.vstack([enc.encode(vocab, batch_size=64, show_progress_bar=False), np.zeros((1, 768))]).astype(np.float32)
    return torch.tensor(emb), torch.tensor(ids, dtype=torch.long), y, split, keys
def batches(emb, ids, y, idx, bs=64, shuffle=True):
    idx = idx.copy()
    if shuffle: np.random.shuffle(idx)
    for i in range(0, len(idx), bs):
        j = idx[i:i + bs]
        yield emb[ids[j]].to(dev), (torch.tensor(np.eye(2, dtype=np.float32)[y[j]]).to(dev) if y is not None else None)
def train(model, emb, ids, y, idx, epochs, lr=1e-5):
    ps = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(ps, lr=lr)
    steps = int(np.ceil(len(idx) / 64))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, epochs=epochs, steps_per_epoch=steps)
    crit = nn.BCEWithLogitsLoss(); model.train()
    for ep in range(epochs):
        t = time.time(); tot = 0
        for x, yy in batches(emb, ids, y, idx):
            loss = crit(model(x), yy); opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(ps, 0.5); opt.step(); sch.step(); tot += loss.item()
        print(f"  epoch {ep+1}/{epochs} loss {tot/steps:.4f} {time.time()-t:.0f}s", flush=True)
pre = {}
for ds in sys.argv[1:]:
    src = SOURCE[ds]
    if src not in pre:
        emb, ids, y, split, _ = tensors(src)
        m = Model(mode="classifier", num_layers=1, adapter_size=64, dim=768, window_size=20, nhead=8, dim_feedforward=4 * 768, dropout=0.1).to(dev)
        print("pretrain on", src, flush=True); train(m, emb, ids, y, np.where(split == "train")[0], epochs=5)
        pre[src] = {k: v.cpu() for k, v in m.state_dict().items() if not k.startswith("fc1")}
    emb, ids, y, split, keys = tensors(ds)
    m = Model(mode="adapter", num_layers=1, adapter_size=64, dim=768, window_size=20, nhead=8, dim_feedforward=4 * 768, dropout=0.1)
    print("loaded", m.load_state_dict(pre[src], strict=False).unexpected_keys, flush=True)
    m = m.to(dev); m.train_adapter()
    print("tune on", ds, flush=True); train(m, emb, ids, y, np.where(split == "train")[0], epochs=20)
    m.eval(); out = np.zeros(len(y), dtype=np.float32); all_idx = np.arange(len(y))
    with torch.no_grad():
        for i in range(0, len(y), 256):
            j = all_idx[i:i + 256]
            out[j] = torch.softmax(m(emb[ids[j]].to(dev)), 1)[:, 1].cpu().numpy()
    save("logformer", ds, keys, split, y, out)
