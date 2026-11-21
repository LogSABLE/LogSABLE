import json, os, random, sys, time
import numpy as np, torch
B = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, B)
from common import load, f1
from transformers import AutoTokenizer, LongformerForMaskedLM
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
NAME, MAX_LEN, TRAIN_N, EPOCHS, LR, K, CW, BS = "allenai/longformer-base-4096", 1024, 6000, 3, 1e-4, 5, 0.25, 4
tok = AutoTokenizer.from_pretrained(NAME); SEP = tok("\n", add_special_tokens=False)["input_ids"]
def encode(lines, rng):
    ids, lab = [tok.cls_token_id], [-100]
    n_mask = rng.randint(0, max(0, int(0.75 * len(lines))))
    chosen = set(rng.sample(range(len(lines)), n_mask)) if n_mask else set()
    for i, l in enumerate(lines):
        for t in tok(l, add_special_tokens=False)["input_ids"]:
            if i in chosen:
                r = rng.random(); lab.append(t)
                ids.append(tok.mask_token_id if r < 0.8 else (rng.randint(5, tok.vocab_size - 1) if r < 0.9 else t))
            else:
                ids.append(t); lab.append(-100)
        ids += SEP; lab += [-100] * len(SEP)
    ids, lab = ids[:MAX_LEN - 1] + [tok.sep_token_id], lab[:MAX_LEN - 1] + [-100]
    return ids, lab
def batch(seqs):
    m = max(len(a) for a, _ in seqs)
    x = torch.full((len(seqs), m), tok.pad_token_id); y = torch.full((len(seqs), m), -100); a = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, (s, l) in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s); y[i, :len(l)] = torch.tensor(l); a[i, :len(s)] = 1
    g = torch.zeros_like(a); g[:, 0] = 1
    return x.to(dev), y.to(dev), a.to(dev), g.to(dev)
for ds in sys.argv[1:]:
    torch.manual_seed(123); np.random.seed(123); rng = random.Random(123)
    ids, y, split, keys, vocab = load(ds)
    paras = [[vocab[i] for i in r] for r in ids]
    texts = ["\n".join(p) for p in paras]
    uniq = {t: i for i, t in enumerate(dict.fromkeys(texts))}; ut = list(uniq); inv = np.array([uniq[t] for t in texts])
    print(ds, "sessions", len(texts), "unique", len(ut), "device", dev, flush=True)
    model = LongformerForMaskedLM.from_pretrained(NAME).to(dev)
    tr = np.where((split == "train") & (y == 0))[0]; rs = np.random.RandomState(123); tr = rs.choice(tr, min(TRAIN_N, len(tr)), replace=False)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.99), eps=1e-5, weight_decay=0.01)
    centroid = None
    for ep in range(EPOCHS):
        model.train(); rs.shuffle(tr); t = time.time(); tot = 0; n = 0; cvs = []
        for i in range(0, len(tr), BS):
            x, lab, a, g = batch([encode(paras[j], rng) for j in tr[i:i + BS]])
            o = model(input_ids=x, attention_mask=a, global_attention_mask=g, labels=lab, output_hidden_states=True)
            cv = o.hidden_states[-1][:, 0]; cvs.append(cv.detach().cpu())
            loss = o.loss if centroid is None else o.loss + CW * ((cv - centroid.to(dev)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); n += 1
        centroid = torch.cat(cvs).mean(0)
        print(f"  epoch {ep+1}/{EPOCHS} loss {tot/n:.4f} {time.time()-t:.0f}s", flush=True)
    model.eval(); acc = np.zeros(len(ut), dtype=np.float32); dist = np.zeros(len(ut), dtype=np.float32); order = np.argsort([len(t) for t in ut])
    lines = {t: t.split("\n") for t in ut}
    with torch.no_grad():
        for i in range(0, len(ut), 4):
            j = order[i:i + 4]
            x, lab, a, g = batch([encode(lines[ut[q]], rng) for q in j])
            o = model(input_ids=x, attention_mask=a, global_attention_mask=g, output_hidden_states=True)
            m = lab != -100; top = o.logits[m].topk(K, dim=-1).indices; h = (top == lab[m].unsqueeze(-1)).any(-1)
            hit = torch.zeros_like(m); hit[m] = h
            acc[j] = ((hit & m).sum(1).float() / m.sum(1).clamp(min=1)).cpu().numpy()
            dist[j] = ((o.hidden_states[-1][:, 0].cpu() - centroid) ** 2).sum(1).sqrt().numpy()
    A, D = acc[inv], dist[inv]; va, te = split == "val", split == "test"
    normal_radius = float(np.mean(D[(split == "train") & (y == 0)]))
    best = None
    for th in np.arange(0.65, 1.0, 0.02):
        for mult in np.arange(1.2, 1.95, 0.1):
            p = ((A < th) | (D > mult * normal_radius)).astype(int); s = f1(y[va], p[va])[2]
            if best is None or s > best[0]: best = (s, float(th), float(mult))
    pred = ((A < best[1]) | (D > best[2] * normal_radius)).astype(int)
    out = {"model": "logfit", "dataset": ds, "threshold": best[1:], "val_prf": f1(y[va], pred[va]), "test_prf": f1(y[te], pred[te]),
           "keys_va": keys[va].tolist(), "keys_te": keys[te].tolist(), "pred_va": pred[va].tolist(), "pred_te": pred[te].tolist()}
    os.makedirs(os.path.join(B, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(B, "out", f"logfit_{ds}.json"), "w"))
    print("logfit", ds, "acc_th, dist_mult", best[1:], "val", [round(v, 3) for v in out["val_prf"]], "test", [round(v, 3) for v in out["test_prf"]], flush=True)
