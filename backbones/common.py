import json, os
import numpy as np
B = os.path.dirname(os.path.abspath(__file__))
def load(ds):
    d = np.load(os.path.join(B, "data", f"{ds}.npz"), allow_pickle=False)
    vocab = json.load(open(os.path.join(B, "data", f"{ds}_vocab.json")))
    return d["ids"], d["y"], d["split"], d["keys"], vocab
def f1(y, p):
    tp = ((y == 1) & (p == 1)).sum(); fp = ((y == 0) & (p == 1)).sum(); fn = ((y == 1) & (p == 0)).sum()
    pr = tp / max(1, tp + fp); re = tp / max(1, tp + fn)
    return float(pr), float(re), float(2 * pr * re / max(1e-9, pr + re))
def best_threshold(y_va, s_va):
    cands = np.unique(np.quantile(s_va, np.linspace(0, 1, 401)))
    best = max(cands, key=lambda t: f1(y_va, (s_va >= t).astype(int))[2])
    return float(best)
def save(name, ds, keys, split, y, score):
    va, te = split == "val", split == "test"
    th = best_threshold(y[va], score[va])
    pred = (score >= th).astype(int)
    out = {"model": name, "dataset": ds, "threshold": th, "val_prf": f1(y[va], pred[va]), "test_prf": f1(y[te], pred[te]),
           "keys_va": keys[va].tolist(), "keys_te": keys[te].tolist(), "pred_va": pred[va].tolist(), "pred_te": pred[te].tolist()}
    os.makedirs(os.path.join(B, "out"), exist_ok=True)
    json.dump(out, open(os.path.join(B, "out", f"{name}_{ds}.json"), "w"))
    print(name, ds, "threshold", round(th, 4), "val P/R/F1", [round(x, 3) for x in out["val_prf"]], "test P/R/F1", [round(x, 3) for x in out["test_prf"]], flush=True)
