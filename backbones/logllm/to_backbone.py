"""Turns LogLLM val/test predictions into the backbone format used by fuse.py.
Usage: python backbones/logllm/to_backbone.py <Liberty|BGL|HDFS>"""
import json, os, sys
import numpy as np, pandas as pd
L = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(L)
ds = sys.argv[1]
def load(split):
    p = np.load(os.path.join(L, "runs", ds, f"{split}_preds.npy")).astype(int).tolist()
    d = pd.read_csv(os.path.join(L, "data", ds, f"{split}.csv"), usecols=["key", "Label"])
    assert len(p) == len(d); return d["key"].astype(str).tolist(), p, d["Label"].astype(int).tolist()
kv, pv, yv = load("val"); kt, pt, yt = load("test")
def prf(y, p):
    y, p = np.array(y), np.array(p); tp = ((p == 1) & (y == 1)).sum(); P = tp / max(p.sum(), 1); R = tp / max(y.sum(), 1); return [float(P), float(R), float(2 * P * R / max(P + R, 1e-12))]
out = {"model": "logllm", "dataset": ds.lower(), "threshold": None, "val_prf": prf(yv, pv), "test_prf": prf(yt, pt), "keys_va": kv, "keys_te": kt, "pred_va": pv, "pred_te": pt}
os.makedirs(os.path.join(B, "out"), exist_ok=True)
json.dump(out, open(os.path.join(B, "out", f"logllm_{ds.lower()}.json"), "w"))
print("val P/R/F1 %.3f %.3f %.3f" % tuple(out["val_prf"]), "test P/R/F1 %.3f %.3f %.3f" % tuple(out["test_prf"]))
