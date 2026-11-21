import os, re, sys
import numpy as np, pandas as pd
sys.dont_write_bytecode = True
L = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(os.path.dirname(L))
sys.path.insert(0, os.path.join(R, "src"))
from logsable.data import load_hdfs_sequences, load_bgl_sequences, load_liberty_sequences, make_splits
hdr = re.compile(r'^\S+\s+\S+\s+\w{3}\s+\d+\s+[\d:]+\s+\S+\s+')
for ds in ["Liberty", "BGL", "HDFS"]:
    if ds == "HDFS":
        raw, tmpl, seqs = load_hdfs_sequences(os.path.join(R, "hdfs_data")); kc = "BlockId"; lines = raw.groupby("BlockId", sort=False)["Content"].apply(list).to_dict()
    elif ds == "BGL":
        raw, tmpl, seqs = load_bgl_sequences(os.path.join(R, "bgl")); kc = "BlockId"; lines = raw.groupby("SeqKey", sort=False)["Content"].apply(list).to_dict()
    else:
        raw, tmpl, seqs = load_liberty_sequences(os.path.join(R, "liberty"), bucket_sec=100, filename="liberty_150k.csv"); kc = "seq_key"
        raw["_c"] = raw["log"].astype(str).map(lambda l: hdr.sub("", l)); lines = raw.groupby("_seq_key", sort=False)["_c"].apply(list).to_dict()
    used = seqs[seqs["EventSeq"].apply(lambda s: len(s or []) >= 20)].reset_index(drop=True)
    y = used["Label"].astype(int).values; keys = used[kc].astype(str).values
    idx = np.arange(len(y))
    (tr, _, _), (va, _, _), (te, _, _) = make_splits(idx, y, keys, dict(zip(keys, used["StartTime"])), 0.10, 0.20)
    bad = sum(len(lines[k]) != len(s) for k, s in zip(keys, used["EventSeq"]))
    out = os.path.join(L, "data", ds); os.makedirs(out, exist_ok=True)
    for name, ix in [("train", tr), ("val", va), ("test", te)]:
        pd.DataFrame({"key": [keys[i] for i in ix], "Label": y[ix], "Content": [" ;-; ".join(map(str, lines[keys[i]])).replace("\n", " ") for i in ix]}).to_csv(f"{out}/{name}.csv", index=False)
    n = np.array([len(lines[k]) for k in keys])
    print(ds, "sessions", len(y), "train/val/test", len(tr), len(va), len(te), "anomalies", y[tr].sum(), y[va].sum(), y[te].sum(), "length mismatches", bad, "median/mean/p95 lines", np.median(n), round(n.mean(), 1), np.percentile(n, 95), flush=True)
