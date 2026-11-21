"""Exports each session's first 20 log templates, label and split. Uses the same split as the pipeline."""
import json, os, re, sys
import numpy as np
sys.dont_write_bytecode = True
B = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(B)
sys.path.insert(0, os.path.join(R, "src"))
from logsable.data import load_hdfs_sequences, load_bgl_sequences, load_liberty_sequences, make_splits
hdr = re.compile(r'^\S+\s+\S+\s+\w{3}\s+\d+\s+[\d:]+\s+\S+\s+')
def mask(l):
    m = hdr.sub('', l); m = re.sub(r'\[\d+\]', '[<*>]', m); m = re.sub(r'\b\d{1,3}(\.\d{1,3}){3}\b', '<*>', m)
    m = re.sub(r'0x[0-9a-fA-F]+|\b[0-9a-f]{6,}\b', '<*>', m); return re.sub(r'\d+', '<*>', m)
def split_sessions(keys, y, start_time):
    (_, _, va), (_, _, te) = make_splits(np.arange(len(keys)), y, keys, start_time, 0.10, 0.20)[1:]
    split = np.array(["train"] * len(keys), dtype=object)
    split[np.isin(keys, va)] = "val"; split[np.isin(keys, te)] = "test"
    return split
os.makedirs(os.path.join(B, "data"), exist_ok=True)
for ds in ["Liberty", "BGL", "HDFS"]:
    if ds == "HDFS": raw, tmpl, seqs = load_hdfs_sequences(os.path.join(R, "hdfs_data")); kc = "BlockId"
    elif ds == "BGL": raw, tmpl, seqs = load_bgl_sequences(os.path.join(R, "bgl")); kc = "BlockId"
    else: raw, tmpl, seqs = load_liberty_sequences(os.path.join(R, "liberty"), bucket_sec=100, filename="liberty_150k.csv"); kc = "seq_key"
    used = seqs[seqs["EventSeq"].apply(lambda s: len(s or []) >= 20)].reset_index(drop=True)
    y = used["Label"].astype(int).values
    keys = used[kc].astype(str).values
    split = split_sessions(keys, y, dict(zip(keys, used["StartTime"])))
    if ds == "Liberty":
        rid2t = {i: mask(t) for i, t in enumerate(tmpl["EventTemplate"].astype(str))}
    else:
        rid2t = {i: t for i, t in enumerate(tmpl["EventTemplate"].astype(str))}
    rows = [[rid2t.get(int(r), "") for r in s[:20]] for s in used["EventSeq"]]
    vocab = sorted({t for r in rows for t in r}); v = {t: i for i, t in enumerate(vocab)}
    np.savez_compressed(os.path.join(B, "data", f"{ds.lower()}.npz"), ids=np.array([[v[t] for t in r] for r in rows], dtype=np.int32),
                        y=y, split=split.astype(str), keys=keys)
    json.dump(vocab, open(os.path.join(B, "data", f"{ds.lower()}_vocab.json"), "w"))
    print(ds, len(y), {s: int((split == s).sum()) for s in ["train", "val", "test"]}, "vocab", len(vocab), flush=True)
