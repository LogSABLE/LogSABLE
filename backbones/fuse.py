"""Plugs a backbone's predictions into LogSABLE using the pipeline's own fusion selection and saved module signals.
Usage: python backbones/fuse.py <model> <dataset> <signals.json>   e.g. logformer hdfs outputs/signals_n3.json"""
import json, os, sys
import numpy as np, yaml
sys.dont_write_bytecode = True
B = os.path.dirname(os.path.abspath(__file__)); R = os.path.dirname(B)
sys.path.insert(0, os.path.join(R, "src"))
import main as M
from logsable.train import anomaly_f1
name, ds, sig_path = sys.argv[1], sys.argv[2].lower(), sys.argv[3]
b = json.load(open(os.path.join(B, "out", f"{name}_{ds}.json"))); s = json.load(open(sig_path))
ev = yaml.safe_load(open(os.path.join(R, "configs", "default.yaml")))["eval"]
va = dict(zip(b["keys_va"], b["pred_va"])); te = dict(zip(b["keys_te"], b["pred_te"]))
keys_va, keys_te = s["val"]["keys"], s["test"]["keys"]
ypt_va = np.array([va[k] for k in keys_va]); ypt = np.array([te[k] for k in keys_te])
dyn = {k: (v, c) for sp in ("val", "test") for k, v, c in zip(s[sp]["keys"], s[sp]["llm_verdict"], s[sp]["llm_conf"]) if v is not None}
vec = lambda sp, f: None if s[sp][f] is None else np.array(s[sp][f])
ys_va, ys_te = np.array(s["val"]["label"]), np.array(s["test"]["label"])
nm, pred, _, _ = M._pick_final_hybrid_on_val(ys_va, ypt_va, ys_te, ypt, dyn, keys_va, keys_te,
    vec("val", "ngram"), vec("test", "ngram"), vec("val", "rules"), vec("test", "rules"), anomaly_f1, ev,
    default_dyn_conf=float(s["dynamic_min_conf"]), low_conf_anom_max=s["low_conf_anom_max"],
    fixed_dyn_min_conf=s["fixed_dyn_min_conf"], cluster_enabled=s["test"]["rules"] is not None)
bp = anomaly_f1(ys_te, ypt); fp = anomaly_f1(ys_te, pred)
print(f"{name} {ds}: backbone P/R/F1 {bp[0]:.3f} {bp[1]:.3f} {bp[2]:.3f} -> LogSABLE ({nm}) {fp[0]:.3f} {fp[1]:.3f} {fp[2]:.3f}", flush=True)
