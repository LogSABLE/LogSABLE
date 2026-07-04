from logsable.utils import seed_everything, print_banner
from logsable.config import load_config
from logsable.data import *
from logsable.train import * 
from logsable.model import *
from logsable.dynamic import *
from logsable.cluster import * 
from logsable.kb import *
from logsable.explain import superlog_demo
from logsable.rules import *
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import os, traceback, torch
from logsable.logrobust import *
from logsable.common_imports import *
from logsable.llm_openai import *
import time


def _append_run_timing_csv(
    cfg,
    train_s,
    baseline_infer_s,
    hybrid_setup_s,
    hybrid_online_s,
    hybrid_ms_per_session,
    hybrid_single_session_ms,
    dataset,
    model_name,
    n_test_sessions=0,
):
    """Append timing row: train, baseline eval, hybrid offline setup, hybrid online test infer."""
    import csv
    path = str(cfg.run.get("timing_csv", "outputs/run_training_inference.csv"))
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "timestamp",
        "dataset",
        "model",
        "n_test_sessions",
        "train_seconds",
        "baseline_inference_seconds",
        "hybrid_setup_seconds",
        "hybrid_online_infer_seconds",
        "hybrid_online_ms_per_session",
        "hybrid_single_session_ms",
    ]
    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(dataset),
        "model": str(model_name),
        "n_test_sessions": int(n_test_sessions),
        "train_seconds": round(float(train_s), 6),
        "baseline_inference_seconds": round(float(baseline_infer_s), 6),
        "hybrid_setup_seconds": round(float(hybrid_setup_s), 6),
        "hybrid_online_infer_seconds": round(float(hybrid_online_s), 6),
        "hybrid_online_ms_per_session": round(float(hybrid_ms_per_session), 6),
        "hybrid_single_session_ms": round(float(hybrid_single_session_ms), 6),
    }
    write_header = not os.path.exists(path) or os.path.getsize(path) == 0
    if not write_header:
        with open(path, "r", encoding="utf-8") as f:
            first_line = f.readline()
        if "hybrid_online_infer_seconds" not in first_line:
            path = path.replace(".csv", "_timing_v3.csv")
            write_header = not os.path.exists(path) or os.path.getsize(path) == 0
            print(f"[timing] legacy CSV schema; writing to {path}")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(
        f"[timing] train={row['train_seconds']:.3f}s "
        f"baseline_infer={row['baseline_inference_seconds']:.3f}s "
        f"hybrid_setup={row['hybrid_setup_seconds']:.3f}s "
        f"hybrid_online_infer={row['hybrid_online_infer_seconds']:.3f}s "
        f"({row['hybrid_online_ms_per_session']:.3f} ms/session, "
        f"single_session={row['hybrid_single_session_ms']:.3f} ms) → {path}"
    )


def _rules_pred_test_sessions(seqs_df, rules, keys_te, session_key_col, eid2rid, rid2eid, rid2tmpl=None):
    if not rules or len(keys_te) == 0:
        return {}
    key_set = {str(k) for k in keys_te}
    sess_col = session_key_col if session_key_col in seqs_df.columns else "BlockId"
    preds = {}
    for _, row in seqs_df.iterrows():
        sid = str(row[sess_col])
        if sid not in key_set:
            continue
        seq_rids = _to_row_ids(row["EventSeq"], eid2rid)
        fired = any(
            rule_fires_on_seq(seq_rids, r, rid2eid, rid2tmpl=rid2tmpl) for r in rules
        )
        preds[sid] = 1 if fired else 0
    return preds


def _time_hybrid_online_inference(
    seqs_for_rules,
    session_key_col,
    ses_keys,
    ypt,
    dyn_pred,
    rules,
    eid2rid,
    rid2eid,
    rid2tmpl,
    default_dyn_conf,
    kb_vec=None,
    low_conf_anom_max=None,
):
    t0 = time.perf_counter()
    dyn_te = _dyn_vec_from_pred(dyn_pred, ses_keys, default_dyn_conf, low_conf_anom_max)
    rule_preds = _rules_pred_test_sessions(
        seqs_for_rules, rules, ses_keys, session_key_col, eid2rid, rid2eid, rid2tmpl=rid2tmpl
    )
    rhat = np.array([rule_preds.get(str(k), 0) for k in ses_keys], dtype=int)
    parts = [np.asarray(ypt, dtype=int), dyn_te, rhat]
    if kb_vec is not None:
        parts.append(np.asarray(kb_vec, dtype=int))
    _ = np.maximum.reduce(parts)
    elapsed = time.perf_counter() - t0
    n = max(1, len(ses_keys))
    return elapsed, (elapsed / n) * 1000.0


def _benchmark_single_session_hybrid(
    seqs_for_rules,
    session_key_col,
    sample_key,
    ypt_by_key,
    dyn_pred,
    rules,
    eid2rid,
    rid2eid,
    rid2tmpl,
    default_dyn_conf,
    kb_by_key=None,
    low_conf_anom_max=None,
):
    """Time one session: rule checks + fuse base, LLM, rules, optional KB."""
    sess_col = session_key_col if session_key_col in seqs_for_rules.columns else "BlockId"
    row = seqs_for_rules[seqs_for_rules[sess_col].astype(str) == str(sample_key)]
    if len(row) == 0:
        return 0.0
    row = row.iloc[0]
    t0 = time.perf_counter()
    seq_rids = _to_row_ids(row["EventSeq"], eid2rid)
    rhat = 1 if any(rule_fires_on_seq(seq_rids, r, rid2eid, rid2tmpl=rid2tmpl) for r in (rules or [])) else 0
    val = dyn_pred.get(str(sample_key), (0, 0.0))
    if isinstance(val, (int, float)):
        risk, conf = int(val), 0.5
    else:
        risk, conf = int(val[0]), float(val[1]) if val[1] is not None else 0.5
    dyn = _dyn_session_fires(risk, conf, default_dyn_conf, low_conf_anom_max)
    base = int(ypt_by_key.get(str(sample_key), 0))
    kb = int((kb_by_key or {}).get(str(sample_key), 0))
    pred = max(base, dyn, rhat, kb)
    _ = pred
    return (time.perf_counter() - t0) * 1000.0


def _get_train_session_ids(goto_post_model, session_key_col, seqs_df,
                           TR_blk=None, keys=None, i_tr=None):
    """Return session ids used for training."""
    if goto_post_model:
        assert keys is not None and i_tr is not None
        return set(str(keys[i]) for i in i_tr)

    assert TR_blk is not None
    return set(map(str, TR_blk))



def pick_session_key_col(dataset: str, seqs_df: pd.DataFrame) -> str:
    ds = dataset.upper()
    if ds == "HDFS":
        cand = ["BlockId", "SessionId", "seq_key"]
    elif ds == "BGL":
        cand = ["Node", "SessionId", "BlockId", "seq_key"]
    elif ds == "THUNDERBIRD":
        cand = ["seq_key", "BlockId", "SessionId", "Node", "Location"]
    elif ds == "SPIRIT":
        cand = ["seq_key", "BlockId", "SessionId", "Node"]
    elif ds == "LIBERTY":
        cand = ["seq_key", "BlockId", "SessionId", "Node"]
    else:
        cand = ["SessionId", "BlockId", "Node", "seq_key"]

    for c in cand:
        if c in seqs_df.columns:
            return c
    return "seq_key" if "seq_key" in seqs_df.columns else cand[0]

def build_model(cfg, num_tokens: int):
    name = cfg.model["name"].lower()
    if name == "neurallog":
        return NeuralLogClassifier(
            num_tokens=num_tokens,
            d_model=cfg.model["neurallog"]["d_model"],
            dropout=cfg.model["neurallog"]["dropout"]
        )
    elif name == "deeplog":
        return DeepLogClassifier(
            num_tokens=num_tokens,
            embed_dim=cfg.model["deeplog"].get("embed_dim", 64),
            hidden_size=cfg.model["deeplog"].get("hidden_size", 64),
            num_layers=cfg.model["deeplog"].get("num_layers", 1),
            dropout=cfg.model["deeplog"].get("dropout", 0.1),
        )
    elif name == "logbert":
        return LogBERTClassifier(
            num_tokens=num_tokens,
            d_model=cfg.model["logbert"].get("d_model", 256),
            nhead=cfg.model["logbert"].get("nhead", 8),
            num_layers=cfg.model["logbert"].get("num_layers", 4),
            dim_ff=cfg.model["logbert"].get("dim_ff", 1024),
            dropout=cfg.model["logbert"].get("dropout", 0.1),
            max_len=cfg.model["logbert"].get("max_len", 512),
        )
    elif name == "loganomaly":
        return LogAnomalyClassifier(
            num_tokens=num_tokens,
            embed_dim=cfg.model.get("loganomaly", {}).get("embed_dim", 64),
            hidden_size=cfg.model.get("loganomaly", {}).get("hidden_size", 128),
            num_layers=cfg.model.get("loganomaly", {}).get("num_layers", 2),
            dropout=cfg.model.get("loganomaly", {}).get("dropout", 0.1),
        )
    raise ValueError(f"Unknown model: {name}")

def _coerce_list(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            v = json.loads(s)
            if isinstance(v, list):
                return v
        except Exception:
            pass
        s = s.strip("[]")
        parts = [p.strip() for p in s.split(",") if p.strip() != ""]
        return parts
    return list(x) if x is not None else []


def _to_row_ids(evseq, eid2rid):
    """Parse EventSeq to integer row ids (accepts ints or EventId strings via eid2rid)."""
    toks = _coerce_list(evseq)
    out = []
    for t in toks:
        try:
            out.append(int(t))
            continue
        except (TypeError, ValueError):
            pass
        ts = str(t)
        if ts in eid2rid:
            out.append(int(eid2rid[ts]))
    return out


def _fuse_signals(signals_list, strategy="or", vote_min=None):
    """Fuse session-level 0/1 signals. strategy: 'or' | 'and' | 'vote' (vote_min required)."""
    stack = np.stack([np.asarray(s, dtype=int) for s in signals_list], axis=1)
    n_src = stack.shape[1]
    if strategy == "or" or (strategy == "vote" and vote_min == 1):
        return (stack.sum(axis=1) >= 1).astype(int)
    if strategy == "and":
        return (stack.sum(axis=1) == n_src).astype(int)
    if strategy == "vote" and vote_min is not None:
        return (stack.sum(axis=1) >= vote_min).astype(int)
    return (stack.sum(axis=1) >= 1).astype(int)


def _leg_val_fire_rate_on_normals(sig, ys_va):
    """Fraction of normal validation sessions where this leg fires."""
    sig = np.asarray(sig, dtype=int)
    ys_va = np.asarray(ys_va, dtype=int)
    norm = ys_va == 0
    if norm.sum() == 0:
        return float(sig.mean())
    return float(sig[norm].mean())


def _filter_fusion_signals(signals_va, signals_te, ys_va, max_leg_fire=0.35):
    if not signals_va:
        return signals_va, signals_te
    out_va, out_te = [signals_va[0]], [signals_te[0]]
    for s_va, s_te in zip(signals_va[1:], signals_te[1:]):
        if _leg_val_fire_rate_on_normals(s_va, ys_va) <= max_leg_fire:
            out_va.append(s_va)
            out_te.append(s_te)
    return out_va, out_te


def _tune_fusion_on_val(ys_va, signals_va, signals_te, anomaly_f1_fn):
    """Resolve fusion strategy on the validation split."""
    best_f1 = -1.0
    best_name = "or"
    best_te_pred = _fuse_signals(signals_te, "or")
    n_src = len(signals_va)
    for name, strategy, vote_min in [
        ("or", "or", None),
        ("and", "and", None),
    ] + [(f"vote{k}", "vote", k) for k in range(2, n_src + 1)]:
        pred_va = _fuse_signals(signals_va, strategy, vote_min)
        _, _, f1, _ = anomaly_f1_fn(ys_va, pred_va)
        if f1 > best_f1:
            best_f1, best_name = float(f1), name
            best_te_pred = _fuse_signals(signals_te, strategy, vote_min)
    return best_name, best_te_pred, best_f1


def _gate_llm_with_symbolic(dyn, rhat, kb=None):
    dyn = np.asarray(dyn, dtype=int)
    sym = np.asarray(rhat, dtype=int) > 0
    if kb is not None:
        sym = sym | (np.asarray(kb, dtype=int) > 0)
    return np.asarray((dyn > 0) & sym, dtype=int)


def _full_hybrid_llm_gate_enabled(eval_cfg, include_llm, include_ngram, include_cluster, has_kb):
    return bool(
        include_llm
        and include_ngram
        and include_cluster
        and has_kb
        and eval_cfg.get("hybrid_llm_requires_symbolic_for_full", True)
    )


def _fusion_strategy_tie_key(name):
    name = str(name)
    if name.startswith("vote") and name[4:].isdigit():
        return (0, int(name[4:]))
    order = {"or": 1, "and": 2, "base+ngram_or": 3, "base+kb_or": 4, "base+ngram+kb_or": 5}
    return (1, order.get(name, 9))


def _tune_hybrid_on_val(ys_va, ypt_va, ys_te, ypt, dyn_pred_with_conf, keys_va, keys_te,
                        rhat_va, rhat_te_s, anomaly_f1_fn,
                        kb_vec_va=None, kb_vec=None, eval_cfg=None, low_conf_anom_max=None,
                        fixed_dyn_min_conf=None,
                        include_llm=True, include_ngram=True, include_cluster=True):
    """Resolve LLM confidence and fusion strategy on the validation split."""
    eval_cfg = eval_cfg or {}
    max_leg_fire = float(eval_cfg.get("fusion_max_leg_fire_on_normals", 0.35))
    fallback_base = bool(eval_cfg.get("hybrid_fallback_to_base", True))
    _, _, base_va_f1, _ = anomaly_f1_fn(ys_va, ypt_va)
    if fixed_dyn_min_conf is not None:
        conf_candidates = [float(fixed_dyn_min_conf)]
    else:
        conf_candidates = [0.5, 0.6, 0.7, 0.8, 0.9]
    best_val_f1 = -1.0
    best_min_conf = 0.5
    best_name = "or"
    best_te_pred = np.copy(ypt)
    keys_va = list(keys_va)
    keys_te = list(keys_te)
    n_va, n_te = len(keys_va), len(keys_te)
    has_kb = (
        include_cluster
        and kb_vec_va is not None
        and kb_vec is not None
        and len(kb_vec_va) == n_va
        and len(kb_vec) == n_te
    )
    gate_llm = _full_hybrid_llm_gate_enabled(
        eval_cfg, include_llm, include_ngram, include_cluster, has_kb
    )
    rhat_va_a = np.asarray(rhat_va, dtype=int) if include_ngram else np.zeros(n_va, dtype=int)
    rhat_te_a = np.asarray(rhat_te_s, dtype=int) if include_ngram else np.zeros(n_te, dtype=int)
    kb_va_a = np.asarray(kb_vec_va, dtype=int) if has_kb else None
    kb_te_a = np.asarray(kb_vec, dtype=int) if has_kb else None
    for min_conf in conf_candidates:
        if include_llm:
            dyn_va = _dyn_vec_from_pred(
                dyn_pred_with_conf, keys_va, min_conf, low_conf_anom_max=low_conf_anom_max
            )
            dyn_te = _dyn_vec_from_pred(
                dyn_pred_with_conf, keys_te, min_conf, low_conf_anom_max=low_conf_anom_max
            )
            if gate_llm:
                dyn_va = _gate_llm_with_symbolic(dyn_va, rhat_va_a, kb_va_a)
                dyn_te = _gate_llm_with_symbolic(dyn_te, rhat_te_a, kb_te_a)
        else:
            dyn_va = np.zeros(n_va, dtype=int)
            dyn_te = np.zeros(n_te, dtype=int)
        signals_va = [ypt_va, dyn_va, rhat_va_a]
        signals_te = [ypt, dyn_te, rhat_te_a]
        if has_kb:
            signals_va.append(kb_va_a)
            signals_te.append(kb_te_a)
        signals_va, signals_te = _filter_fusion_signals(
            signals_va, signals_te, ys_va, max_leg_fire=max_leg_fire
        )
        name, te_pred, val_f1 = _tune_fusion_on_val(
            ys_va, signals_va, signals_te, anomaly_f1_fn
        )
        cand_f1, cand_name, cand_te = float(val_f1), name, te_pred
        if include_ngram and int(rhat_va_a.sum()) > 0:
            bn_va = np.maximum(np.asarray(ypt_va, dtype=int), rhat_va_a)
            _, _, f1_bn, _ = anomaly_f1_fn(ys_va, bn_va)
            if f1_bn > cand_f1:
                cand_f1, cand_name, cand_te = float(f1_bn), "base+ngram_or", np.maximum(
                    np.asarray(ypt, dtype=int), rhat_te_a
                )
        better = cand_f1 > best_val_f1 + 1e-9
        tie_break = (
            not better
            and abs(cand_f1 - best_val_f1) <= 1e-9
            and (
                min_conf > best_min_conf
                or (
                    min_conf == best_min_conf
                    and _fusion_strategy_tie_key(cand_name) < _fusion_strategy_tie_key(best_name)
                )
            )
        )
        if better or tie_break:
            best_val_f1, best_min_conf, best_name, best_te_pred = cand_f1, min_conf, cand_name, cand_te
    if fallback_base and best_val_f1 < base_va_f1 - 1e-9:
        return 0.5, "base_only", np.copy(ypt), float(base_va_f1)
    return best_min_conf, best_name, best_te_pred, best_val_f1


def _parse_fusion_strategy(name):
    """Map fusion strategy name to fusion mode for component evaluation."""
    name = str(name)
    if name == "base+ngram_or":
        return "base+ngram_or"
    if name == "base+kb_or":
        return "base+kb_or"
    if name == "base+ngram+kb_or":
        return "base+ngram+kb_or"
    if name in ("base_only", "ngram_only"):
        return name
    if name.startswith("vote") and name[4:].isdigit():
        return ("vote", int(name[4:]))
    if name in ("or", "and", "4-way_or"):
        return "or" if name == "4-way_or" else name
    return "or"


def _fuse_hybrid_signals(signals, best_name, ypt_ref, rhat_ref, has_kb, kb_ref):
    """Apply parsed fusion strategy to a list of session-level leg signals."""
    parsed = _parse_fusion_strategy(best_name)
    if parsed == "base+ngram_or":
        return np.maximum(ypt_ref, rhat_ref)
    if parsed == "base+ngram+kb_or":
        kb_a = np.asarray(kb_ref, dtype=int) if has_kb else np.zeros_like(ypt_ref, dtype=int)
        return np.maximum(np.maximum(ypt_ref, rhat_ref), kb_a)
    if parsed == "base+kb_or":
        kb_a = np.asarray(kb_ref, dtype=int) if has_kb else np.zeros_like(ypt_ref, dtype=int)
        return np.maximum(ypt_ref, kb_a)
    if parsed == "base_only":
        return np.copy(ypt_ref)
    if parsed == "ngram_only":
        return np.copy(rhat_ref)
    if isinstance(parsed, tuple) and parsed[0] == "vote":
        k = max(1, min(int(parsed[1]), len(signals)))
        return _fuse_signals(signals, "vote", k)
    if parsed == "and":
        return _fuse_signals(signals, "and")
    return _fuse_signals(signals, "or")


def _build_hybrid_signals(
    ypt_split,
    ypt_va,
    dyn_pred,
    keys_split,
    keys_va,
    rhat_split,
    rhat_va,
    kb_split,
    kb_va,
    best_conf,
    eval_cfg,
    low_conf_anom_max,
    include_llm=True,
    include_ngram=True,
    include_cluster=True,
    ys_va=None,
    apply_leg_filter=True,
):
    """Build filtered fusion leg signals for one split (test or val)."""
    eval_cfg = eval_cfg or {}
    keys_split = list(keys_split)
    keys_va = list(keys_va)
    n_split, n_va = len(keys_split), len(keys_va)
    ypt_split = np.asarray(ypt_split, dtype=int)
    ypt_va = np.asarray(ypt_va, dtype=int)
    rhat_split_a = np.asarray(rhat_split, dtype=int) if include_ngram else np.zeros(n_split, dtype=int)
    rhat_va_a = np.asarray(rhat_va, dtype=int) if include_ngram else np.zeros(n_va, dtype=int)
    has_kb = (
        include_cluster
        and kb_split is not None
        and kb_va is not None
        and len(kb_split) == n_split
        and len(kb_va) == n_va
    )
    kb_split_a = np.asarray(kb_split, dtype=int) if has_kb else None
    kb_va_a = np.asarray(kb_va, dtype=int) if has_kb else None
    gate_llm = _full_hybrid_llm_gate_enabled(
        eval_cfg, include_llm, include_ngram, include_cluster, has_kb
    )
    if include_llm:
        dyn_split = _dyn_vec_from_pred(dyn_pred, keys_split, best_conf, low_conf_anom_max)
        dyn_va = _dyn_vec_from_pred(dyn_pred, keys_va, best_conf, low_conf_anom_max)
        if gate_llm:
            dyn_split = _gate_llm_with_symbolic(dyn_split, rhat_split_a, kb_split_a)
            dyn_va = _gate_llm_with_symbolic(dyn_va, rhat_va_a, kb_va_a)
    else:
        dyn_split = np.zeros(n_split, dtype=int)
        dyn_va = np.zeros(n_va, dtype=int)
    signals_split = [ypt_split, dyn_split, rhat_split_a]
    signals_va = [ypt_va, dyn_va, rhat_va_a]
    if has_kb:
        signals_split.append(kb_split_a)
        signals_va.append(kb_va_a)
    if apply_leg_filter and ys_va is not None and n_va > 0:
        max_leg_fire = float(eval_cfg.get("fusion_max_leg_fire_on_normals", 0.35))
        signals_va, signals_split = _filter_fusion_signals(
            signals_va, signals_split, ys_va, max_leg_fire=max_leg_fire
        )
    return signals_split, signals_va, rhat_split_a, rhat_va_a, kb_split_a, has_kb


def _ablation_vote_k(full_vote_k, n_full, n_signals):
    """Keep full-hybrid vote-k when legs are removed."""
    full_k = int(full_vote_k)
    n_removed = max(0, int(n_full) - int(n_signals))
    if n_removed > 0 and full_k >= int(n_full) - 1:
        return max(1, min(int(n_signals), full_k - n_removed))
    return max(1, min(full_k, int(n_signals)))


def _apply_fixed_hybrid_fusion(
    ypt,
    ypt_va,
    dyn_pred,
    keys_te,
    keys_va,
    rhat_te,
    rhat_va,
    kb_te,
    kb_va,
    best_conf,
    best_name,
    eval_cfg,
    low_conf_anom_max,
    include_llm=True,
    include_ngram=True,
    include_cluster=True,
    ys_va=None,
    apply_leg_filter=True,
    fuse_split="te",
    full_vote_k=None,
):
    """Fuse hybrid legs using fixed confidence and strategy from the full hybrid."""
    signals_te, signals_va, rhat_te_a, rhat_va_a, kb_te_a, has_kb = _build_hybrid_signals(
        ypt, ypt_va, dyn_pred, keys_te, keys_va, rhat_te, rhat_va, kb_te, kb_va,
        best_conf, eval_cfg, low_conf_anom_max,
        include_llm=include_llm, include_ngram=include_ngram, include_cluster=include_cluster,
        ys_va=ys_va, apply_leg_filter=apply_leg_filter,
    )
    ypt_ref = np.asarray(ypt_va if fuse_split == "va" else ypt, dtype=int)
    rhat_ref = rhat_va_a if fuse_split == "va" else rhat_te_a
    kb_ref = kb_va if fuse_split == "va" else kb_te_a
    signals = signals_va if fuse_split == "va" else signals_te
    parsed = _parse_fusion_strategy(best_name)
    if isinstance(parsed, tuple) and parsed[0] == "vote" and full_vote_k is not None:
        n_full = int(eval_cfg.get("hybrid_full_n_fusion_legs", len(signals_te)))
        k = _ablation_vote_k(full_vote_k, n_full, len(signals))
        return _fuse_signals(signals, "vote", k)
    return _fuse_hybrid_signals(signals, best_name, ypt_ref, rhat_ref, has_kb, kb_ref)


def _apply_base_gate_if_needed(cfg, p_te_s, ypt, pred):
    if cfg.eval.get("base_gate_enabled", False) and cfg.model["name"].lower() == "logrobust":
        lo = float(cfg.eval.get("base_gate_lo", 0.05))
        hi = float(cfg.eval.get("base_gate_hi", 0.95))
        return gate_fusion_with_base_proba(p_te_s, ypt, pred, lo, hi)
    return pred


def _print_final_hybrid_ablations(
    ys_va,
    ypt_va,
    ys_te,
    ypt,
    dyn_pred,
    keys_va,
    keys_te,
    rhat_va,
    rhat_te_s,
    kb_vec_va,
    kb_vec,
    cfg,
    eval_cfg,
    low_conf_anom_max,
    best_conf,
    best_name,
    full_hybrid_f1,
    p_te_s=None,
    cluster_enabled=True,
):
    if len(keys_va) == 0:
        print("[HYBRID] component eval skipped (no VAL split)")
        return

    eval_cfg_abl = dict(eval_cfg or {})
    _, signals_va_full, _, _, _, _ = _build_hybrid_signals(
        ypt, ypt_va, dyn_pred, keys_te, keys_va, rhat_te_s, rhat_va, kb_vec, kb_vec_va,
        best_conf, eval_cfg_abl, low_conf_anom_max,
        include_llm=True, include_ngram=True, include_cluster=cluster_enabled,
        ys_va=ys_va,
    )
    eval_cfg_abl["hybrid_full_n_fusion_legs"] = len(signals_va_full)
    parsed = _parse_fusion_strategy(best_name)
    full_vote_k = int(parsed[1]) if isinstance(parsed, tuple) and parsed[0] == "vote" else None

    print(
        f"[HYBRID] component eval (conf={best_conf:.2f}, strategy={best_name}; "
        f"full TEST F1={full_hybrid_f1:.3f}):"
    )

    def _run_one(label, include_llm, include_ngram, include_cluster):
        pred = _apply_fixed_hybrid_fusion(
            ypt, ypt_va, dyn_pred, keys_te, keys_va,
            rhat_te_s, rhat_va, kb_vec, kb_vec_va,
            best_conf, best_name, eval_cfg_abl, low_conf_anom_max,
            include_llm=include_llm,
            include_ngram=include_ngram,
            include_cluster=include_cluster,
            ys_va=ys_va,
            full_vote_k=full_vote_k,
        )
        if p_te_s is not None:
            pred = _apply_base_gate_if_needed(cfg, p_te_s, ypt, pred)
        p, r, f1, cm = anomaly_f1(ys_te, pred)
        delta = float(f1) - float(full_hybrid_f1)
        sign = "+" if delta >= 0 else ""
        print(
            f"[HYBRID] {label:22s} TEST P={p:.3f} R={r:.3f} F1={f1:.3f} "
            f"(Δ vs full {sign}{delta:.3f}) | cm={cm.tolist()}"
        )

    _run_one("hybrid_no_llm", include_llm=False, include_ngram=True, include_cluster=True)
    _run_one("hybrid_no_ngram", include_llm=True, include_ngram=False, include_cluster=True)
    if cluster_enabled:
        _run_one("hybrid_no_cluster", include_llm=True, include_ngram=True, include_cluster=False)
    else:
        print("[HYBRID] hybrid_no_cluster skipped (cluster disabled; same as full hybrid)")


def _pick_final_hybrid_on_val(
    ys_va,
    ypt_va,
    ys_te,
    ypt,
    dyn_pred,
    keys_va,
    keys_te,
    rhat_va,
    rhat_te_s,
    kb_vec_va,
    kb_vec,
    anomaly_f1_fn,
    eval_cfg,
    default_dyn_conf=0.5,
    low_conf_anom_max=None,
    fixed_dyn_min_conf=None,
    cluster_enabled=True,
):
    eval_cfg = eval_cfg or {}
    ys_va = np.asarray(ys_va, dtype=int)
    ypt_va = np.asarray(ypt_va, dtype=int)
    ypt = np.asarray(ypt, dtype=int)
    rhat_va_a = np.asarray(rhat_va, dtype=int)
    rhat_te_a = np.asarray(rhat_te_s, dtype=int)
    has_kb = (
        cluster_enabled
        and kb_vec_va is not None
        and kb_vec is not None
        and len(kb_vec_va) == len(keys_va)
    )

    _, _, base_va_f1, _ = anomaly_f1_fn(ys_va, ypt_va)
    best_val_f1 = base_va_f1
    best_name = "base_only"
    best_pred = np.copy(ypt)
    best_conf = float(default_dyn_conf)

    def _try(name, pred_va, pred_te):
        nonlocal best_val_f1, best_name, best_pred
        _, _, f1, _ = anomaly_f1_fn(ys_va, np.asarray(pred_va, dtype=int))
        if f1 > best_val_f1 + 1e-9:
            best_val_f1 = float(f1)
            best_name = name
            best_pred = np.asarray(pred_te, dtype=int).copy()

    _try("base_only", ypt_va, ypt)
    _try("base+ngram_or", np.maximum(ypt_va, rhat_va_a), np.maximum(ypt, rhat_te_a))
    if has_kb:
        kb_va_a = np.asarray(kb_vec_va, dtype=int)
        kb_te_a = np.asarray(kb_vec, dtype=int)
        _try("base+kb_or", np.maximum(ypt_va, kb_va_a), np.maximum(ypt, kb_te_a))
        _try(
            "base+ngram+kb_or",
            np.maximum(np.maximum(ypt_va, rhat_va_a), kb_va_a),
            np.maximum(np.maximum(ypt, rhat_te_a), kb_te_a),
        )
    res_va = np.where(ypt_va == 1, 1, rhat_va_a)
    res_te = np.where(ypt == 1, 1, rhat_te_a)
    _try("base+residual_ngram", res_va, res_te)

    if eval_cfg.get("tune_fusion", True) and len(keys_va) > 0:
        conf, name, te_pred, val_f1 = _tune_hybrid_on_val(
            ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, keys_te,
            rhat_va, rhat_te_s, anomaly_f1_fn,
            kb_vec_va=kb_vec_va, kb_vec=kb_vec, eval_cfg=eval_cfg,
            low_conf_anom_max=low_conf_anom_max,
            fixed_dyn_min_conf=fixed_dyn_min_conf,
        )
        if float(val_f1) > best_val_f1 + 1e-9:
            best_val_f1 = float(val_f1)
            best_name = name
            best_conf = float(conf)

    eval_cfg_fusion = dict(eval_cfg or {})
    _, signals_va_full, _, _, _, _ = _build_hybrid_signals(
        ypt, ypt_va, dyn_pred, keys_te, keys_va, rhat_te_s, rhat_va, kb_vec, kb_vec_va,
        best_conf, eval_cfg_fusion, low_conf_anom_max,
        include_llm=True, include_ngram=True, include_cluster=cluster_enabled and has_kb,
        ys_va=ys_va,
    )
    eval_cfg_fusion["hybrid_full_n_fusion_legs"] = len(signals_va_full)
    parsed = _parse_fusion_strategy(best_name)
    full_vote_k = int(parsed[1]) if isinstance(parsed, tuple) and parsed[0] == "vote" else None

    best_pred = _apply_fixed_hybrid_fusion(
        ypt, ypt_va, dyn_pred, keys_te, keys_va,
        rhat_te_s, rhat_va, kb_vec, kb_vec_va,
        best_conf, best_name, eval_cfg_fusion, low_conf_anom_max,
        include_llm=True, include_ngram=True, include_cluster=cluster_enabled and has_kb,
        ys_va=ys_va,
        full_vote_k=full_vote_k,
    )
    pred_va = _apply_fixed_hybrid_fusion(
        ypt, ypt_va, dyn_pred, keys_te, keys_va,
        rhat_te_s, rhat_va, kb_vec, kb_vec_va,
        best_conf, best_name, eval_cfg_fusion, low_conf_anom_max,
        include_llm=True, include_ngram=True, include_cluster=cluster_enabled and has_kb,
        ys_va=ys_va,
        fuse_split="va",
        full_vote_k=full_vote_k,
    )
    _, _, best_val_f1, _ = anomaly_f1_fn(ys_va, pred_va)
    return best_name, best_pred, best_conf, best_val_f1


def _dyn_session_fires(risk, conf, min_conf=0.0, low_conf_anom_max=None):
    """Session-level LLM vote from risk label and confidence."""
    if int(risk) == 1 and float(conf) >= float(min_conf):
        return 1
    if low_conf_anom_max is not None and int(risk) == 0 and float(conf) < float(low_conf_anom_max):
        return 1
    return 0


def _dyn_vec_from_pred(dyn_pred_with_conf, keys, min_conf=0.0, low_conf_anom_max=None):
    """Build binary dynamic vector from LLM session judgments.
    dyn_pred_with_conf: dict seq_key -> (is_high_risk, confidence). Missing keys -> 0."""
    out = []
    for k in keys:
        val = dyn_pred_with_conf.get(k, (0, 0.0))
        if isinstance(val, (int, float)):
            risk, conf = int(val), 0.5
        else:
            risk, conf = int(val[0]), float(val[1]) if val[1] is not None else 0.5
        out.append(_dyn_session_fires(risk, conf, min_conf, low_conf_anom_max))
    return np.array(out, dtype=int)


def _tune_hdfs_dyn_rules_on_val(dyn_pred_with_conf, keys_va, ys_va, eval_cfg=None):
    """Resolve LLM confidence thresholds on the validation split."""
    eval_cfg = eval_cfg or {}
    hi_cands = eval_cfg.get("hdfs_dynamic_hi_min_conf_candidates", [0.0, 0.5, 0.7])
    low_cands = eval_cfg.get(
        "hdfs_dynamic_low_conf_candidates",
        [0.74, 0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83],
    )
    best_f1, best_hi, best_low = -1.0, float(eval_cfg.get("hdfs_dynamic_min_conf", 0.0)), None
    ys_va = np.asarray(ys_va, dtype=int)
    for hi_min in hi_cands:
        for low_max in low_cands:
            vec = _dyn_vec_from_pred(
                dyn_pred_with_conf, keys_va, float(hi_min), low_conf_anom_max=float(low_max)
            )
            _, _, f1, _ = anomaly_f1(ys_va, vec)
            if float(f1) > best_f1:
                best_f1 = float(f1)
                best_hi = float(hi_min)
                best_low = float(low_max)
    return best_low, best_hi, best_f1


def gate_fusion_with_base_proba(base_proba, base_pred, fused_pred, lo=0.05, hi=0.95):
    out = np.asarray(fused_pred, dtype=int).copy()
    confident = (base_proba <= lo) | (base_proba >= hi)
    out[confident] = np.asarray(base_pred, dtype=int)[confident]
    return out


def make_tok2text(templates_df):
    eid2tmpl_str = {}
    if {"EventId", "EventTemplate"}.issubset(templates_df.columns):
        eid2tmpl_str = {str(e): str(t)
                        for e, t in zip(templates_df["EventId"], templates_df["EventTemplate"])}

    rid2tmpl = {int(i): str(t)
                for i, t in enumerate(templates_df["EventTemplate"].tolist())}

    def tok2text(tok):
        s = str(tok)
        if s in eid2tmpl_str:
            return eid2tmpl_str[s]
        try:
            x = int(s)
            if x in rid2tmpl:
                return rid2tmpl[x]
        except Exception:
            pass
        return f"[T{tok}]"

    return tok2text

def main():
    import numpy as np
    cfg = load_config("configs/default.yaml")
    print_banner(cfg)
    _timing_ok = False
    train_seconds = 0.0
    baseline_inference_seconds = 0.0
    hybrid_setup_seconds = 0.0
    hybrid_online_seconds = 0.0
    hybrid_ms_per_session = 0.0
    hybrid_single_session_ms = 0.0
    _hybrid_online_recorded = False
    _t_hybrid_setup_start = None
    ses_keys = []
    keys_te = []

    print(f"[main] seeding… {cfg.run['seed']}")
    seed_everything(cfg.run["seed"])

    dataset = str(cfg.data.get("dataset", "HDFS"))
    run_tag = f"{dataset}_win{cfg.data['window']}_str{cfg.data['stride']}_seed{cfg.run['seed']}"

    print(f"[main] ensuring data in: {cfg.data['data_dir']}")
    raw_df, templates_df, seqs_df = load_sequences(
        cfg.data["data_dir"],
        dataset=dataset,
        data_cfg=cfg.data,
    )
    RID_E978 = None
    if dataset.strip().upper() == "THUNDERBIRD":
        eid_list = templates_df["EventId"].astype(str).tolist()
        eid2rid_tmp = {eid: i for i, eid in enumerate(eid_list)}
        RID_E978 = eid2rid_tmp.get("E978", 977)

        def _mask_seq(seq):
            if seq is None:
                return []
            return [rid for rid in (seq if isinstance(seq, (list, tuple, np.ndarray)) else list(seq)) if rid != RID_E978]

        seqs_df = seqs_df.copy()
        seqs_df["EventSeq_masked"] = seqs_df["EventSeq"].apply(_mask_seq)
        print(f"[main] feature mask applied (row_id={RID_E978})")
    else:
        seqs_df["EventSeq_masked"] = seqs_df["EventSeq"]
    session_key_col = pick_session_key_col(dataset, seqs_df)
    print(f"[main] session_key_col={session_key_col}")

    model_name_early = str(cfg.model.get("name", "")).lower()
    seqs_df_windows = seqs_df
    loganomaly_num_tokens = None
    if (
        dataset.strip().upper() == "LIBERTY"
        and model_name_early == "loganomaly"
        and cfg.model.get("loganomaly", {}).get("liberty_use_label_token_vocab", True)
    ):
        from logsable.data import load_liberty_sequences
        _, templates_la, seqs_la = load_liberty_sequences(
            cfg.data["data_dir"],
            bucket_sec=int(cfg.data.get("liberty_bucket_sec", 600)),
            filename=str(cfg.data.get("liberty_filename", "liberty_150k.csv")),
            event_id_source="label_token",
        )
        seqs_df_windows = seqs_la
        loganomaly_num_tokens = len(templates_la)
        print(
            f"[main] compact backbone vocab (num_tokens={loganomaly_num_tokens}); "
            f"full templates ({len(templates_df)} entries) used for rules and hybrid."
        )

    goto_post_model = False

    if dataset.strip().upper() == "THUNDERBIRD" and "EventSeq_masked" in seqs_df.columns:
        X_win, y_cls, blk_ids = make_windows_from_sequences(
            seqs_df, window_size=cfg.data["window"], stride=cfg.data["stride"],
            filter_len_col="EventSeq", content_col="EventSeq_masked",
            pad_short_sequences=True
        )
    else:
        X_win, y_cls, blk_ids = make_windows_from_sequences(
            seqs_df_windows, window_size=cfg.data["window"], stride=cfg.data["stride"]
        )
    if cfg.data.get("group_split", True):
        _train_size = cfg.data.get("train_size")
        splits = make_group_splits(
            X_win, y_cls, blk_ids,
            cfg.data["val_size"], cfg.data["test_size"], cfg.run["seed"],
            train_size=_train_size,
        )
        (TR_X, TR_y, TR_blk), (VA_X, VA_y, VA_blk), (TE_X, TE_y, TE_blk) = splits
        _eff_train = (
            float(_train_size)
            if _train_size is not None
            else 1.0 - float(cfg.data["val_size"]) - float(cfg.data["test_size"])
        )
        _split_msg = (
            f"[split] session fractions: train={_eff_train:.2f} "
            f"val={float(cfg.data['val_size']):.2f} test={float(cfg.data['test_size']):.2f}"
        )
        if _train_size is not None:
            _used = _eff_train + float(cfg.data["val_size"]) + float(cfg.data["test_size"])
            _split_msg += f" unused={max(0.0, 1.0 - _used):.2f} (explicit train_size)"
        else:
            _split_msg += " (train = remainder)"
        print(_split_msg)
    else:
        import numpy as np
        try:
            from sklearn.model_selection import train_test_split
        except ImportError:
            raise RuntimeError("Please `pip install scikit-learn` for non-group splits.")

        X_win = np.asarray(X_win)
        y_cls = np.asarray(y_cls)
        blk_ids_arr = np.asarray(blk_ids)

        test_size = float(cfg.data["test_size"])
        val_size = float(cfg.data["val_size"])
        temp_size = val_size + test_size
        if temp_size <= 0 or temp_size >= 1.0:
            raise ValueError("val_size + test_size must be in (0, 1).")

        unique_y = np.unique(y_cls)
        stratify_y = y_cls if unique_y.size >= 2 else None

        idx = np.arange(len(y_cls))
        i_tr, i_tmp = train_test_split(
            idx, test_size=temp_size, random_state=cfg.run["seed"], stratify=stratify_y
        )

        val_frac_of_tmp = val_size / temp_size
        stratify_tmp = y_cls[i_tmp] if np.unique(y_cls[i_tmp]).size >= 2 else None
        i_va, i_te = train_test_split(
            i_tmp, test_size=(1.0 - val_frac_of_tmp), random_state=cfg.run["seed"], stratify=stratify_tmp
        )

        TR_X, TR_y, TR_blk = X_win[i_tr], y_cls[i_tr], blk_ids_arr[i_tr]
        VA_X, VA_y, VA_blk = X_win[i_va], y_cls[i_va], blk_ids_arr[i_va]
        TE_X, TE_y, TE_blk = X_win[i_te], y_cls[i_te], blk_ids_arr[i_te]
    
    train_session_ids = _get_train_session_ids(
        goto_post_model=False,
        session_key_col=session_key_col,
        seqs_df=seqs_df,
        TR_blk=TR_blk
    )

    llm_map = None
    if cfg.distill.get("enabled", False):
        from logsable.dynamic import load_llm_session_labels
        m = load_llm_session_labels(
            db_path="logdb.sqlite",
            dataset=dataset,
            run_tag=run_tag,
            min_conf=cfg.distill.get("min_conf", 0.0),
        )
        llm_map = {k: y for k, (y, conf) in m.items()}
        print(f"[distill] loaded LLM labels for train (dataset={dataset}, run_tag={run_tag!r}): {len(llm_map)} sessions")

    if llm_map is not None:
        tr_blks = set(map(str, TR_blk))
        va_blks = set(map(str, VA_blk))
        te_blks = set(map(str, TE_blk))

        cov_tr = sum(1 for b in tr_blks if b in llm_map) / max(1, len(tr_blks))
        cov_va = sum(1 for b in va_blks if b in llm_map) / max(1, len(va_blks))
        cov_te = sum(1 for b in te_blks if b in llm_map) / max(1, len(te_blks))
        n_win = len(TR_blk)
        n_win_labeled = sum(1 for b in map(str, TR_blk) if b in llm_map)
        print(f"[distill] train windows with teacher label: {n_win_labeled}/{n_win} ({n_win_labeled/max(1,n_win):.3f})")
        print(f"[distill] coverage: train={cov_tr:.3f} val={cov_va:.3f} test={cov_te:.3f}")


    print(f"[main] loaded: templates_df={len(templates_df)} rows, seqs_df={len(seqs_df)} rows")
    print(f"[main] windowing (WINDOW={cfg.data['window']}, STRIDE={cfg.data['stride']})…")
    print(f"[main] windowed: X={len(X_win)}, y={len(y_cls)}")
    print("[main] making splits/loaders…")
    print(f"[main] splits: train={len(TR_y)}, val={len(VA_y)}, test={len(TE_y)}")

    _t_train_start = time.perf_counter()
    if cfg.model["name"].lower() == "logrobust":
        print("[main] fitting window-level classifier and aggregating to sessions")
        max_len = cfg.logrobust.get("max_len", None)
        def _window_to_text(w):
            toks = [str(int(r)) for r in w]
            if max_len and max_len > 0:
                toks = toks[:max_len]
            return " ".join(toks)
        window_texts_tr = [_window_to_text(w) for w in TR_X]
        window_texts_va = [_window_to_text(w) for w in VA_X]
        window_texts_te = [_window_to_text(w) for w in TE_X]
        lr_bundle = fit_logrobust(window_texts_tr, TR_y, cfg.logrobust)
        y_va = np.asarray(VA_y)
        y_te = np.asarray(TE_y)
        b_va = VA_blk
        b_te = TE_blk
    else:
        device = cfg.device
        tr_loader = DataLoader(WindowDataset(TR_X, TR_y, TR_blk, llm_map=llm_map),
                            batch_size=cfg.train["batch_size"], shuffle=True,
                            num_workers=cfg.train["num_workers"], pin_memory=cfg.train["pin_memory"])
        va_loader = DataLoader(WindowDataset(VA_X, VA_y, VA_blk), batch_size=cfg.train["batch_size"])
        te_loader = DataLoader(WindowDataset(TE_X, TE_y, TE_blk), batch_size=cfg.train["batch_size"])
        print(f"[main] building model on {cfg.device}…")
        print(f"[main] training for {cfg.train['epochs']} epoch(s)…")
        if loganomaly_num_tokens is not None:
            num_tokens = int(loganomaly_num_tokens)
        else:
            num_tokens = int(templates_df[cfg.model["vocab_from"]].nunique())
        model = build_model(cfg, num_tokens).to(device)
        print(f"[main] num_tokens={num_tokens}")

        cw = class_weights_from(TR_y)
        if cw is None or (hasattr(cw, "numel") and cw.numel() < 2):
            print("[warn] class_weights_from returned None/1-class; using uniform loss (no weights).")
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.CrossEntropyLoss(weight=cw.to(device))

        lr = cfg.train["lr"]
        if cfg.model.get("name", "").lower() == "logbert" and cfg.model.get("logbert") and "lr" in cfg.model["logbert"]:
            lr = float(cfg.model["logbert"]["lr"])
            print(f"[main] lr={lr}")
        optim = torch.optim.AdamW(model.parameters(),
                                lr=lr,
                                weight_decay=cfg.train["weight_decay"])

        for ep in range(cfg.train["epochs"]):
            loss = train_epoch(model, tr_loader, criterion, optim, device=device, distill_cfg=cfg.distill)
            print(f"[epoch {ep+1}/{cfg.train['epochs']}] loss={loss:.4f}")

    train_seconds = time.perf_counter() - _t_train_start
    print(f"[timing] baseline training: {train_seconds:.3f}s")

    model_name = cfg.model.get("name", "").lower()
    drop_at_eval_infer = bool(cfg.eval.get("dropout_at_eval", False))

    _t_baseline_infer_start = time.perf_counter()
    if cfg.model["name"].lower() == "logrobust":
        p_va = predict_logrobust_proba(lr_bundle, window_texts_va)
        p_te = predict_logrobust_proba(lr_bundle, window_texts_te)
    else:
        y_va, p_va, b_va = collect_window_scores(
            model, va_loader, device=device, dropout_at_eval=drop_at_eval_infer
        )
        y_te, p_te, b_te = collect_window_scores(
            model, te_loader, device=device, dropout_at_eval=drop_at_eval_infer
        )

    force_tune = model_name == "logbert"
    if model_name == "logbert" and cfg.model.get("logbert"):
        session_policy = cfg.model["logbert"].get("session_policy", "max")
        k_of_n = cfg.model["logbert"].get("k_of_n", cfg.eval["k_of_n"])
    else:
        session_policy = cfg.eval["session_policy"]
        k_of_n = cfg.eval["k_of_n"]

    print("[eval] selecting session threshold on VAL…")
    print(f"[eval] policy={session_policy} k={k_of_n}")
    p_win, r_win, f1_win, _ = anomaly_f1(y_te, (p_te >= 0.50).astype(int))
    print(f"[window][test] th=0.50  P={p_win:.3f} R={r_win:.3f} F1={f1_win:.3f}")

    liberty_tune_th = (
        str(dataset).strip().upper() == "LIBERTY"
        and cfg.eval.get("liberty_tune_threshold", True)
    )
    if cfg.eval.get("tune_threshold", False) or force_tune or liberty_tune_th:
        th = tune_session_threshold(y_va, p_va, b_va, policy=session_policy, k=k_of_n)
    else:
        th = 0.50

    drop_at_eval = drop_at_eval_infer
    if drop_at_eval:
        print(f"[session] dropout_at_eval enabled")
    print(f"[session] selected threshold on VAL: th*={th:.2f}")
    keys_te, ys_te, ypt = aggregate_to_session(
        y_te, p_te, b_te,
        policy=session_policy,
        th=th, k=k_of_n
    )
    ses_keys = [str(k) for k in keys_te]
    if model_name == "logrobust":
        sess_sum_p, sess_cnt_p = {}, {}
        for p, b in zip(p_te, b_te):
            sess_sum_p[b] = sess_sum_p.get(b, 0.0) + float(p)
            sess_cnt_p[b] = sess_cnt_p.get(b, 0) + 1
        p_te_s = np.array([sess_sum_p[k] / sess_cnt_p[k] for k in keys_te])
    keys_va, ys_va, ypt_va = aggregate_to_session(
        y_va, p_va, b_va,
        policy=session_policy,
        th=th, k=k_of_n
    )
    keys_va = [str(k) for k in keys_va]
    ys_va = np.asarray(ys_va)
    ypt_va = np.asarray(ypt_va)

    sess_col = session_key_col if session_key_col in seqs_df.columns else "BlockId"
    total_sessions_loaded = len(seqs_df)
    total_log_lines = seqs_df["EventSeq"].apply(lambda s: len(s) if s is not None else 0).sum()
    train_mask = seqs_df[sess_col].astype(str).isin(train_session_ids)
    seqs_train = seqs_df.loc[train_mask]
    n_train = len(seqs_train)
    n_train_anom = int(seqs_train["Label"].sum()) if "Label" in seqs_train.columns else 0
    n_val = len(keys_va)
    n_val_anom = int(np.sum(ys_va))
    n_test = len(keys_te)
    n_test_anom = int(np.sum(ys_te))
    total_sessions_used = n_train + n_val + n_test  # sessions that had >= window_size events
    pct_train = 100.0 * n_train / total_sessions_used if total_sessions_used else 0
    pct_val = 100.0 * n_val / total_sessions_used if total_sessions_used else 0
    pct_test = 100.0 * n_test / total_sessions_used if total_sessions_used else 0
    anom_pct_train = 100.0 * n_train_anom / n_train if n_train else 0
    anom_pct_val = 100.0 * n_val_anom / n_val if n_val else 0
    anom_pct_test = 100.0 * n_test_anom / n_test if n_test else 0
    dropped = total_sessions_loaded - total_sessions_used
    print(f"[dataset] loaded: total_sessions={total_sessions_loaded} total_log_lines={total_log_lines}")
    print(f"[dataset] used in split (len>=window): {total_sessions_used} sessions ({dropped} dropped)")
    print(f"[dataset] train: n_sessions={n_train} ({pct_train:.1f}% of used) n_anomalies={n_train_anom} anomaly%={anom_pct_train:.2f}%")
    print(f"[dataset] val:   n_sessions={n_val} ({pct_val:.1f}% of used) n_anomalies={n_val_anom} anomaly%={anom_pct_val:.2f}%")
    print(f"[dataset] test:  n_sessions={n_test} ({pct_test:.1f}% of used) n_anomalies={n_test_anom} anomaly%={anom_pct_test:.2f}%")

    p_s, r_s, f1_s, _ = anomaly_f1(ys_te, ypt)
    print(f"[session][test] policy={session_policy} th={th:.2f} "
        f"P={p_s:.3f} R={r_s:.3f} F1={f1_s:.3f}")

    if cfg.eval.get("export_auprc_csv", False):
        sess_sum, sess_cnt = {}, {}
        for p, b in zip(p_te, b_te):
            sess_sum[b] = sess_sum.get(b, 0.0) + float(p)
            sess_cnt[b] = sess_cnt.get(b, 0) + 1
        session_scores = np.array([sess_sum[k] / sess_cnt[k] for k in keys_te])
        _write_auprc_csv(
            keys_te, ys_te, session_scores,
            dataset=dataset,
            model_name=cfg.model.get("name", "deeplog"),
            out_dir=cfg.eval.get("auprc_csv_dir", "outputs"),
        )

    if cfg.distill.get("enabled", False):
        from logsable.dynamic import load_llm_session_labels
        m = load_llm_session_labels(
            db_path="logdb.sqlite",
            dataset=dataset,
            run_tag=run_tag,
            min_conf=cfg.distill.get("min_conf", 0.0),
        )
        teacher_map = {str(k): int(v[0]) for k, v in m.items()}  # (y, conf)
        teacher_vec = np.array([teacher_map.get(str(k), 0) for k in keys_te], dtype=int)

        p_t, r_t, f1_t, cm_t = anomaly_f1(ys_te, teacher_vec)
        print(f"[teacher-LLM][test] (dataset={dataset}) P={p_t:.3f} R={r_t:.3f} F1={f1_t:.3f} | cm={cm_t.tolist()} | support={teacher_vec.sum()}")

    baseline_inference_seconds = time.perf_counter() - _t_baseline_infer_start
    print(f"[timing] baseline inference: {baseline_inference_seconds:.3f}s")
    _timing_ok = True

    _t_hybrid_setup_start = time.perf_counter()
    ensure_dynamic_table("logdb.sqlite")

    if str(dataset).strip().upper() == "THUNDERBIRD":
        templates_df["EventTemplate"] = templates_df["EventTemplate"].fillna("[MISSING_TEMPLATE]").astype(str)

    _llm_cfg = cfg.llm or {}
    _llm_enabled = bool(_llm_cfg.get("enabled", True))
    _llm_model = str(_llm_cfg.get("model") or os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini-2024-07-18"))
    _llm_preview_k = int(_llm_cfg.get("session_preview_k", cfg.eval.get("llm_session_preview_k", 20)))
    _llm_use_cache = bool(_llm_cfg.get("use_cache", True))
    _llm_cross_run = bool(_llm_cfg.get("cross_run_cache", True))
    _llm_max_sessions = _llm_cfg.get("max_sessions", cfg.eval.get("llm_max_sessions"))
    _llm_sleep_s = float(_llm_cfg.get("sleep_s", cfg.eval.get("llm_sleep_s", 0.0)))

    session_prompt_items = build_session_prompt_items(
        seqs_df, templates_df,
        preview_k=_llm_preview_k,
        prompt_template=None,
        dataset=dataset,
        raw_df=raw_df if str(dataset).strip().upper() == "LIBERTY" else None,
    )
    if bool(_llm_cfg.get("export_prompts", False)):
        with open("llm_prompts_sessions.json", "w", encoding="utf-8") as f:
            json.dump(session_prompt_items, f, ensure_ascii=False, indent=2)
        print(f"[dynamic] wrote {len(session_prompt_items)} session prompts to llm_prompts_sessions.json")

    _sess_col = session_key_col if session_key_col in seqs_df.columns else "BlockId"
    _current_keys = set(seqs_df[_sess_col].astype(str))
    _ds_upper = str(dataset).strip().upper()
    if _ds_upper == "LIBERTY":
        _llm_promote = bool(cfg.eval.get("liberty_llm_ingest_promote", False))
    else:
        _llm_promote = bool(cfg.eval.get("llm_ingest_promote", False))

    run_llm_sessions_with_cache(
        session_prompt_items,
        db_path="logdb.sqlite",
        dataset=dataset,
        run_tag=run_tag,
        model=_llm_model,
        max_items=_llm_max_sessions,
        sleep_s=_llm_sleep_s,
        use_cache=_llm_use_cache,
        cross_run_cache=_llm_cross_run,
        llm_enabled=_llm_enabled,
        debug_jsonl_path=str(_llm_cfg.get("debug_sessions_jsonl", "debug_llm_sessions.jsonl")),
        valid_keys=_current_keys,
        promote_negatives=_llm_promote,
        promote_min_conf=float(cfg.eval.get("llm_ingest_promote_min_conf", 0.85)),
        promote_min_count=int(cfg.eval.get("llm_ingest_promote_min_count", 2)),
    )

    import sqlite3, pandas as pd, numpy as np
    dyn_pred = {}  # seq_key -> (is_high_risk, confidence)
    with sqlite3.connect("logdb.sqlite") as con:
        try:
            q = """SELECT TRIM(seq_key) AS seq_key, is_high_risk, COALESCE(confidence, 0.5) AS confidence
                   FROM dynamic_patterns WHERE dataset=? AND run_tag=?"""
            df_dyn = pd.read_sql(q, con, params=[dataset, run_tag])
            for _, row in df_dyn.iterrows():
                k = str(row["seq_key"])
                dyn_pred[k] = (int(row["is_high_risk"]), float(row["confidence"]))
        except Exception:
            q = "SELECT TRIM(seq_key) AS seq_key, is_high_risk FROM dynamic_patterns WHERE dataset=? AND run_tag=?"
            df_dyn = pd.read_sql(q, con, params=[dataset, run_tag])
            for _, row in df_dyn.iterrows():
                k = str(row["seq_key"])
                dyn_pred[k] = (int(row["is_high_risk"]), 0.5)

    current_run_keys = set(seqs_df[_sess_col].astype(str))
    n_in_db = len(dyn_pred)
    dyn_pred = {k: v for k, v in dyn_pred.items() if k in current_run_keys}
    if n_in_db > 0:
        print(f"[dynamic] DB rows for this run_tag: {n_in_db}; session keys matching current run seq_key: {len(dyn_pred)}")
    if n_in_db > 0 and len(dyn_pred) == 0:
        print("[dynamic] WARNING: no LLM rows match current run session keys (check dataset/run_tag or re-run LLM).")

    ses_set = set(ses_keys)
    dyn_set = set(dyn_pred.keys())
    n_match = len(ses_set & dyn_set)
    print(f"[dynamic] key alignment (test): len(dyn_pred)={len(dyn_pred)} len(ses_keys)={len(ses_keys)} intersection={n_match} (match_ratio={n_match / max(1, len(ses_keys)):.3f})")

    default_dyn_conf = float(cfg.eval.get("dynamic_min_conf", 0.5))
    dyn_low_conf_max = None
    hdfs_dyn_hi_min_conf = None
    if str(dataset).strip().upper() == "LIBERTY" and cfg.eval.get("liberty_dynamic_min_conf") is not None:
        default_dyn_conf = float(cfg.eval["liberty_dynamic_min_conf"])
    elif str(dataset).strip().upper() == "HDFS":
        default_dyn_conf = float(cfg.eval.get("hdfs_dynamic_min_conf", 0.0))
        if cfg.eval.get("hdfs_tune_dynamic_on_val", True) and len(keys_va) > 0:
            dyn_low_conf_max, hdfs_dyn_hi_min_conf, _hdfs_dyn_val_f1 = _tune_hdfs_dyn_rules_on_val(
                dyn_pred, keys_va, ys_va, cfg.eval
            )
            default_dyn_conf = float(hdfs_dyn_hi_min_conf)
            print(
                f"[dynamic] selected hi_min_conf={hdfs_dyn_hi_min_conf} "
                f"low_conf_anom_max={dyn_low_conf_max} "
                f"(val LLM-only F1={_hdfs_dyn_val_f1:.3f})"
            )
        else:
            if cfg.eval.get("hdfs_dynamic_low_conf_max") is not None:
                dyn_low_conf_max = float(cfg.eval["hdfs_dynamic_low_conf_max"])
            if cfg.eval.get("hdfs_dynamic_hi_min_conf") is not None:
                default_dyn_conf = float(cfg.eval["hdfs_dynamic_hi_min_conf"])
    dyn_vec = _dyn_vec_from_pred(
        dyn_pred, ses_keys, default_dyn_conf, low_conf_anom_max=dyn_low_conf_max
    )
    dyn_vec_va = (
        _dyn_vec_from_pred(dyn_pred, keys_va, default_dyn_conf, low_conf_anom_max=dyn_low_conf_max)
        if len(keys_va) else np.array([], dtype=int)
    )

    p_d, r_d, f1_d, _ = anomaly_f1(ys_te, dyn_vec)
    print(f"[dynamic-only][test] P={p_d:.3f} R={r_d:.3f} F1={f1_d:.3f} (support={dyn_vec.sum()})")

    hy_or  = np.maximum(ypt, dyn_vec)
    hy_and = np.minimum(ypt, dyn_vec)
    p_or, r_or, f1_or, _    = anomaly_f1(ys_te, hy_or)
    p_and, r_and, f1_and, _ = anomaly_f1(ys_te, hy_and)
    print(f"[hybrid-OR][test]  P={p_or:.3f} R={r_or:.3f} F1={f1_or:.3f}")
    print(f"[hybrid-AND][test] P={p_and:.3f} R={r_and:.3f} F1={f1_and:.3f}")


    con = kb_init("logdb.sqlite")
    eid_list = templates_df["EventId"].astype(str).tolist()
    eid2rid  = {eid: i for i, eid in enumerate(eid_list)}
    rid2eid  = {i: eid for i, eid in enumerate(eid_list)}

    def _coerce_list_for_std(x):
        if isinstance(x, (list, tuple, np.ndarray)): return list(x)
        if isinstance(x, str):
            s = x.strip()
            try:
                v = json.loads(s)
                if isinstance(v, list): return v
            except Exception:
                pass
            s = s.strip("[]")
            return [p.strip() for p in s.split(",") if p.strip() != ""]
        return list(x) if x is not None else []

    sess_col_local = session_key_col if session_key_col in seqs_df.columns else "BlockId"

    mask_train = seqs_df[sess_col_local].astype(str).isin(train_session_ids)
    seqs_train = seqs_df.loc[mask_train].reset_index(drop=True)

    if "Label" in seqs_train.columns:
        seqs_for_cluster = seqs_train[seqs_train["Label"].astype(int) == 1].reset_index(drop=True)
    else:
        raise RuntimeError("Clustering expects a Label column for selecting anomalies.")

    seqs_df_std = pd.DataFrame({
        "seq_key": seqs_train[sess_col_local].astype(str).values,
        "label":   seqs_train["Label"].astype(int).values,
        "row_ids": seqs_train["EventSeq_masked"].apply(
            lambda s: _to_row_ids(s, eid2rid) if s is not None else []
        ).values
    })

    import sqlite3
    with sqlite3.connect("logdb.sqlite") as kcon:
        cur = kcon.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS templates (
                row_id INTEGER PRIMARY KEY,
                event_id TEXT
            )""")
        cur.execute("DELETE FROM templates")
        cur.executemany(
            "INSERT INTO templates(row_id,event_id) VALUES (?,?)",
            [(i, rid2eid[i]) for i in range(len(rid2eid))]
        )

        cur.execute("""
            CREATE TABLE IF NOT EXISTS unigrams_by_label (
                row_id INTEGER,
                label  INTEGER,
                cnt    INTEGER
            )""")
        cur.execute("DELETE FROM unigrams_by_label")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ngrams_by_label (
                n      INTEGER,
                grams  TEXT,
                label  INTEGER,
                cnt    INTEGER
            )""")
        cur.execute("DELETE FROM ngrams_by_label")

        from collections import Counter
        uni_cnt = Counter()
        bi_cnt  = Counter()
        tri_cnt = Counter()

        for _, r in seqs_df_std.iterrows():
            lbl = int(r["label"])
            ids = list(map(int, r["row_ids"]))
            if not ids:
                continue
            for rid in ids:
                uni_cnt[(rid, lbl)] += 1
            for i in range(len(ids) - 1):
                bi_cnt[(" ".join(map(str, ids[i:i+2])), lbl)] += 1
            for i in range(len(ids) - 2):
                tri_cnt[(" ".join(map(str, ids[i:i+3])), lbl)] += 1

        if uni_cnt:
            cur.executemany(
                "INSERT INTO unigrams_by_label(row_id,label,cnt) VALUES (?,?,?)",
                [(rid, lbl, cnt) for (rid, lbl), cnt in uni_cnt.items()]
            )
        if bi_cnt:
            cur.executemany(
                "INSERT INTO ngrams_by_label(n,grams,label,cnt) VALUES (?,?,?,?)",
                [(2, grams, lbl, cnt) for (grams, lbl), cnt in bi_cnt.items()]
            )
        if tri_cnt:
            cur.executemany(
                "INSERT INTO ngrams_by_label(n,grams,label,cnt) VALUES (?,?,?,?)",
                [(3, grams, lbl, cnt) for (grams, lbl), cnt in tri_cnt.items()]
            )
        kcon.commit()

    auto_rules = []
    try:
        with sqlite3.connect("logdb.sqlite") as conn:
            cur = conn.cursor()
            eid_map = {row_id: eid for row_id, eid in cur.execute("SELECT row_id, event_id FROM templates")}

            q_disc = """
            WITH a AS (SELECT row_id, SUM(cnt) AS cnt FROM unigrams_by_label WHERE label=1 GROUP BY row_id),
                 n AS (SELECT row_id, SUM(cnt) AS cnt FROM unigrams_by_label WHERE label=0 GROUP BY row_id)
            SELECT a.row_id, COALESCE(a.cnt,0) AS ca, COALESCE(n.cnt,0) AS cn
            FROM a LEFT JOIN n USING(row_id)
            """
            min_cnt_disc = int(cfg.eval.get("min_cnt_disc", 10))
            disc_cn_max  = int(cfg.eval.get("disc_cn_max", 1))
            if str(dataset).strip().upper() == "LIBERTY":
                min_cnt_disc = int(cfg.eval.get("liberty_min_cnt_disc", min_cnt_disc))
            if str(dataset).strip().upper() == "HDFS":
                disc_cn_max = int(cfg.eval.get("hdfs_disc_cn_max", 0))
            if str(dataset).strip().upper() == "BGL":
                disc_cn_max = int(cfg.eval.get("bgl_disc_cn_max", 0))
                min_cnt_disc = int(cfg.eval.get("bgl_min_cnt_disc", min_cnt_disc))
            max_disc     = int(cfg.eval.get("max_disc_rules", 200))
            if str(dataset).strip().upper() == "BGL":
                max_disc = int(cfg.eval.get("bgl_max_disc_rules", max_disc))

            disc_candidates = []
            for rid, ca, cn in cur.execute(q_disc):
                if ca >= min_cnt_disc and cn <= disc_cn_max:
                    disc_candidates.append((rid, ca, cn))
            disc_candidates.sort(key=lambda t: (t[1], -t[2]), reverse=True)
            for rid, ca, cn in disc_candidates[:max_disc]:
                eid = eid_map.get(rid, f"E{rid}")
                auto_rules.append({
                    "name": f"disc_anom_{eid}",
                    "if_any": [],
                    "if_all": [{"min_count": {"event_id": eid, "count": 1}}],
                    "explanation": f"{eid} nearly-exclusive to anomalies (a={ca}, n={cn}).",
                    "confidence": 0.95
                })

            min_cnt_anom = int(cfg.eval.get("min_cnt_anom", 2))
            min_lift     = float(cfg.eval.get("min_lift", 1.3))
            if str(dataset).strip().upper() == "LIBERTY":
                min_cnt_anom = int(cfg.eval.get("liberty_min_cnt_anom", min_cnt_anom))
                min_lift     = float(cfg.eval.get("liberty_min_lift", min_lift))
            if str(dataset).strip().upper() == "BGL":
                min_cnt_anom = int(cfg.eval.get("bgl_min_cnt_anom", min_cnt_anom))
                min_lift     = float(cfg.eval.get("bgl_min_lift", min_lift))
            top_k_uni    = int(cfg.eval.get("top_k_unigrams", 20))
            max_lift_rules = int(cfg.eval.get("max_lift_rules", 500))
            if str(dataset).strip().upper() == "BGL":
                max_lift_rules = int(cfg.eval.get("bgl_max_lift_rules", max_lift_rules))

            uni_candidates = []
            for rid, ca, cn in cur.execute(q_disc):
                if ca >= min_cnt_anom:
                    lift = (ca + 1.0) / (cn + 1.0)
                    if lift >= min_lift:
                        uni_candidates.append((rid, ca, cn, lift))
            uni_candidates.sort(key=lambda t: (t[3], t[1]), reverse=True)
            for rid, ca, cn, lift in uni_candidates[:top_k_uni]:
                eid = eid_map.get(rid, f"E{rid}")
                auto_rules.append({
                    "name": f"uni_lift_{eid}",
                    "if_any": [],
                    "if_all": [{"min_count": {"event_id": eid, "count": 1}}],
                    "explanation": f"Unigram {eid} lifted (a={ca}, n={cn}, lift={lift:.2f}).",
                    "confidence": 0.8
                })

            ngram_lift_candidates = []
            for n in (2, 3):
                q = f"""
                WITH a AS (SELECT grams, SUM(cnt) AS cnt FROM ngrams_by_label WHERE label=1 AND n={n} GROUP BY grams),
                     nrm AS (SELECT grams, SUM(cnt) AS cnt FROM ngrams_by_label WHERE label=0 AND n={n} GROUP BY grams)
                SELECT a.grams, COALESCE(a.cnt,0) AS ca, COALESCE(nrm.cnt,0) AS cn
                FROM a LEFT JOIN nrm USING(grams)
                """
                for grams, ca, cn in cur.execute(q):
                    if ca >= min_cnt_anom:
                        lift = (ca + 1.0) / (cn + 1.0)
                        if lift >= min_lift:
                            ngram_lift_candidates.append((n, grams, ca, cn, lift))
            ngram_lift_candidates.sort(key=lambda t: (t[4], t[2]), reverse=True)
            for n, grams, ca, cn, lift in ngram_lift_candidates[:max_lift_rules]:
                ids = list(map(int, grams.split()))
                auto_rules.append({
                    "name": f"ng{n}_lift_{grams.replace(' ','_')}",
                    "if_any": [],
                    "if_all": [{"ordered_subset": ids}],
                    "explanation": f"{n}-gram {ids} lifted (a={ca}, n={cn}, lift={lift:.2f}).",
                    "confidence": 0.8
                })

        print(f"[rules] mined auto_rules: {len(auto_rules)}")
        if auto_rules[:5]:
            print("[rules] sample:", [r["name"] for r in auto_rules[:5]])
    except Exception as e:
        print(f"[rules] mining failed: {e}")
        auto_rules = []

    if dataset.strip().upper() == "THUNDERBIRD" and len(auto_rules) == 0 and "EventSeq_masked" in seqs_df.columns:
        auto_rules.append({
            "name": "empty_seq_anomaly",
            "if_any": [],
            "if_all": [{"is_empty": True}],
            "explanation": "Empty or masked-only session.",
            "confidence": 0.9,
        })
        print("[rules] added empty-sequence rule")

    rid2tmpl = {i: str(t) for i, t in enumerate(templates_df["EventTemplate"].tolist())}

    if str(dataset).strip().upper() == "LIBERTY":
        from logsable.rules import mine_liberty_template_keyword_rules
        kw_rules = mine_liberty_template_keyword_rules(
            seqs_train,
            rid2tmpl,
            session_key_col=sess_col_local,
            min_anom_sessions=int(cfg.eval.get("liberty_keyword_min_anom_sessions", 3)),
            max_norm_sessions=int(cfg.eval.get("liberty_keyword_max_norm_sessions", 0)),
        )
        if kw_rules:
            print(f"[rules] template-keyword rules: {len(kw_rules)} "
                  f"(names={[r['name'] for r in kw_rules[:5]]}...)")
            auto_rules = (auto_rules or []) + kw_rules

    def rules_pred_by_session(seqs_df, rules, rid2eid, rid2tmpl_map=None):
        preds = {}
        sess_col_local = session_key_col if session_key_col in seqs_df.columns else "BlockId"
        for _, row in seqs_df.iterrows():
            sess_id = str(row[sess_col_local])
            seq_rids = _to_row_ids(row["EventSeq"], eid2rid)
            fired = any(
                rule_fires_on_seq(seq_rids, r, rid2eid, rid2tmpl=rid2tmpl_map)
                for r in rules
            )
            preds[sess_id] = 1 if fired else 0
        return preds

    seqs_for_rules = seqs_df.copy()
    seqs_for_rules["EventSeq"] = seqs_for_rules["EventSeq_masked"]
    sess_col_rules = session_key_col if session_key_col in seqs_for_rules.columns else "BlockId"
    if len(keys_va) > 0 and auto_rules:
        from logsable.rules import filter_rules_on_validation
        va_mask = seqs_for_rules[sess_col_rules].astype(str).isin(set(map(str, keys_va)))
        auto_rules = filter_rules_on_validation(
            auto_rules,
            seqs_for_rules.loc[va_mask],
            keys_va,
            ys_va,
            sess_col_rules,
            lambda s: _to_row_ids(s, eid2rid),
            rid2eid,
            rid2tmpl=rid2tmpl,
            cfg_eval=cfg.eval,
            dataset=dataset,
        )
    rule_by_sess = rules_pred_by_session(seqs_for_rules, auto_rules, rid2eid, rid2tmpl_map=rid2tmpl)
    rhat_te_s    = np.array([rule_by_sess.get(str(k), 0) for k in ses_keys], dtype=int)
    rhat_va      = np.array([rule_by_sess.get(str(k), 0) for k in keys_va], dtype=int) if len(keys_va) else np.array([], dtype=int)

    p_rt, r_rt, f1_rt, cm_rt = anomaly_f1(ys_te, rhat_te_s)
    print(f"[RULES] TEST session P={p_rt:.3f} R={r_rt:.3f} F1={f1_rt:.3f} | cm={cm_rt.tolist()}")
    if int(rhat_te_s.sum()) == 0:
        print("[rules] WARNING: no rule fires on TEST.")
    else:
        print(f"[rules] TEST rule support={int(rhat_te_s.sum())}/{len(rhat_te_s)}")

    hy_te = ((ypt == 1) | (rhat_te_s == 1)).astype(int)
    p_ht, r_ht, f1_ht, cm_ht = anomaly_f1(ys_te, hy_te)
    print(f"[HYBRID] TEST session P={p_ht:.3f} R={r_ht:.3f} F1={f1_ht:.3f} | cm={cm_ht.tolist()}")

    cluster_cfg_peek = getattr(cfg, "cluster", None)
    if cluster_cfg_peek is not None and not isinstance(cluster_cfg_peek, dict):
        try:
            cluster_cfg_peek = cluster_cfg_peek.to_dict()
        except Exception:
            try:
                cluster_cfg_peek = dict(cluster_cfg_peek)
            except Exception:
                cluster_cfg_peek = vars(cluster_cfg_peek) if hasattr(cluster_cfg_peek, "__dict__") else {}
    cluster_cfg_peek = cluster_cfg_peek or {}
    if (
        _timing_ok
        and not _hybrid_online_recorded
        and not cluster_cfg_peek.get("enabled", False)
        and len(ses_keys) > 0
    ):
        if _t_hybrid_setup_start is not None and hybrid_setup_seconds == 0.0:
            hybrid_setup_seconds = time.perf_counter() - _t_hybrid_setup_start
        hybrid_online_seconds, hybrid_ms_per_session = _time_hybrid_online_inference(
            seqs_for_rules, session_key_col, ses_keys, ypt, dyn_pred,
            auto_rules or [], eid2rid, rid2eid, rid2tmpl, default_dyn_conf, kb_vec=None,
            low_conf_anom_max=dyn_low_conf_max,
        )
        ypt_map = {str(k): int(ypt[i]) for i, k in enumerate(keys_te)}
        hybrid_single_session_ms = _benchmark_single_session_hybrid(
            seqs_for_rules, session_key_col, str(ses_keys[0]), ypt_map,
            dyn_pred, auto_rules or [], eid2rid, rid2eid, rid2tmpl, default_dyn_conf, kb_by_key=None,
            low_conf_anom_max=dyn_low_conf_max,
        )
        _hybrid_online_recorded = True
        print(
            f"[timing] hybrid online (test, no-cluster): {hybrid_online_seconds:.3f}s "
            f"({hybrid_ms_per_session:.3f} ms/session, single={hybrid_single_session_ms:.3f} ms)"
        )

    cluster_cfg = getattr(cfg, "cluster", None)

    if cluster_cfg is None:
        cluster_cfg = {}
    elif not isinstance(cluster_cfg, dict):
        try:
            cluster_cfg = cluster_cfg.to_dict()  # type: ignore[attr-defined]
        except Exception:
            try:
                cluster_cfg = dict(cluster_cfg)
            except Exception:
                try:
                    cluster_cfg = vars(cluster_cfg)
                except Exception:
                    cluster_cfg = {}

    if not cluster_cfg.get("enabled", False) and cfg.eval.get("tune_fusion", True) and len(keys_va) > 0:
        best_name, best_te_pred, best_conf, _best_val_f1 = _pick_final_hybrid_on_val(
            ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, ses_keys,
            rhat_va, rhat_te_s, None, None,
            anomaly_f1, cfg.eval, default_dyn_conf=default_dyn_conf,
            low_conf_anom_max=dyn_low_conf_max,
            fixed_dyn_min_conf=hdfs_dyn_hi_min_conf,
            cluster_enabled=False,
        )
        if cfg.eval.get("base_gate_enabled", False) and cfg.model["name"].lower() == "logrobust":
            lo = float(cfg.eval.get("base_gate_lo", 0.05))
            hi = float(cfg.eval.get("base_gate_hi", 0.95))
            best_te_pred = gate_fusion_with_base_proba(p_te_s, ypt, best_te_pred, lo, hi)
        p_f, r_f, f1_f, cm_f = anomaly_f1(ys_te, best_te_pred)
        print(f"[FINAL HYBRID] (3-way, dynamic_conf={best_conf}, strategy={best_name}) P={p_f:.3f} R={r_f:.3f} F1={f1_f:.3f} | cm={cm_f.tolist()}")
        _print_final_hybrid_ablations(
            ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, ses_keys,
            rhat_va, rhat_te_s, None, None, cfg, cfg.eval,
            dyn_low_conf_max, best_conf, best_name, f1_f,
            p_te_s=p_te_s if model_name == "logrobust" else None,
            cluster_enabled=False,
        )

    if cluster_cfg.get("enabled", False):
        print("[cluster] building sequence vectors…")
        sess_col_local = session_key_col if session_key_col in seqs_df.columns else "BlockId"
        mask_train = seqs_df[sess_col_local].astype(str).isin(train_session_ids)
        seqs_train = seqs_df.loc[mask_train].reset_index(drop=True)

        if "Label" not in seqs_train.columns:
            raise RuntimeError("Clustering expects a Label column.")

        _ds_cluster = str(dataset).strip().upper()
        if _ds_cluster == "LIBERTY":
            cluster_cfg = dict(cluster_cfg)
            cluster_cfg["source"] = cfg.eval.get("liberty_cluster_source", "train")
            cluster_cfg["min_cluster_size"] = int(cfg.eval.get("liberty_cluster_min_cluster_size", 3))
            cluster_cfg["vectorizer"] = str(cfg.eval.get("liberty_cluster_vectorizer", "tfidf-template"))
            print(
                f"[cluster] config: source={cluster_cfg['source']} "
                f"min_cluster_size={cluster_cfg['min_cluster_size']} "
                f"vectorizer={cluster_cfg['vectorizer']}"
            )

        seqs_for_cluster = seqs_train[seqs_train["Label"].astype(int) == 1].reset_index(drop=True)
        print(f"[cluster] TRAIN anomalies only: n={len(seqs_for_cluster)}")

        source = cluster_cfg.get("source", "all")
        if source == "train":
            if goto_post_model:
                print("[cluster] source=train unavailable; using all sessions")
                seqs_for_cluster = seqs_df.reset_index(drop=True)
            elif _ds_cluster == "LIBERTY":
                seqs_for_cluster = seqs_train[seqs_train["Label"].astype(int) == 1].reset_index(drop=True)
                print(f"[cluster] train anomalies: n={len(seqs_for_cluster)}")
            else:
                tr_keys = set(map(str, TR_blk))
                mask = seqs_df[session_key_col].astype(str).isin(tr_keys)
                seqs_for_cluster = seqs_df.loc[mask].reset_index(drop=True)
        else:
            seqs_for_cluster = seqs_df.reset_index(drop=True)

        if "Label" in seqs_for_cluster.columns:
            seqs_for_cluster = seqs_for_cluster[seqs_for_cluster["Label"].astype(int) == 1].reset_index(drop=True)
            print(f"[cluster] filtered to anomalies only: n={len(seqs_for_cluster)}")

        n_train_anom = len(seqs_for_cluster)
        clustering_ran = False
        bundle = None
        seqs_for_cluster_for_matrix = None
        cluster_llm_results = []
        if n_train_anom == 0:
            print("[cluster] n_train_anom==0, skipping clustering")
        else:
            seqs_for_cluster_for_matrix = seqs_for_cluster.copy()
            seqs_for_cluster_for_matrix["EventSeq"] = seqs_for_cluster_for_matrix["EventSeq_masked"]
            nonempty = seqs_for_cluster_for_matrix["EventSeq"].apply(lambda s: len(s or []) > 0)
            seqs_for_cluster_for_matrix = seqs_for_cluster_for_matrix.loc[nonempty].reset_index(drop=True)
            if len(seqs_for_cluster_for_matrix) == 0:
                print("[cluster] all anomaly sequences empty after masking, skipping clustering (no cluster rules)")
            else:
                clustering_ran = True
                X, _ = build_sequence_matrix(
                    seqs_for_cluster_for_matrix,
                    templates_df,
                    vectorizer=cluster_cfg.get("vectorizer", "tfidf-eid")
                )
                n_anom = len(seqs_for_cluster_for_matrix)
                print(f"[cluster] vectorized: shape={getattr(X,'shape',None)} (n_anomalies={n_anom})")

                min_cs = cluster_cfg.get("min_cluster_size", 10)
                labels, probs, _ = run_hdbscan(
                    X,
                    min_cluster_size=min_cs,
                    min_samples=cluster_cfg.get("min_samples", None),
                    metric=cluster_cfg.get("metric", "cosine"),
                )
                n_clusters = len(set(labels[labels >= 0]))
                for retry_cs in [5, 3, 2]:
                    if n_clusters > 0:
                        break
                    if retry_cs >= min_cs:
                        continue
                    labels, probs, _ = run_hdbscan(
                        X, min_cluster_size=retry_cs,
                        min_samples=cluster_cfg.get("min_samples", None),
                        metric=cluster_cfg.get("metric", "cosine"),
                    )
                    n_clusters = len(set(labels[labels >= 0]))
                    if n_clusters > 0:
                        print(f"[cluster] retried min_cluster_size={retry_cs}: clusters={n_clusters}, noise={(labels<0).sum()}")

                used_single_cluster = False
                if n_clusters == 0 and n_anom > 0:
                    labels = np.zeros(n_anom, dtype=np.int64)
                    probs = np.ones(n_anom, dtype=np.float64)
                    used_single_cluster = True
                    print(f"[cluster] single cluster assigned (n={n_anom})")

                centers, extremes = pick_representatives(
                    X, labels, probs, k_select=cluster_cfg.get("select_k", 3)
                )
                n_clusters_final = len(set(labels[labels >= 0]))
                print(f"[cluster] clusters={n_clusters_final}, noise={(labels<0).sum()}" + (" [single cluster]" if used_single_cluster else ""))

                bundle = build_cluster_bundle(
                    seqs_for_cluster_for_matrix, templates_df, labels, probs, centers, extremes
                )
            if clustering_ran and bundle is not None:
                cluster_prompt_items = build_cluster_prompt_items(
                    bundle, dataset=cfg.data.get("dataset", "HDFS")
                )
                if bool(_llm_cfg.get("export_prompts", False)):
                    export_llm_prompts_for_clusters(
                        bundle, out_path="llm_prompts_clusters.json",
                        dataset=cfg.data.get("dataset", "HDFS"),
                    )
                cluster_llm_results = run_llm_clusters_with_cache(
                    cluster_prompt_items,
                    db_path="logdb.sqlite",
                    dataset=dataset,
                    run_tag=run_tag,
                    model=_llm_model,
                    max_items=_llm_cfg.get("max_clusters", cfg.eval.get("llm_max_clusters")),
                    sleep_s=_llm_sleep_s,
                    use_cache=_llm_use_cache,
                    cross_run_cache=_llm_cross_run,
                    llm_enabled=_llm_enabled,
                    debug_jsonl_path=str(_llm_cfg.get("debug_clusters_jsonl", "debug_llm_clusters.jsonl")),
                )
            if not clustering_ran:
                promote_rules_to_kb(
                    [], cfg, source="llm_cluster", scope="dataset", run_tag=run_tag,
                    db_path="logdb.sqlite", clear_existing=True,
                )

            llm_rules_raw = (
                ingest_llm_cluster_rules_from_responses(cluster_llm_results)
                if clustering_ran and cluster_llm_results
                else []
            )
            llm_rules_std = (
                standardize_llm_rules_to_row_ids(llm_rules_raw, eid2rid) if llm_rules_raw else []
            )
            cluster_kb_rules = []
            n_cluster_candidates = 0
            if clustering_ran and bundle is not None and seqs_for_cluster_for_matrix is not None:
                from logsable.rules import build_cluster_kb_rules, filter_cluster_rules_on_validation
                cluster_kb_rules, n_repr, n_llm_rel = build_cluster_kb_rules(
                    bundle, llm_rules_std, seqs_for_cluster_for_matrix,
                    session_key_col, lambda s: _to_row_ids(s, eid2rid),
                    eval_cfg=cfg.eval, dataset=dataset, rid2eid=eid2rid,
                )
                n_cluster_candidates = len(cluster_kb_rules)
                print(
                    f"[cluster] built KB candidates: repr={n_repr} llm_ngram={n_llm_rel} "
                    f"combined={n_cluster_candidates}"
                )
                if str(dataset).strip().upper() == "LIBERTY":
                    _kw_cluster = [
                        r for r in (auto_rules or [])
                        if str(r.get("name", "")).startswith("liberty_kw_")
                    ]
                    if _kw_cluster:
                        cluster_kb_rules = _kw_cluster + list(cluster_kb_rules)
                        n_cluster_candidates = len(cluster_kb_rules)
                        print(
                            f"[cluster] prepended {len(_kw_cluster)} "
                            f"template-keyword rules (combined={n_cluster_candidates})"
                        )
                if cluster_kb_rules and len(keys_va) > 0:
                    va_mask = seqs_for_rules[sess_col_rules].astype(str).isin(set(map(str, keys_va)))
                    cluster_kb_rules = filter_cluster_rules_on_validation(
                        cluster_kb_rules,
                        seqs_for_rules.loc[va_mask],
                        keys_va,
                        ys_va,
                        sess_col_rules,
                        lambda s: _to_row_ids(s, eid2rid),
                        rid2eid,
                        rid2tmpl=rid2tmpl,
                        eval_cfg=cfg.eval,
                        dataset=dataset,
                    )
                    print(f"[cluster] validation-filtered cluster KB rules: {len(cluster_kb_rules)}")
                elif cluster_kb_rules:
                    print(
                        f"[cluster] no validation split; cluster rules require validation filtering "
                        f"({n_cluster_candidates} candidates skipped)"
                    )
                    cluster_kb_rules = []
            elif llm_rules_std:
                from logsable.rules import prepare_llm_cluster_rules, filter_cluster_rules_on_validation
                from logsable.rules import liberty_cluster_rules_use_event_ids
                cluster_kb_rules = prepare_llm_cluster_rules(
                    llm_rules_std, mode=cfg.eval.get("cluster_llm_rule_mode", "if_any_only")
                )
                if str(dataset).strip().upper() == "LIBERTY":
                    cluster_kb_rules = liberty_cluster_rules_use_event_ids(cluster_kb_rules, eid2rid)
                n_cluster_candidates = len(cluster_kb_rules)
                if cluster_kb_rules and len(keys_va) > 0:
                    va_mask = seqs_for_rules[sess_col_rules].astype(str).isin(set(map(str, keys_va)))
                    cluster_kb_rules = filter_cluster_rules_on_validation(
                        cluster_kb_rules,
                        seqs_for_rules.loc[va_mask],
                        keys_va,
                        ys_va,
                        sess_col_rules,
                        lambda s: _to_row_ids(s, eid2rid),
                        rid2eid,
                        rid2tmpl=rid2tmpl,
                        eval_cfg=cfg.eval,
                        dataset=dataset,
                    )
                elif cluster_kb_rules:
                    print(
                        f"[cluster] no validation split; cluster rules require validation filtering "
                        f"({n_cluster_candidates} candidates skipped)"
                    )
                    cluster_kb_rules = []
            llm_rules_std = cluster_kb_rules
            if llm_rules_std:
                merged_rules = (auto_rules or []) + llm_rules_std

                rule_by_blk_merged = rules_pred_by_session(
                    seqs_for_rules, merged_rules, rid2eid, rid2tmpl_map=rid2tmpl
                )
                rhat_te_s_merged   = np.array([rule_by_blk_merged.get(str(k), 0) for k in keys_te], dtype=int)
                p_r2, r_r2, f1_r2, cm_r2 = anomaly_f1(ys_te, rhat_te_s_merged)
                print(f"[RULES+LLM] TEST session P={p_r2:.3f} R={r_r2:.3f} F1={f1_r2:.3f} | cm={cm_r2.tolist()}")

                hy_te2 = ((ypt == 1) | (rhat_te_s_merged == 1)).astype(int)
                p_h2, r_h2, f1_h2, cm_h2 = anomaly_f1(ys_te, hy_te2)
                print(f"[HYBRID+LLM] TEST session P={p_h2:.3f} R={r_h2:.3f} F1={f1_h2:.3f} | cm={cm_h2.tolist()}")
            else:
                print("[RULES+LLM] No cluster LLM rules; skipping cluster rule evaluation.")

            accepted_rules = llm_rules_std if llm_rules_std else []

            n_promoted_cluster = 0
            if clustering_ran:
                n_promoted_cluster = promote_rules_to_kb(
                    accepted_rules, cfg, source="llm_cluster", scope="dataset", run_tag=run_tag,
                    db_path="logdb.sqlite", clear_existing=True,
                )
                if (
                    str(dataset).strip().upper() == "LIBERTY"
                    and cfg.eval.get("liberty_promote_auto_rules_to_kb", False)
                    and auto_rules
                ):
                    cap = int(cfg.eval.get("liberty_promote_auto_kb_rules", 300))
                    promote_rules_to_kb(
                        auto_rules[:cap], cfg, source="auto_mined", scope="dataset",
                        run_tag=run_tag, db_path="logdb.sqlite", clear_existing=True,
                    )

            _kb_source = "llm_cluster" if str(dataset).strip().upper() == "LIBERTY" else None
            kb_rules = load_active_kb_rules(
                cfg, allow_global=False, run_tag=run_tag,
                max_rules=cfg.eval.get("max_kb_rules"), db_path="logdb.sqlite",
                source=_kb_source,
            ) if clustering_ran else []
            if clustering_ran and n_promoted_cluster != len(accepted_rules):
                print(
                    f"[KB] WARN: promoted {n_promoted_cluster} cluster rules but "
                    f"accepted_rules had {len(accepted_rules)}"
                )
            if clustering_ran:
                n_loaded_cluster = count_active_kb_rules(
                    cfg, run_tag=run_tag, source="llm_cluster", db_path="logdb.sqlite",
                )
                if n_loaded_cluster != n_promoted_cluster:
                    print(
                        f"[KB] WARN: DB has {n_loaded_cluster} llm_cluster rules but "
                        f"promoted {n_promoted_cluster}"
                    )
            kb_rule_by_blk = rules_pred_by_session(seqs_for_rules, kb_rules, rid2eid, rid2tmpl_map=rid2tmpl)
            cluster_rules_eval = accepted_rules if accepted_rules else kb_rules
            cluster_rule_by_blk = rules_pred_by_session(
                seqs_for_rules, cluster_rules_eval, rid2eid, rid2tmpl_map=rid2tmpl
            )
            if clustering_ran and llm_rules_raw:
                try:
                    from logsable.explainability import build_and_store_cluster_artifacts
                    build_and_store_cluster_artifacts(
                        bundle,
                        cluster_results=cluster_llm_results,
                        rid2eid=rid2eid, run_tag=run_tag, dataset=dataset, db_path="logdb.sqlite"
                    )
                    from logsable.kb import store_seq_key_to_cluster_from_bundle
                    store_seq_key_to_cluster_from_bundle(bundle, run_tag, dataset, db_path="logdb.sqlite")
                except Exception as e:
                    print(f"[EXPLAIN] cluster artifact build failed: {e}")
            kb_vec = np.array([kb_rule_by_blk.get(str(k), 0) for k in keys_te], dtype=int)
            kb_vec_va = np.array([kb_rule_by_blk.get(str(k), 0) for k in keys_va], dtype=int) if len(keys_va) else np.array([], dtype=int)
            cluster_vec = np.array([cluster_rule_by_blk.get(str(k), 0) for k in keys_te], dtype=int)
            cluster_vec_va = (
                np.array([cluster_rule_by_blk.get(str(k), 0) for k in keys_va], dtype=int)
                if len(keys_va) else np.array([], dtype=int)
            )
            if str(dataset).strip().upper() == "LIBERTY" and len(cluster_rules_eval) > 0:
                _p_c, _r_c, _f1_c, _ = anomaly_f1(ys_te, cluster_vec)
                print(
                    f"[cluster] TEST cluster-only rules: n={len(cluster_rules_eval)} "
                    f"support={int(cluster_vec.sum())}/{len(cluster_vec)} "
                    f"P={_p_c:.3f} R={_r_c:.3f} F1={_f1_c:.3f}"
                )

            if _timing_ok and not _hybrid_online_recorded and len(ses_keys) > 0:
                if _t_hybrid_setup_start is not None and hybrid_setup_seconds == 0.0:
                    hybrid_setup_seconds = time.perf_counter() - _t_hybrid_setup_start
                online_rules = kb_rules if kb_rules else (auto_rules or [])
                hybrid_online_seconds, hybrid_ms_per_session = _time_hybrid_online_inference(
                    seqs_for_rules, session_key_col, ses_keys, ypt, dyn_pred,
                    online_rules, eid2rid, rid2eid, rid2tmpl, default_dyn_conf, kb_vec=kb_vec,
                    low_conf_anom_max=dyn_low_conf_max,
                )
                ypt_map = {str(k): int(ypt[i]) for i, k in enumerate(keys_te)}
                kb_map = {str(k): int(kb_vec[i]) for i, k in enumerate(keys_te)}
                hybrid_single_session_ms = _benchmark_single_session_hybrid(
                    seqs_for_rules, session_key_col, str(ses_keys[0]), ypt_map,
                    dyn_pred, online_rules, eid2rid, rid2eid, rid2tmpl, default_dyn_conf, kb_by_key=kb_map,
                    low_conf_anom_max=dyn_low_conf_max,
                )
                _hybrid_online_recorded = True
                print(
                    f"[timing] hybrid online (test, KB rules): {hybrid_online_seconds:.3f}s "
                    f"({hybrid_ms_per_session:.3f} ms/session, single={hybrid_single_session_ms:.3f} ms)"
                )

            best_conf = default_dyn_conf
            _liberty_hybrid_preset = (
                str(dataset).strip().upper() == "LIBERTY"
                and cfg.eval.get("liberty_hybrid_base_ngram", True)
                and cfg.eval.get("liberty_hybrid_shortcut_final", False)
                and len(keys_va) > 0
            )
            if _liberty_hybrid_preset:
                _, _, f1_ngram_va, _ = anomaly_f1(ys_va, rhat_va)
                _, _, f1_base_va, _ = anomaly_f1(ys_va, ypt_va)
                ngram_min = float(cfg.eval.get("liberty_hybrid_ngram_val_f1_min", 0.85))
                base_va_min = float(cfg.eval.get("liberty_hybrid_base_val_f1_min", 0.5))
                use_ngram_only = bool(cfg.eval.get("liberty_hybrid_use_ngram_only_when_base_bad", True))
                if f1_ngram_va >= ngram_min and int(rhat_va.sum()) > 0:
                    if use_ngram_only and f1_base_va < base_va_min:
                        final_pred = np.asarray(rhat_te_s, dtype=int).copy()
                        best_conf = default_dyn_conf
                        best_name = "ngram_only"
                        print(
                            f"[FINAL HYBRID] (ngram_only, val_ngram_F1={f1_ngram_va:.3f}, "
                            f"val_base_F1={f1_base_va:.3f}) ",
                            end="",
                        )
                    else:
                        final_pred = np.maximum(ypt, rhat_te_s)
                        best_conf = default_dyn_conf
                        best_name = "base+ngram_or"
                        print(
                            f"[FINAL HYBRID] (base+ngram_or, val_ngram_F1={f1_ngram_va:.3f}) ",
                            end="",
                        )
                elif cfg.eval.get("tune_fusion", True):
                    best_name, final_pred, best_conf, _best_val_f1 = _pick_final_hybrid_on_val(
                        ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, ses_keys,
                        rhat_va, rhat_te_s, kb_vec_va, kb_vec,
                        anomaly_f1, cfg.eval, default_dyn_conf=default_dyn_conf,
                        low_conf_anom_max=dyn_low_conf_max,
                        fixed_dyn_min_conf=hdfs_dyn_hi_min_conf,
                        cluster_enabled=True,
                    )
                    print(
                        f"[FINAL HYBRID] (4-way, dynamic_conf={best_conf}, strategy={best_name}, "
                        f"val_hybrid_F1={_best_val_f1:.3f}) ",
                        end="",
                    )
                else:
                    final_pred = np.maximum(ypt, rhat_te_s)
                    best_conf = default_dyn_conf
                    best_name = "base+ngram_or"
                    print(f"[FINAL HYBRID] (base+ngram_or) ", end="")
            elif len(keys_va) > 0:
                best_name, final_pred, best_conf, _best_val_f1 = _pick_final_hybrid_on_val(
                    ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, ses_keys,
                    rhat_va, rhat_te_s, kb_vec_va if "kb_vec_va" in locals() else None,
                    kb_vec if "kb_vec" in locals() else None,
                    anomaly_f1, cfg.eval, default_dyn_conf=default_dyn_conf,
                    low_conf_anom_max=dyn_low_conf_max,
                    fixed_dyn_min_conf=hdfs_dyn_hi_min_conf,
                    cluster_enabled=True,
                )
                print(f"[FINAL HYBRID] (strategy={best_name}, val_hybrid_F1={_best_val_f1:.3f}) ", end="")
            else:
                kb_v = kb_vec if "kb_vec" in locals() else np.zeros_like(ypt)
                final_pred = np.maximum.reduce([ypt, rhat_te_s, dyn_vec, kb_v])
                best_name = "4-way_or"
                print(f"[FINAL HYBRID] (4-way OR) ", end="")
            if cfg.eval.get("base_gate_enabled", False) and cfg.model["name"].lower() == "logrobust":
                lo = float(cfg.eval.get("base_gate_lo", 0.05))
                hi = float(cfg.eval.get("base_gate_hi", 0.95))
                final_pred = gate_fusion_with_base_proba(p_te_s, ypt, final_pred, lo, hi)
            p_f, r_f, f1_f, cm_f = anomaly_f1(ys_te, final_pred)
            print(f"P={p_f:.3f} R={r_f:.3f} F1={f1_f:.3f} | cm={cm_f.tolist()}")

            _kb_va_abl = kb_vec_va if "kb_vec_va" in locals() else None
            _kb_te_abl = kb_vec if "kb_vec" in locals() else None
            _print_final_hybrid_ablations(
                ys_va, ypt_va, ys_te, ypt, dyn_pred, keys_va, ses_keys,
                rhat_va, rhat_te_s, _kb_va_abl, _kb_te_abl, cfg, cfg.eval,
                dyn_low_conf_max, best_conf, best_name, f1_f,
                p_te_s=p_te_s if model_name == "logrobust" else None,
                cluster_enabled=True,
            )

            rules_for_explain = kb_rules if kb_rules else auto_rules
            if cfg.eval.get("explain_sessions", True) and rules_for_explain and len(ses_keys) > 0:
                try:
                    from logsable.explainability import explain_sessions_batch
                    from logsable.kb import load_rule_to_cluster_map
                    rule_to_cluster = load_rule_to_cluster_map(run_tag, dataset, db_path="logdb.sqlite") if clustering_ran else {}
                    sess_col = session_key_col if session_key_col in seqs_for_rules.columns else "BlockId"
                    seqs_te_explain = seqs_for_rules[seqs_for_rules[sess_col].astype(str).isin(ses_keys)].copy()
                    if len(seqs_te_explain) == 0:
                        seqs_te_explain = seqs_for_rules[seqs_for_rules["BlockId"].astype(str).isin(ses_keys)].copy()
                    model_scores_by_session = {}
                    for p, b in zip(p_te, b_te):
                        b = str(b)
                        model_scores_by_session.setdefault(b, []).append(float(p))
                    model_scores_by_session = {k: (sum(v) / len(v)) for k, v in model_scores_by_session.items()}
                    llm_votes_by_session = {k: int(v[0]) for k, v in dyn_pred.items()} if dyn_pred else {}
                    instance_explanations = explain_sessions_batch(
                        seqs_te_explain,
                        session_key_col=sess_col,
                        kb_rules=rules_for_explain,
                        rid2eid=rid2eid,
                        rule_to_cluster=rule_to_cluster,
                        model_scores_by_session=model_scores_by_session,
                        llm_votes_by_session=llm_votes_by_session,
                    )
                    if instance_explanations:
                        session_to_label = {str(k): int(ys_te[i]) for i, k in enumerate(keys_te)}
                        session_to_detected = {str(k): int(final_pred[i]) for i, k in enumerate(keys_te)}
                        session_to_event_ids = {}
                        for _, row in seqs_te_explain.iterrows():
                            sid = str(row[sess_col])
                            seq_rids = _to_row_ids(row.get("EventSeq"), eid2rid)
                            session_to_event_ids[sid] = [rid2eid.get(int(r), f"E{int(r)}") for r in seq_rids]
                        eid2tmpl = {}
                        if "EventId" in templates_df.columns and "EventTemplate" in templates_df.columns:
                            eid2tmpl = dict(zip(templates_df["EventId"].astype(str), templates_df["EventTemplate"].astype(str)))
                        for ex in instance_explanations:
                            sid = ex.get("session_id")
                            ex["true_label"] = session_to_label.get(sid)
                            ex["detected"] = session_to_detected.get(sid, 0)
                            event_ids = session_to_event_ids.get(sid, [])
                            ex["event_ids"] = event_ids
                            raw_preview = [eid2tmpl.get(e, e) for e in event_ids][:40]
                            ex["raw_session_preview"] = raw_preview
                            ex["raw_session_text"] = " [SEP] ".join(raw_preview) if raw_preview else ""
                        out_path = cfg.eval.get("instance_explanations_path", "instance_explanations.json")
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(instance_explanations, f, indent=2, ensure_ascii=False)
                        meta_path = out_path.replace(".json", "_meta.json")
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump({"run_tag": run_tag, "dataset": dataset}, f, indent=2)
                        print(f"[EXPLAIN] wrote {len(instance_explanations)} instance explanations → {out_path}")
                        if cfg.eval.get("user_study_enabled", False):
                            try:
                                from logsable.user_study import build_user_study_samples
                                us_out = cfg.eval.get("user_study_output_path", None)
                                n_norm = int(cfg.eval.get("user_study_n_normal", 10))
                                n_anom = int(cfg.eval.get("user_study_n_anom", 10))
                                build_user_study_samples(
                                    explanations_path=out_path,
                                    db_path="logdb.sqlite",
                                    out_path=us_out,
                                    n_normal=n_norm,
                                    n_anom=n_anom,
                                )
                            except Exception as e:
                                print(f"[user_study] WARNING: failed to build user-study samples: {e}")
                        sample = next((e for e in instance_explanations if e.get("triggered_rules") or e.get("cluster_reference")), instance_explanations[0] if instance_explanations else None)
                        if sample:
                            print(f"[EXPLAIN] sample: {json.dumps(sample, indent=2)}")
                except Exception as e:
                    print(f"[EXPLAIN] instance explanation failed: {e}")

    if _timing_ok and not _hybrid_online_recorded and len(ses_keys) > 0:
        if _t_hybrid_setup_start is not None and hybrid_setup_seconds == 0.0:
            hybrid_setup_seconds = time.perf_counter() - _t_hybrid_setup_start
        _locals = locals()
        _rules_o = _locals.get("kb_rules") or _locals.get("auto_rules") or []
        _kb_v = _locals.get("kb_vec")
        _eid2rid_f = _locals.get("eid2rid")
        hybrid_online_seconds, hybrid_ms_per_session = _time_hybrid_online_inference(
            seqs_for_rules, session_key_col, ses_keys, ypt, dyn_pred,
            _rules_o, _eid2rid_f, rid2eid, rid2tmpl, default_dyn_conf, kb_vec=_kb_v,
            low_conf_anom_max=_locals.get("dyn_low_conf_max"),
        )
        ypt_map = {str(k): int(ypt[i]) for i, k in enumerate(keys_te)}
        kb_map = (
            {str(k): int(_kb_v[i]) for i, k in enumerate(keys_te)}
            if _kb_v is not None and len(_kb_v) == len(keys_te)
            else None
        )
        hybrid_single_session_ms = _benchmark_single_session_hybrid(
            seqs_for_rules, session_key_col, str(ses_keys[0]), ypt_map,
            dyn_pred, _rules_o, _eid2rid_f, rid2eid, rid2tmpl, default_dyn_conf, kb_by_key=kb_map,
            low_conf_anom_max=_locals.get("dyn_low_conf_max"),
        )
        _hybrid_online_recorded = True
        print(
            f"[timing] hybrid online (test): {hybrid_online_seconds:.3f}s "
            f"({hybrid_ms_per_session:.3f} ms/session, single={hybrid_single_session_ms:.3f} ms)"
        )

    if _t_hybrid_setup_start is not None and hybrid_setup_seconds == 0.0:
        hybrid_setup_seconds = time.perf_counter() - _t_hybrid_setup_start
    if _timing_ok and hybrid_setup_seconds > 0:
        print(f"[timing] hybrid offline setup: {hybrid_setup_seconds:.3f}s")
    if _timing_ok:
        _append_run_timing_csv(
            cfg,
            train_seconds,
            baseline_inference_seconds,
            hybrid_setup_seconds,
            hybrid_online_seconds,
            hybrid_ms_per_session,
            hybrid_single_session_ms,
            dataset=dataset,
            model_name=cfg.model.get("name", "unknown"),
            n_test_sessions=len(ses_keys),
        )


if __name__ == "__main__":
    main()