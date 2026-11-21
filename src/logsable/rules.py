from logsable.common_imports import *


# === LLM cluster-rules ingestion helpers ===

def _normalize_llm_rule(r: dict) -> dict:
    """
    Keep only the supported primitives and fields we use downstream.
    Allowed keys inside if_all / if_any:
      - {"min_count": {"event_id": "E7", "count": 2}}
      - {"ordered_subset": [int row_ids ...]}
      - {"contains_ngram": [int row_ids ...]}
      - {"absent_within": {"event_id": "E5", "window": 10}}
    """
    out = {
        "name": str(r.get("name", "llm_rule")),
        "if_all": [],
        "if_any": [],
        "explanation": str(r.get("explanation", "")),
        "confidence": float(r.get("confidence", 0.0)),
    }

    def keep_clause(x):
        if not isinstance(x, dict): return None
        if "min_count" in x and isinstance(x["min_count"], dict):
            mc = x["min_count"]
            eid = str(mc.get("event_id", ""))
            cnt = int(mc.get("count", 1))
            return {"min_count": {"event_id": eid, "count": cnt}}
        if "ordered_subset" in x:
            seq = x["ordered_subset"]
            if isinstance(seq, (list, tuple)) and all(isinstance(i, int) for i in seq):
                return {"ordered_subset": list(map(int, seq))}
        if "contains_ngram" in x:
            seq = x["contains_ngram"]
            if isinstance(seq, (list, tuple)) and all(isinstance(i, int) for i in seq):
                return {"contains_ngram": list(map(int, seq))}
        if "absent_within" in x and isinstance(x["absent_within"], dict):
            aw = x["absent_within"]
            eid = str(aw.get("event_id", ""))
            win = int(aw.get("window", 1))
            return {"absent_within": {"event_id": eid, "window": win}}
        return None

    for k in ("if_all", "if_any"):
        arr = r.get(k, [])
        if isinstance(arr, list):
            for clause in arr:
                c = keep_clause(clause)
                if c is not None:
                    out[k].append(c)
    return out


def standardize_llm_rules_to_row_ids(llm_rules: list, eid2rid: dict) -> list:
    """
    Convert event-id based clauses to our internal row_id representation where needed.
    - min_count.event_id -> row_id via eid2rid
    - absent_within.event_id -> row_id via eid2rid
    ordered_subset / contains_ngram are expected as row_ids already; leave as-is.
    """
    std = []
    for r in llm_rules:
        rr = _normalize_llm_rule(r)

        def map_event_id_clause(clause):
            if "min_count" in clause:
                eid = str(clause["min_count"]["event_id"])
                if eid in eid2rid:
                    clause["min_count"]["row_id"] = int(eid2rid[eid])
                # keep original event_id for explainability
            if "absent_within" in clause:
                eid = str(clause["absent_within"]["event_id"])
                if eid in eid2rid:
                    clause["absent_within"]["row_id"] = int(eid2rid[eid])
            return clause

        rr["if_all"] = [map_event_id_clause(dict(c)) for c in rr["if_all"]]
        rr["if_any"] = [map_event_id_clause(dict(c)) for c in rr["if_any"]]
        std.append(rr)
    return std


def _eval_rule_clause(seq_row_ids, clause, rid2eid, rid2tmpl=None, seq_eids=None):
    """Return True if a single rule clause matches the session sequence."""
    if seq_eids is None:
        seq_eids = [rid2eid.get(int(r), f"E{int(r)}") for r in seq_row_ids]

    if "template_contains" in clause:
        if not rid2tmpl:
            return False
        sub = str(clause["template_contains"]).lower()
        return any(
            sub in str(rid2tmpl.get(int(r), "")).lower()
            for r in seq_row_ids
        )

    if "contains_ngram" in clause:
        ngram = clause["contains_ngram"]
        n = len(ngram)
        if n == 0:
            return False
        for i in range(len(seq_row_ids) - n + 1):
            if seq_row_ids[i:i + n] == ngram:
                return True
        return False

    if "ordered_subset" in clause:
        it = iter(seq_row_ids)
        return all(any(x == tok for x in it) for tok in clause["ordered_subset"])

    if "min_count" in clause:
        spec = clause["min_count"]
        need = int(spec.get("count", 1))
        if spec.get("row_id") is not None:
            rid = int(spec["row_id"])
            return sum(1 for r in seq_row_ids if int(r) == rid) >= need
        eid = str(spec.get("event_id", ""))
        return sum(1 for e in seq_eids if e == eid) >= need

    if "absent_within" in clause:
        aw = clause["absent_within"]
        row_id = aw.get("row_id")
        if row_id is None:
            eid = str(aw.get("event_id", ""))
            for r, e in zip(seq_row_ids, seq_eids):
                if e == eid:
                    row_id = int(r)
                    break
        if row_id is None:
            return False
        # Event must not appear anywhere in the session (LLM cluster-rule semantics).
        return int(row_id) not in [int(r) for r in seq_row_ids]

    return False


def rules_vector_for_keys(seqs_df, rules, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=None):
    """Session-level 0/1 vector: 1 if any rule fires on that session."""
    key_set = {str(k) for k in keys}
    preds = {}
    for _, row in seqs_df.iterrows():
        sid = str(row[session_key_col])
        if sid not in key_set:
            continue
        seq_rids = to_row_ids_fn(row["EventSeq"])
        fired = any(
            rule_fires_on_seq(seq_rids, r, rid2eid, rid2tmpl=rid2tmpl) for r in rules
        )
        preds[sid] = 1 if fired else 0
    return np.array([preds.get(str(k), 0) for k in keys], dtype=int)


def cluster_rule_filter_cfg(cfg_eval: dict | None, dataset: str | None = None) -> dict:
    """Validation filter settings for LLM cluster rules."""
    cfg = dict(cfg_eval or {})
    if not cfg.get("cluster_rule_filter_enabled", True):
        cfg["rule_filter_enabled"] = False
        return cfg
    cfg["rule_filter_enabled"] = True
    ds = str(dataset or "").strip().upper()
    if ds == "LIBERTY":
        cfg["rule_max_val_fire_rate"] = float(cfg.get("liberty_cluster_rule_max_val_fire_rate", 0.50))
        cfg["rule_min_val_f1"] = float(cfg.get("liberty_cluster_rule_min_val_f1", 0.05))
        cfg["rule_min_val_precision"] = float(cfg.get("liberty_cluster_rule_min_val_precision", 0.10))
        cfg["rule_max_single_fire_rate"] = float(cfg.get("liberty_cluster_rule_max_single_fire_rate", 0.25))
        cfg["rule_filter_max_per_rule"] = int(cfg.get("liberty_cluster_rule_filter_max_per_rule", 600))
    else:
        cfg["rule_max_val_fire_rate"] = float(cfg.get("cluster_rule_max_val_fire_rate", 0.35))
        cfg["rule_min_val_f1"] = float(cfg.get("cluster_rule_min_val_f1", 0.30))
        cfg["rule_min_val_precision"] = float(cfg.get("cluster_rule_min_val_precision", 0.35))
        cfg["rule_max_single_fire_rate"] = float(cfg.get("cluster_rule_max_single_fire_rate", 0.15))
        cfg["rule_filter_max_per_rule"] = int(cfg.get("cluster_rule_filter_max_per_rule", 600))
    return cfg


def liberty_cluster_rules_use_event_ids(rules, rid2eid):
    """
    Rewrite cluster rules as EventId min_count rules.
    """
    if not rules or not rid2eid:
        return list(rules or [])
    out = []
    seen = set()
    for r in rules:
        has_event_rule = any(
            "min_count" in c and c["min_count"].get("event_id")
            for c in (r.get("if_all") or []) + (r.get("if_any") or [])
        )
        if has_event_rule or any(
            "template_contains" in c for c in (r.get("if_all") or []) + (r.get("if_any") or [])
        ):
            out.append(r)
            continue
        eids = []
        for c in (r.get("if_any") or []) + (r.get("if_all") or []):
            if "contains_ngram" in c:
                for x in c["contains_ngram"]:
                    e = rid2eid.get(int(x))
                    if e:
                        eids.append(str(e))
            elif "ordered_subset" in c:
                for x in c["ordered_subset"]:
                    e = rid2eid.get(int(x))
                    if e:
                        eids.append(str(e))
        if not eids:
            continue
        uniq = list(dict.fromkeys(eids))
        if len(uniq) == 1:
            key = uniq[0]
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "name": f"{r.get('name', 'lib_cluster')}_{key}",
                "if_any": [],
                "if_all": [{"min_count": {"event_id": key, "count": 1}}],
                "explanation": str(r.get("explanation", "")),
                "confidence": float(r.get("confidence", 0.8)),
                "cluster_id": r.get("cluster_id"),
            })
        else:
            key = tuple(uniq)
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "name": f"{r.get('name', 'lib_cluster')}_ng{'_'.join(uniq[:3])}",
                "if_any": [],
                "if_all": [{"min_count": {"event_id": e, "count": 1}} for e in uniq],
                "explanation": str(r.get("explanation", "")),
                "confidence": float(r.get("confidence", 0.8)),
                "cluster_id": r.get("cluster_id"),
            })
    return out


def prepare_llm_cluster_rules(rules, mode=None):
    mode = (mode or "if_any_only").strip().lower()
    if mode == "full":
        return list(rules)
    out = []
    seen = set()
    for r in rules:
        clauses = []
        for c in (r.get("if_any") or []) + (r.get("if_all") or []):
            if "contains_ngram" in c:
                key = ("ng", tuple(c["contains_ngram"]))
            elif "ordered_subset" in c:
                key = ("os", tuple(c["ordered_subset"]))
            else:
                continue
            if key in seen:
                continue
            seen.add(key)
            clauses.append(
                {"contains_ngram": list(c["contains_ngram"])}
                if "contains_ngram" in c
                else {"ordered_subset": list(c["ordered_subset"])}
            )
        if not clauses:
            continue
        out.append({
            "name": str(r.get("name", "llm_cluster_ngram")),
            "if_any": clauses,
            "if_all": [],
            "explanation": str(r.get("explanation", "")),
            "confidence": float(r.get("confidence", 0.0)),
            "cluster_score_if_any_only": True,
            "source_llm_rule": str(r.get("name", "")),
        })
    return out


def mine_cluster_repr_rules_from_bundle(
    bundle,
    seqs_df,
    session_key_col,
    to_row_ids_fn,
    eval_cfg=None,
    dataset=None,
    rid2eid=None,
):
    """Mine bigram/trigram rules from HDBSCAN cluster representatives (train anomalies)."""
    from collections import Counter

    eval_cfg = eval_cfg or {}
    if not bundle or not len(seqs_df):
        return []
    ds = str(dataset or "").strip().upper()
    liberty_event_rules = ds == "LIBERTY" and rid2eid
    max_bi = int(eval_cfg.get("cluster_repr_max_bigrams", 8))
    max_tri = int(eval_cfg.get("cluster_repr_max_trigrams", 5))
    min_support = int(eval_cfg.get("cluster_repr_min_support", 1))
    events = seqs_df["EventSeq"].tolist()
    sess_keys = seqs_df[session_key_col].astype(str).tolist()
    key_to_ev = {sk: events[i] for i, sk in enumerate(sess_keys)}
    rules = []
    seen_ng = set()

    def _add_rule(cid, ng, cnt, n):
        key = (n, tuple(ng))
        if key in seen_ng:
            return
        seen_ng.add(key)
        if liberty_event_rules:
            eids = [rid2eid.get(int(r)) for r in ng if rid2eid.get(int(r))]
            if len(eids) != n:
                return
            if n == 1:
                rules.append({
                    "name": f"cluster_{cid}_e_{eids[0]}",
                    "if_any": [],
                    "if_all": [{"min_count": {"event_id": eids[0], "count": 1}}],
                    "explanation": f"Cluster {cid} representative event {eids[0]} (support={cnt}).",
                    "confidence": 0.8,
                    "cluster_id": str(cid),
                })
            else:
                rules.append({
                    "name": f"cluster_{cid}_ng{n}_{'_'.join(eids)}",
                    "if_any": [],
                    "if_all": [{"min_count": {"event_id": e, "count": 1}} for e in eids],
                    "explanation": f"Cluster {cid} representative {n}-gram {eids} (support={cnt}).",
                    "confidence": 0.8,
                    "cluster_id": str(cid),
                })
        else:
            rules.append({
                "name": f"cluster_{cid}_ng{n}_{'_'.join(map(str, ng))}",
                "if_any": [{"contains_ngram": list(map(int, ng))}],
                "if_all": [],
                "explanation": f"Cluster {cid} representative {n}-gram (support={cnt}).",
                "confidence": 0.8,
                "cluster_id": str(cid),
                "cluster_score_if_any_only": True,
            })
        if liberty_event_rules and n > 1:
            for r in ng:
                e = rid2eid.get(int(r))
                if not e:
                    continue
                ukey = ("u", e)
                if ukey in seen_ng:
                    continue
                seen_ng.add(ukey)
                rules.append({
                    "name": f"cluster_{cid}_e_{e}",
                    "if_any": [],
                    "if_all": [{"min_count": {"event_id": e, "count": 1}}],
                    "explanation": f"Cluster {cid} event {e} (from {n}-gram, support={cnt}).",
                    "confidence": 0.75,
                    "cluster_id": str(cid),
                })

    for cid, cl in bundle.get("clusters", {}).items():
        seqs_ids = []
        reps = cl.get("representatives") or {}
        for rk in ("center", "end_a", "end_b"):
            info = reps.get(rk) or {}
            if "row" in info:
                idx = int(info["row"])
                if 0 <= idx < len(events):
                    seqs_ids.append(to_row_ids_fn(events[idx]))
            elif info.get("seq_key") is not None:
                ev = key_to_ev.get(str(info["seq_key"]))
                if ev is not None:
                    seqs_ids.append(to_row_ids_fn(ev))
        for ex in (cl.get("examples") or [])[:15]:
            ev = ex.get("event_ids")
            if ev is None and ex.get("seq_key") is not None:
                ev = key_to_ev.get(str(ex["seq_key"]))
            if ev is not None:
                seqs_ids.append(to_row_ids_fn(ev))
        if not seqs_ids:
            continue
        bi, tri = Counter(), Counter()
        for s in seqs_ids:
            if len(s) < 2:
                continue
            for i in range(len(s) - 1):
                bi[tuple(s[i:i + 2])] += 1
            for i in range(len(s) - 2):
                tri[tuple(s[i:i + 3])] += 1
        for ng, cnt in bi.most_common(max_bi):
            if cnt >= min_support:
                _add_rule(cid, ng, cnt, 2)
        for ng, cnt in tri.most_common(max_tri):
            if cnt >= min_support:
                _add_rule(cid, ng, cnt, 3)
    return rules


def filter_cluster_rules_on_validation(
    rules,
    seqs_df,
    keys,
    ys,
    session_key_col,
    to_row_ids_fn,
    rid2eid,
    rid2tmpl=None,
    eval_cfg=None,
    dataset=None,
):
    """Validation filter for cluster KB rules."""
    from logsable.train import anomaly_f1

    eval_cfg = eval_cfg or {}
    if not rules or len(keys) == 0:
        return list(rules or [])
    ys = np.asarray(ys, dtype=int)
    ds = str(dataset or "").strip().upper()
    if ds == "LIBERTY":
        min_target = float(eval_cfg.get("liberty_cluster_rule_min_val_f1_target", 0.05))
        max_keep = int(eval_cfg.get("liberty_cluster_rule_fallback_max", 120))
    else:
        min_target = float(eval_cfg.get("cluster_rule_min_val_f1_target", 0.25))
        max_keep = int(eval_cfg.get("cluster_rule_fallback_max", 120))

    def _val_f1(rule_list):
        if not rule_list:
            return -1.0
        vec = rules_vector_for_keys(
            seqs_df, rule_list, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        _, _, f1, _ = anomaly_f1(ys, vec)
        return float(f1)

    kept = filter_rules_on_validation(
        rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl,
        cfg_eval=cluster_rule_filter_cfg(eval_cfg, dataset=dataset), dataset=dataset,
    )
    if kept and _val_f1(kept) >= min_target:
        return _guard_cluster_rules_not_too_broad(
            kept, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl, eval_cfg,
            dataset=dataset,
        )

    if ds == "LIBERTY":
        max_fr = float(eval_cfg.get("liberty_cluster_rule_max_val_fire_rate", 0.50))
        max_norm = float(eval_cfg.get("liberty_cluster_rule_max_norm_fire_rate", 0.25))
        min_p_greedy = float(eval_cfg.get("liberty_cluster_rule_min_val_precision", 0.10)) * 0.5
    else:
        max_fr = float(eval_cfg.get("cluster_rule_max_val_fire_rate", 0.35))
        max_norm = float(eval_cfg.get("cluster_rule_max_norm_fire_rate", 0.12))
        min_p_greedy = float(eval_cfg.get("cluster_rule_min_val_precision", 0.35)) * 0.5
    greedy = _greedy_select_rules_on_val(
        rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl,
        max_fire_rate=max_fr,
        min_f1=min_target * 0.5,
        max_check=int(eval_cfg.get("cluster_rule_filter_max_per_rule", 600)),
        max_norm_fire_rate=max_norm,
        min_precision=min_p_greedy,
    )
    if greedy and _val_f1(greedy) >= _val_f1(kept):
        print(
            f"[cluster] validation filter: kept {len(greedy)}/{len(rules)} rules "
            f"(F1={_val_f1(greedy):.3f})"
        )
        return _guard_cluster_rules_not_too_broad(
            greedy, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl, eval_cfg,
            dataset=dataset,
        )
    if kept:
        return _guard_cluster_rules_not_too_broad(
            kept, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl, eval_cfg,
            dataset=dataset,
        )

    scored = []
    max_check = int(eval_cfg.get("cluster_rule_filter_max_per_rule", 600))
    for rule in rules[:max_check]:
        vec = rules_vector_for_keys(
            seqs_df, [rule], keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        anom_hit = int(vec[ys == 1].sum())
        if anom_hit <= 0:
            continue
        p, _, f1, _ = anomaly_f1(ys, vec)
        scored.append((anom_hit, float(f1), float(p), rule))
    scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    fallback = [r for _, _, _, r in scored[:max_keep]]
    if fallback:
        print(
            f"[cluster] validation filter: kept {len(fallback)}/{len(rules)} rules "
            f"(F1={_val_f1(fallback):.3f})"
        )
    return _guard_cluster_rules_not_too_broad(
        fallback, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl, eval_cfg,
        dataset=dataset,
    )


def _cluster_rule_val_stats(rule_list, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl):
    """Return (f1, precision, recall, norm_fire_rate) for a rule list on VAL."""
    from logsable.train import anomaly_f1

    if not rule_list:
        return 0.0, 0.0, 0.0, 0.0
    vec = rules_vector_for_keys(
        seqs_df, rule_list, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
    )
    ys = np.asarray(ys, dtype=int)
    p, r, f1, _ = anomaly_f1(ys, vec)
    norm = ys == 0
    norm_fire = float(vec[norm].mean()) if norm.sum() > 0 else float(vec.mean())
    return float(f1), float(p), float(r), norm_fire


def _guard_cluster_rules_not_too_broad(
    rules,
    seqs_df,
    keys,
    ys,
    session_key_col,
    to_row_ids_fn,
    rid2eid,
    rid2tmpl,
    eval_cfg,
    dataset=None,
):
    """Apply validation precision and normal-session fire-rate limits to cluster rules."""
    from logsable.train import anomaly_f1

    eval_cfg = eval_cfg or {}
    if not rules or len(keys) == 0:
        return list(rules or [])
    ds = str(dataset or "").strip().upper()
    if ds == "LIBERTY":
        max_norm_fire = float(eval_cfg.get("liberty_cluster_rule_max_norm_fire_rate", 0.25))
        min_val_f1 = float(eval_cfg.get("liberty_cluster_rule_min_val_f1_target", 0.05))
        min_precision = float(eval_cfg.get("liberty_cluster_rule_min_val_precision", 0.10))
    else:
        max_norm_fire = float(eval_cfg.get("cluster_rule_max_norm_fire_rate", 0.12))
        min_val_f1 = float(eval_cfg.get("cluster_rule_min_val_f1_target", 0.25))
        min_precision = float(eval_cfg.get("cluster_rule_min_val_precision", 0.35))
    f1, p, r, norm_fire = _cluster_rule_val_stats(
        rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl
    )
    if f1 >= min_val_f1 and p >= min_precision and norm_fire <= max_norm_fire:
        return rules
    print(
        f"[cluster] validation filter: refining rules "
        f"(F1={f1:.3f}, P={p:.3f}, R={r:.3f}, norm_fire={norm_fire:.3f})"
    )
    greedy = _greedy_select_rules_on_val(
        rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl,
        max_fire_rate=float(
            eval_cfg.get("liberty_cluster_rule_max_val_fire_rate", 0.50)
            if ds == "LIBERTY"
            else eval_cfg.get("cluster_rule_max_val_fire_rate", 0.35)
        ),
        min_f1=min_val_f1,
        max_check=len(rules),
        max_norm_fire_rate=max_norm_fire,
        min_precision=min_precision,
    )
    if greedy:
        f1g, pg, _, nfg = _cluster_rule_val_stats(
            greedy, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl
        )
        if f1g >= min_val_f1 * 0.5 and pg >= min_precision * 0.5 and nfg <= max_norm_fire * 1.5:
            print(f"[cluster] validation filter: kept {len(greedy)} rules (F1={f1g:.3f})")
            return greedy
    if ds == "LIBERTY" and rules:
        scored = []
        ys_a = np.asarray(ys, dtype=int)
        for rule in rules:
            vec = rules_vector_for_keys(
                seqs_df, [rule], keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
            )
            anom_hit = int(vec[ys_a == 1].sum())
            if anom_hit <= 0:
                continue
            p, _, f1, _ = anomaly_f1(ys_a, vec)
            scored.append((anom_hit, float(f1), float(p), rule))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        fallback = [r for _, _, _, r in scored[: int(eval_cfg.get("liberty_cluster_rule_fallback_max", 120))]]
        if fallback:
            print(f"[cluster] validation filter: kept {len(fallback)} anomaly-matching rules")
            return fallback
    print("[cluster] validation filter: no cluster rules passed validation limits")
    return []


def build_cluster_kb_rules(
    bundle,
    llm_rules_std,
    seqs_cluster_df,
    session_key_col,
    to_row_ids_fn,
    eval_cfg=None,
    dataset=None,
    rid2eid=None,
):
    """Combine representative n-gram rules with LLM-derived cluster rules."""
    eval_cfg = eval_cfg or {}
    ds = str(dataset or "").strip().upper()
    repr_rules = mine_cluster_repr_rules_from_bundle(
        bundle, seqs_cluster_df, session_key_col, to_row_ids_fn,
        eval_cfg=eval_cfg, dataset=dataset, rid2eid=rid2eid,
    )
    mode = str(eval_cfg.get("cluster_llm_rule_mode", "if_any_only"))
    llm_rules = prepare_llm_cluster_rules(llm_rules_std or [], mode=mode)
    merge = str(eval_cfg.get("cluster_rule_merge", "repr_first")).strip().lower()
    if merge == "llm_first":
        combined = llm_rules + repr_rules
    else:
        combined = repr_rules + llm_rules
    if ds == "LIBERTY" and rid2eid:
        combined = liberty_cluster_rules_use_event_ids(combined, rid2eid)
    # dedupe by rule key
    seen, out = set(), []
    for r in combined:
        key = None
        for c in r.get("if_any") or []:
            if "contains_ngram" in c:
                key = ("ng", tuple(c["contains_ngram"]))
                break
            if "ordered_subset" in c:
                key = ("os", tuple(c["ordered_subset"]))
                break
        if key is None:
            for c in r.get("if_all") or []:
                if "min_count" in c and c["min_count"].get("event_id"):
                    key = ("e", str(c["min_count"]["event_id"]))
                    break
                if "template_contains" in c:
                    key = ("t", str(c["template_contains"]))
                    break
        if key is not None and key in seen:
            continue
        if key is not None:
            seen.add(key)
        out.append(r)
    return out, len(repr_rules), len(llm_rules)


def rule_filter_cfg_for_dataset(dataset: str, cfg_eval: dict | None) -> dict:
    """Merge global rule-filter settings with dataset overrides."""
    cfg = dict(cfg_eval or {})
    ds = str(dataset).strip().upper()
    if ds == "BGL":
        cfg["rule_max_val_fire_rate"] = float(cfg.get("bgl_rule_max_val_fire_rate", 0.70))
        cfg["rule_min_val_f1"] = float(cfg.get("bgl_rule_min_val_f1", 0.90))
        cfg["rule_min_val_precision"] = float(cfg.get("bgl_rule_min_val_precision", 0.92))
        cfg["rule_max_single_fire_rate"] = float(cfg.get("bgl_rule_max_single_fire_rate", 0.15))
        cfg["rule_filter_max_per_rule"] = int(cfg.get("bgl_rule_filter_max_per_rule", 600))
        cfg["rule_max_norm_fire_rate"] = float(cfg.get("bgl_rule_max_norm_fire_rate", 0.04))
    elif ds == "HDFS":
        cfg["rule_max_val_fire_rate"] = float(cfg.get("hdfs_rule_max_val_fire_rate", 0.25))
        cfg["rule_min_val_precision"] = float(cfg.get("hdfs_rule_min_val_precision", 0.85))
    return cfg


def _greedy_select_rules_on_val(
    rules,
    seqs_df,
    keys,
    ys,
    session_key_col,
    to_row_ids_fn,
    rid2eid,
    rid2tmpl,
    max_fire_rate,
    min_f1,
    max_check,
    max_norm_fire_rate=None,
    min_precision=0.0,
):
    """Greedy union of rules ranked by validation precision, with fire-rate caps."""
    from logsable.train import anomaly_f1

    ys = np.asarray(ys, dtype=int)
    norm = ys == 0
    scored = []
    to_check = rules[:max_check] if len(rules) > max_check else rules
    for rule in to_check:
        v = rules_vector_for_keys(
            seqs_df, [rule], keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        if int(v.sum()) == 0:
            continue
        pp, _, ff, _ = anomaly_f1(ys, v)
        if float(pp) < float(min_precision) and int(v[ys == 1].sum()) == 0:
            continue
        scored.append((float(pp), float(ff), rule, v))
    if not scored:
        return []
    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)

    selected = []
    vec = np.zeros(len(keys), dtype=int)
    best_f1 = -1.0
    for pp, ff, rule, v in scored:
        new_vec = np.maximum(vec, v)
        fr = float(new_vec.mean())
        if fr > float(max_fire_rate):
            continue
        if max_norm_fire_rate is not None and norm.sum() > 0:
            if float(new_vec[norm].mean()) > float(max_norm_fire_rate):
                continue
        _, _, f1, _ = anomaly_f1(ys, new_vec)
        if f1 > best_f1 + 1e-9 or (not selected and f1 >= float(min_f1) * 0.5):
            vec = new_vec
            selected.append(rule)
            best_f1 = float(f1)
    if selected and best_f1 >= float(min_f1):
        return selected
    if scored:
        return [scored[0][2]]
    return []


def filter_rules_on_validation(
    rules,
    seqs_df,
    keys,
    ys,
    session_key_col,
    to_row_ids_fn,
    rid2eid,
    rid2tmpl=None,
    cfg_eval=None,
    dataset=None,
):
    """Filter mined rules using validation precision and fire-rate limits."""
    from logsable.train import anomaly_f1

    cfg_eval = rule_filter_cfg_for_dataset(dataset or "", cfg_eval or {})
    if not cfg_eval.get("rule_filter_enabled", True):
        print(f"[rules] validation filter: disabled — keeping all {len(rules)} rules")
        return rules
    if not rules or len(keys) == 0:
        return []

    ds = str(dataset or "").strip().upper()
    max_check = int(cfg_eval.get("rule_filter_max_per_rule", 150))
    max_norm_fr = cfg_eval.get("rule_max_norm_fire_rate")
    min_f1 = float(cfg_eval.get("rule_min_val_f1", 0.80))
    max_fr = float(cfg_eval.get("rule_max_val_fire_rate", 0.25))

    if ds == "BGL":
        selected = _greedy_select_rules_on_val(
            rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl,
            max_fire_rate=max_fr,
            min_f1=min_f1,
            max_check=max_check,
            max_norm_fire_rate=max_norm_fr,
            min_precision=float(cfg_eval.get("rule_min_val_precision", 0.85)),
        )
        if selected:
            vec_g = rules_vector_for_keys(
                seqs_df, selected, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
            )
            p_g, _, f1_g, _ = anomaly_f1(ys, vec_g)
            fr_g = float(vec_g.mean())
            print(
                f"[rules] validation filter: kept {len(selected)}/{len(rules)} rules "
                f"(F1={f1_g:.3f}, P={p_g:.3f}, fire_rate={fr_g:.3f})"
            )
            return selected

    ys = np.asarray(ys, dtype=int)
    vec = rules_vector_for_keys(
        seqs_df, rules, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
    )
    max_fr = float(cfg_eval.get("rule_max_val_fire_rate", 0.25))
    min_f1 = float(cfg_eval.get("rule_min_val_f1", 0.80))
    min_p = float(cfg_eval.get("rule_min_val_precision", 0.85))
    fr = float(vec.mean()) if len(vec) else 0.0
    p, _, f1, _ = anomaly_f1(ys, vec)
    if fr <= max_fr and f1 >= min_f1 and p >= min_p:
        print(
            f"[rules] validation filter: keep all {len(rules)} rules "
            f"(F1={f1:.3f}, P={p:.3f}, fire_rate={fr:.3f})"
        )
        return rules

    max_single_fr = float(cfg_eval.get("rule_max_single_fire_rate", 0.08))
    max_check = int(cfg_eval.get("rule_filter_max_per_rule", 150))
    to_check = rules[:max_check] if len(rules) > max_check else rules
    good = []
    for rule in to_check:
        v = rules_vector_for_keys(
            seqs_df, [rule], keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        if int(v.sum()) == 0:
            continue
        pp, _, ff, _ = anomaly_f1(ys, v)
        fr_r = float(v.mean())
        if fr_r <= max_single_fr and (pp >= 0.9 or (ff >= 0.5 and pp >= 0.5)):
            good.append(rule)

    if good:
        vec2 = rules_vector_for_keys(
            seqs_df, good, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        p2, _, f2, _ = anomaly_f1(ys, vec2)
        fr2 = float(vec2.mean())
        if fr2 <= max_fr and f2 >= min_f1 and p2 >= min_p:
            print(
                f"[rules] validation filter: kept {len(good)}/{len(rules)} rules "
                f"(F1={f2:.3f}, P={p2:.3f}, fire_rate={fr2:.3f})"
            )
            return good

    greedy = _greedy_select_rules_on_val(
        rules, seqs_df, keys, ys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl,
        max_fire_rate=max_fr, min_f1=min_f1, max_check=max_check,
        max_norm_fire_rate=max_norm_fr,
        min_precision=float(cfg_eval.get("rule_min_val_precision", 0.85)) * 0.5,
    )
    if greedy:
        vec3 = rules_vector_for_keys(
            seqs_df, greedy, keys, session_key_col, to_row_ids_fn, rid2eid, rid2tmpl=rid2tmpl
        )
        p3, _, f3, _ = anomaly_f1(ys, vec3)
        fr3 = float(vec3.mean())
        print(
            f"[rules] validation filter: kept {len(greedy)}/{len(rules)} rules "
            f"(F1={f3:.3f}, P={p3:.3f}, fire_rate={fr3:.3f})"
        )
        return greedy

    print(
        f"[rules] validation filter: no usable rules after prune "
        f"(combined F1={f1:.3f}, P={p:.3f}, fire_rate={fr:.3f})"
    )
    return []


def rule_fires_on_seq(seq_row_ids, rule, rid2eid, rid2tmpl=None):
    seq_eids = [rid2eid.get(int(r), f"E{int(r)}") for r in seq_row_ids]
    if_any = rule.get("if_any") or []
    if_all = rule.get("if_all") or []
    if rule.get("cluster_score_if_any_only"):
        if not if_any:
            return False
        return any(
            _eval_rule_clause(seq_row_ids, c, rid2eid, rid2tmpl=rid2tmpl, seq_eids=seq_eids)
            for c in if_any
        )
    any_ok = True if not if_any else any(
        _eval_rule_clause(seq_row_ids, c, rid2eid, rid2tmpl=rid2tmpl, seq_eids=seq_eids)
        for c in if_any
    )
    all_ok = True if not if_all else all(
        _eval_rule_clause(seq_row_ids, c, rid2eid, rid2tmpl=rid2tmpl, seq_eids=seq_eids)
        for c in if_all
    )
    return any_ok and all_ok


LIBERTY_KEYWORD_SEEDS = (
    "bad file descriptor",
    "cannot tm_reply",
    "connection refused",
    "connection timed out",
    "segmentation fault",
    "kernel panic",
    "out of memory",
    "oom-killer",
    "failed",
    "failure",
    "fatal",
    "panic",
    "abort",
    "error",
)


def _liberty_session_template_text(seq_row_ids, rid2tmpl):
    return " ".join(str(rid2tmpl.get(int(r), "")).lower() for r in seq_row_ids)


def mine_liberty_template_keyword_rules(
    seqs_train,
    rid2tmpl,
    session_key_col="seq_key",
    min_anom_sessions=3,
    max_norm_sessions=0,
):
    """
    Liberty: mine session-level rules on template text phrases.
    """
    rules = []
    for phrase in LIBERTY_KEYWORD_SEEDS:
        anom_hits = norm_hits = 0
        for _, row in seqs_train.iterrows():
            ids = row.get("EventSeq") or row.get("EventSeq_masked") or []
            if isinstance(ids, str):
                try:
                    import json
                    ids = json.loads(ids) if ids.strip().startswith("[") else ids.split(",")
                except Exception:
                    ids = []
            text = _liberty_session_template_text(list(ids), rid2tmpl)
            if phrase not in text:
                continue
            if int(row.get("Label", 0)) == 1:
                anom_hits += 1
            else:
                norm_hits += 1
        if anom_hits >= min_anom_sessions and norm_hits <= max_norm_sessions:
            safe = phrase.replace(" ", "_")[:48]
            rules.append({
                "name": f"liberty_kw_{safe}",
                "if_any": [],
                "if_all": [{"template_contains": phrase}],
                "explanation": (
                    f"Liberty template phrase '{phrase}' "
                    f"(train anom_sessions={anom_hits}, norm_sessions={norm_hits})."
                ),
                "confidence": 0.9 if norm_hits == 0 else 0.75,
            })
    return rules


