"""
Explainability artifacts: cluster-level (stored in KB) and instance-level (at inference).

Cluster-level schema (stored in KB):
  cluster_id, representative_sequences (list of event_id lists), llm_explanation, derived_rules

Instance-level schema (produced at inference):
  session_id, model_score, llm_vote, triggered_rules, triggered_ngrams, cluster_reference
"""
from __future__ import annotations

import json
import os
import re
from typing import Any

from logsable.common_imports import *


# ---------------------------------------------------------------------------
# Cluster-level artifact schema (stored in KB)
# ---------------------------------------------------------------------------

def build_cluster_artifact(
    cluster_id: str,
    representative_sequences: list[list[str]],
    llm_explanation: str,
    derived_rules: list[dict],
) -> dict:
    """
    Build one cluster-level explainability artifact (JSON-serializable).
    derived_rules: list of {"rule_id": str, "pattern": list[str] (event_ids), "description": str}
    """
    return {
        "cluster_id": str(cluster_id),
        "representative_sequences": representative_sequences,
        "llm_explanation": llm_explanation,
        "derived_rules": derived_rules,
    }


def _rule_to_pattern_event_ids(rule: dict, rid2eid: dict) -> list[str]:
    """Extract ordered list of event_ids mentioned in a rule (for pattern)."""

    def to_eid(x):
        if isinstance(x, str) and (x.startswith("E") or x.startswith("e")):
            return str(x)
        try:
            return rid2eid.get(int(x), f"E{int(x)}")
        except (TypeError, ValueError):
            return str(x)

    eids = []
    seen = set()
    for clause in rule.get("if_all", []) + rule.get("if_any", []):
        if "min_count" in clause:
            eid = str(clause["min_count"].get("event_id", ""))
            if eid and eid not in seen:
                eids.append(eid)
                seen.add(eid)
        if "ordered_subset" in clause:
            for rid in clause["ordered_subset"]:
                eid = to_eid(rid)
                if eid and eid not in seen:
                    eids.append(eid)
                    seen.add(eid)
        if "contains_ngram" in clause:
            for rid in clause["contains_ngram"]:
                eid = to_eid(rid)
                if eid and eid not in seen:
                    eids.append(eid)
                    seen.add(eid)
    return eids


def load_llm_cluster_results_by_cluster(json_path: str = "llm_results_clusters.json") -> list[dict]:
    """
    Load llm_results_clusters.json and return list of {cluster_id, response} per cluster.
    Handles: list of {cluster_id, response: {rules, ...}} or dict cluster_id -> response.
    """
    if not os.path.exists(json_path):
        return []
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    out = []
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "cluster_id" in item:
                out.append({"cluster_id": str(item["cluster_id"]), "response": item.get("response", {})})
            elif isinstance(item, dict) and "response" in item:
                out.append({"cluster_id": str(item.get("cluster_id", "")), "response": item["response"]})
    elif isinstance(data, dict):
        for cid, resp in data.items():
            out.append({"cluster_id": str(cid), "response": resp if isinstance(resp, dict) else {}})
    return out


def build_and_store_cluster_artifacts(
    bundle: dict,
    llm_results_path: str | None = "llm_results_clusters.json",
    cluster_results: list[dict] | None = None,
    rid2eid: dict | None = None,
    run_tag: str = "",
    dataset: str = "HDFS",
    db_path: str = "logdb.sqlite",
) -> list[dict]:
    """
    Build cluster-level artifacts from bundle + LLM cluster results, store in KB, and
    populate rule_id -> cluster_id mapping. Returns list of stored artifacts.

    cluster_results: optional list of {cluster_id, response} from run_llm_clusters_with_cache.
    If provided, llm_results_path is ignored.
    """
    from logsable.kb import (
        ensure_explainability_tables,
        store_cluster_artifact,
        store_rule_to_cluster_mapping,
    )
    ensure_explainability_tables(db_path)

    # template_map: [{"row_id", "event_id", "event_template"}, ...]
    template_map = bundle.get("template_map", [])
    rid_to_eid = {int(m["row_id"]): str(m["event_id"]) for m in template_map}
    if rid2eid is None:
        rid2eid = rid_to_eid

    clusters_bundle = bundle.get("clusters", {})
    if cluster_results is not None:
        cluster_results_list = cluster_results
    else:
        cluster_results_list = load_llm_cluster_results_by_cluster(llm_results_path or "")
    stored = []

    for cr in cluster_results_list:
        cid = cr["cluster_id"]
        response = cr.get("response", {})
        rules = response.get("rules", [])
        # Cluster-level explanation from response summary or rule text
        llm_explanation = response.get("cluster_summary") or response.get("summary") or ""
        if not llm_explanation and rules:
            parts = [r.get("explanation", "") for r in rules[:3] if r.get("explanation")]
            llm_explanation = "; ".join(parts) if parts else ""

        # Representative sequences: from bundle examples (event_ids in bundle are actually row_ids)
        rep_seqs = []
        if cid in clusters_bundle:
            for ex in clusters_bundle[cid].get("examples", [])[:10]:
                raw = ex.get("event_ids", [])
                seq_eids = [rid_to_eid.get(int(r), rid2eid.get(int(r), f"E{int(r)}")) for r in raw]
                rep_seqs.append(seq_eids)
        if not rep_seqs and cid in clusters_bundle:
            # use bundle examples when center/extreme sequences are unavailable
            for ex in clusters_bundle[cid].get("examples", [])[:3]:
                raw = ex.get("event_ids", [])
                seq_eids = [rid_to_eid.get(int(r), rid2eid.get(int(r), f"E{int(r)}")) for r in raw]
                rep_seqs.append(seq_eids)

        derived_rules = []
        for r in rules:
            rule_id = r.get("name", f"cluster_{cid}_rule")
            pattern = _rule_to_pattern_event_ids(r, rid2eid)
            if not pattern and "min_count" in str(r):
                # min_count-only rule: collect event_ids from min_count
                for clause in r.get("if_any", []) + r.get("if_all", []):
                    if "min_count" in clause:
                        eid = str(clause["min_count"].get("event_id", ""))
                        if eid:
                            pattern.append(eid)
            description = r.get("explanation", "")
            derived_rules.append({
                "rule_id": rule_id,
                "pattern": pattern,
                "description": description,
            })
            store_rule_to_cluster_mapping(rule_id, cid, run_tag, dataset, db_path)

        artifact = build_cluster_artifact(
            cluster_id=cid,
            representative_sequences=rep_seqs,
            llm_explanation=llm_explanation,
            derived_rules=derived_rules,
        )
        store_cluster_artifact(run_tag, dataset, cid, artifact, db_path)
        stored.append(artifact)
    if stored:
        print(f"[EXPLAIN] stored {len(stored)} cluster artifacts (run_tag={run_tag!r}, dataset={dataset})")
    return stored


# ---------------------------------------------------------------------------
# Instance-level explanation (at inference)
# ---------------------------------------------------------------------------

# Match leading "c<digits>_" or "cluster_<digits>_" to strip for deduplication
_CLUSTER_PREFIX_RE = re.compile(r"^(?:c\d+_|cluster_\d+_)", re.IGNORECASE)


def _strip_cluster_prefix(rule_name: str) -> str:
    """Remove leading c0_, c12_, cluster_0_, etc. for pattern-based deduplication."""
    return _CLUSTER_PREFIX_RE.sub("", rule_name, count=1).strip() or rule_name


def _deduplicate_triggered_rules(triggered_rules: list[str]) -> list[str]:
    """
    Keep one representative rule per pattern (same name after stripping cluster prefix).
    E.g. c11_could_not_read_from_stream, c12_could_not_read_from_stream -> keep first only.
    """
    seen_pattern: dict[str, str] = {}
    for name in triggered_rules:
        pattern = _strip_cluster_prefix(name)
        if pattern not in seen_pattern:
            seen_pattern[pattern] = name
    return list(seen_pattern.values())


def explain_session_instance(
    session_id: str,
    seq_row_ids: list[int],
    rid2eid: dict,
    kb_rules: list[dict],
    rule_to_cluster: dict | None = None,
    model_score: float | None = None,
    llm_vote: int | None = None,
) -> dict:
    """
    Produce instance-level explanation for one session.
    Returns dict: session_id, model_score, llm_vote, triggered_rules, triggered_ngrams, cluster_reference.
    """
    from logsable.rules import rule_fires_on_seq

    triggered_rules = []
    triggered_ngrams = []
    seq_list = list(seq_row_ids) if seq_row_ids is not None else []

    for r in kb_rules:
        if rule_fires_on_seq(seq_list, r, rid2eid):
            name = r.get("name", "")
            if name:
                triggered_rules.append(name)
            # N-gram style: contains_ngram or ordered_subset as "E1→E2→E3"
            for clause in r.get("if_any", []) + r.get("if_all", []):
                if "contains_ngram" in clause:
                    gram = clause["contains_ngram"]
                    ngram_str = "→".join(rid2eid.get(int(x), f"E{int(x)}") for x in gram)
                    if ngram_str and ngram_str not in triggered_ngrams:
                        triggered_ngrams.append(ngram_str)
                if "ordered_subset" in clause and len(clause["ordered_subset"]) >= 2:
                    gram_str = "→".join(rid2eid.get(int(x), f"E{int(x)}") for x in clause["ordered_subset"])
                    if gram_str and gram_str not in triggered_ngrams:
                        triggered_ngrams.append(gram_str)

    # Deduplicate triggered_rules by pattern (strip c0_/cluster_0_ etc.), keep one per pattern
    triggered_rules = _deduplicate_triggered_rules(triggered_rules)

    cluster_reference = None
    if rule_to_cluster:
        for rid in triggered_rules:
            if rid in rule_to_cluster:
                cluster_reference = rule_to_cluster[rid]
                break

    return {
        "session_id": str(session_id),
        "model_score": float(model_score) if model_score is not None else None,
        "llm_vote": int(llm_vote) if llm_vote is not None else None,
        "triggered_rules": triggered_rules,
        "triggered_ngrams": triggered_ngrams,
        "cluster_reference": cluster_reference,
    }


def _to_row_ids(evseq) -> list[int]:
    """Convert EventSeq (list/numpy of int or str) to list of int row_ids."""
    if evseq is None:
        return []
    if hasattr(evseq, "tolist"):
        evseq = evseq.tolist()
    out = []
    for x in evseq:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            pass
    return out


def explain_sessions_batch(
    seqs_df,
    session_key_col: str,
    kb_rules: list[dict],
    rid2eid: dict,
    rule_to_cluster: dict | None = None,
    model_scores_by_session: dict | None = None,
    llm_votes_by_session: dict | None = None,
) -> list[dict]:
    """
    Produce instance-level explanations for all sessions in seqs_df.
    model_scores_by_session / llm_votes_by_session: optional dict session_id -> value.
    """
    if "EventSeq" not in seqs_df.columns:
        return []
    key_col = session_key_col if session_key_col in seqs_df.columns else "BlockId"
    out = []
    for _, row in seqs_df.iterrows():
        sid = str(row[key_col])
        seq = _to_row_ids(row.get("EventSeq"))
        score = model_scores_by_session.get(sid) if model_scores_by_session else None
        vote = llm_votes_by_session.get(sid) if llm_votes_by_session else None
        out.append(explain_session_instance(
            session_id=sid,
            seq_row_ids=seq,
            rid2eid=rid2eid,
            kb_rules=kb_rules,
            rule_to_cluster=rule_to_cluster,
            model_score=score,
            llm_vote=vote,
        ))
    return out
