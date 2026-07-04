from __future__ import annotations
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple
import hdbscan
from scipy.sparse import issparse
import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize


def vectorize_sequences_tfidf_eid(seqs_df: pd.DataFrame) -> np.ndarray:
    """
    TF-IDF over EventId tokens (e.g., 'E7', 'E10', ...)
    Returns dense float64 matrix suitable for HDBSCAN.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    # join EventIds into a whitespace string per sequence
    corpus = []
    for _, r in seqs_df.iterrows():
        toks = [str(t) for t in r["EventSeq"]]
        corpus.append(" ".join(toks))

    vec = TfidfVectorizer(lowercase=False, token_pattern=r"[^ ]+")
    X = vec.fit_transform(corpus)
    # ensure float64 for HDBSCAN Cython code
    return X.astype(np.float64).toarray()


def vectorize_sequences_tfidf_template(seqs_df: pd.DataFrame, templates_df: pd.DataFrame) -> np.ndarray:
    """
    TF-IDF over EventTemplate text; map EventSeq tokens -> template strings.
    Returns dense float64 matrix.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    eid2tmpl = {}
    if {"EventId", "EventTemplate"}.issubset(templates_df.columns):
        eid2tmpl = {str(e): str(t) for e, t in zip(templates_df["EventId"], templates_df["EventTemplate"])}

    corpus = []
    for _, r in seqs_df.iterrows():
        toks = [eid2tmpl.get(str(t), f"T{t}") for t in r["EventSeq"]]
        corpus.append(" | ".join(toks))

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"[^|]+")
    X = vec.fit_transform(corpus)
    return X.astype(np.float64).toarray()



def _seq_to_tokens(event_seq) -> List[str]:
    """
    Normalize a sequence into list[str] tokens (EventId strings or row ids).
    Accepts list-like or simple comma/bracketed strings.
    """
    if isinstance(event_seq, (list, tuple, np.ndarray)):
        return [str(t) for t in event_seq]
    s = str(event_seq).strip()
    if s.startswith("["):
        try:
            import json
            val = json.loads(s)
            return [str(t) for t in val]
        except Exception:
            return [p.strip() for p in s.strip("[]").split(",") if p.strip()]
    return [p.strip() for p in s.split(",") if p.strip()]

def build_sequence_matrix(
    seqs_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    vectorizer: str = "tfidf-eid"
) -> Tuple[np.ndarray, List[str]]:
    """
    Return (X, seq_keys)
      X: 2D feature matrix for clustering
      seq_keys: BlockId strings aligned with rows in X
    vectorizer:
      - "tfidf-eid": TF-IDF over EventId tokens (fast & robust)
      - "tfidf-template": TF-IDF over template strings (text-heavy)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    seq_keys = seqs_df["BlockId"].astype(str).tolist()

    if vectorizer == "tfidf-template":
        # Map EventId -> template text, then vectorize concatenated template tokens
        eid2tmpl = {}
        if {"EventId", "EventTemplate"}.issubset(templates_df.columns):
            eid2tmpl = {str(e): str(t) for e, t in zip(templates_df["EventId"], templates_df["EventTemplate"])}
        docs = []
        for _, r in seqs_df.iterrows():
            toks = _seq_to_tokens(r["EventSeq"])
            txts = [eid2tmpl.get(str(t), str(t)) for t in toks]
            docs.append(" ".join(txts))
        vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        X = vec.fit_transform(docs).astype(np.float32)
        return X, seq_keys

    # default: TF-IDF over EventId tokens
    docs = []
    for _, r in seqs_df.iterrows():
        toks = _seq_to_tokens(r["EventSeq"])
        docs.append(" ".join(toks))
    vec = TfidfVectorizer(token_pattern=r"[^ ]+", lowercase=False)
    X = vec.fit_transform(docs).astype(np.float32)
    return X, seq_keys

@dataclass
class ClusterResult:
    labels: np.ndarray        # cluster label per row, -1 = noise
    probs: np.ndarray         # membership strength (0..1)
    centers: Dict[int, int]   # cluster_id -> row index of "center"
    extremes: Dict[int, Tuple[int, int]]  # cluster_id -> (min_idx, max_idx)

import numpy as np

def run_hdbscan(
    X,
    min_cluster_size: int = 30,
    min_samples: int | None = None,
    metric: str = "cosine",
    random_state: int = 43,
):
    """
    Run HDBSCAN on session vectors.

    - Accepts dense or scipy sparse matrices.
    - If metric == "cosine", we L2-normalize X and use euclidean
      (cosine distance on unit vectors).
    """
    # ---- normalize X to dense float64 ----
    if issparse(X):
        X = X.toarray().astype(np.float64)
    else:
        X = np.asarray(X, dtype=np.float64)

    if min_samples is None:
        min_samples = min_cluster_size

    # ---- handle cosine metric manually ----
    metric_used = metric
    if metric == "cosine":
        # On unit vectors, euclidean distance is a monotonic transform of cosine.
        X = normalize(X, norm="l2")
        metric_used = "euclidean"

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric_used,
        core_dist_n_jobs=1,
        cluster_selection_method="eom",
    )

    labels = clusterer.fit_predict(X)
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))
    return labels, probs, clusterer




def pick_representatives(X, labels, probs, k_select=3):
    """
    For each cluster (label>=0):
      center = argmax(probability), extremes = two farthest (cosine distance) from center.
    Returns (centers, extremes).
    """
    from sklearn.metrics.pairwise import cosine_distances
    centers, extremes = {}, {}
    K = np.max(labels) if labels.size else -1
    for cid in range(K + 1):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        c = idx[np.argmax(probs[idx])]
        centers[cid] = int(c)
        if idx.size >= 2:
            d = cosine_distances(X[idx], X[c]).reshape(-1)
            order = np.argsort(-d)
            e1 = int(idx[order[0]]) if order.size >= 1 else int(c)
            e2 = int(idx[order[1]]) if order.size >= 2 else int(c)
            extremes[cid] = (e1, e2)
        else:
            extremes[cid] = (int(c), int(c))
    return centers, extremes

def build_cluster_bundle(
    seqs_df: pd.DataFrame,
    templates_df: pd.DataFrame,
    labels: np.ndarray,
    probs: np.ndarray,
    centers: Dict[int, int],
    extremes: Dict[int, Tuple[int, int]],
    max_examples_per_cluster: int = 50
) -> dict:
    """
    Prepare a compact bundle for LLM prompts:
      - meta: dataset stats
      - template_map: (row_id, event_id, event_template)
      - clusters: cid -> {size, representatives, examples[]}
    """
    # template mapping
    template_map = []
    if "EventId" in templates_df.columns:
        for i, (eid, et) in enumerate(zip(templates_df["EventId"].astype(str),
                                          templates_df["EventTemplate"].astype(str))):
            template_map.append({"row_id": i, "event_id": eid, "event_template": et})
    else:
        for i, et in enumerate(templates_df["EventTemplate"].astype(str)):
            template_map.append({"row_id": i, "event_id": str(i), "event_template": et})

    seq_keys = seqs_df["BlockId"].astype(str).tolist()
    labels_seq = seqs_df["Label"].astype(int).tolist()
    events = seqs_df["EventSeq"].tolist()

    clusters = {}
    uniq = sorted([c for c in np.unique(labels) if c >= 0])
    for cid in uniq:
        idx = np.where(labels == cid)[0]
        center_i = centers.get(cid, int(idx[0]))
        end_a, end_b = extremes.get(cid, (center_i, center_i))
        ex = []
        for j in idx[:max_examples_per_cluster]:
            ev = events[j]
            ev_short = ev[:20] if isinstance(ev, list) else ev
            ex.append({
                "seq_key": seq_keys[j],
                "label": int(labels_seq[j]),
                "event_ids": ev_short
            })
        clusters[str(cid)] = {
            "size": int(idx.size),
            "representatives": {
                "center": {"row": int(center_i), "seq_key": seq_keys[center_i]},
                "end_a": {"row": int(end_a), "seq_key": seq_keys[end_a]},
                "end_b": {"row": int(end_b), "seq_key": seq_keys[end_b]},
            },
            "examples": ex
        }

    bundle = {
        "meta": {
            "num_sequences": len(seqs_df),
            "num_clusters": len(uniq),
            "noise": int(np.sum(labels < 0)),
        },
        "template_map": template_map,
        "clusters": clusters
    }
    return bundle
