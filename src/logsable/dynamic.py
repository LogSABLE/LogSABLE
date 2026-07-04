def ensure_dynamic_table(db_path="logdb.sqlite"):
    import sqlite3

    con = sqlite3.connect(db_path)
    c = con.cursor()

    # Create base table (new installs get the full schema)
    c.execute("""
    CREATE TABLE IF NOT EXISTS dynamic_patterns (
        seq_key TEXT,
        dataset TEXT,
        run_tag TEXT,

        -- existing
        snippet TEXT,
        count   INT,
        is_high_risk INT,
        rationale TEXT,

        -- new richer reasoning fields
        summary TEXT,
        core_events TEXT,           -- JSON string
        expected_behavior TEXT,
        observed_behavior TEXT,
        deviations TEXT,            -- JSON string
        confidence REAL,

        PRIMARY KEY (seq_key, dataset, run_tag)
    );
    """)

    # Backfill / migrate old schemas
    c.execute("PRAGMA table_info(dynamic_patterns);")
    cols = {r[1] for r in c.fetchall()}

    # NOTE: SQLite lets you ADD COLUMN, but not easily alter types; we keep it simple.
    adds = [
        ("dataset", "ALTER TABLE dynamic_patterns ADD COLUMN dataset TEXT"),
        ("run_tag", "ALTER TABLE dynamic_patterns ADD COLUMN run_tag TEXT"),
        ("snippet", "ALTER TABLE dynamic_patterns ADD COLUMN snippet TEXT"),
        ("count", "ALTER TABLE dynamic_patterns ADD COLUMN count INT"),
        ("is_high_risk", "ALTER TABLE dynamic_patterns ADD COLUMN is_high_risk INT"),
        ("rationale", "ALTER TABLE dynamic_patterns ADD COLUMN rationale TEXT"),

        ("summary", "ALTER TABLE dynamic_patterns ADD COLUMN summary TEXT"),
        ("core_events", "ALTER TABLE dynamic_patterns ADD COLUMN core_events TEXT"),
        ("expected_behavior", "ALTER TABLE dynamic_patterns ADD COLUMN expected_behavior TEXT"),
        ("observed_behavior", "ALTER TABLE dynamic_patterns ADD COLUMN observed_behavior TEXT"),
        ("deviations", "ALTER TABLE dynamic_patterns ADD COLUMN deviations TEXT"),
        ("confidence", "ALTER TABLE dynamic_patterns ADD COLUMN confidence REAL"),
    ]

    for col, ddl in adds:
        if col not in cols:
            try:
                c.execute(ddl)
            except Exception:
                pass

    con.commit()
    con.close()


def llm_risk_scorer(text):
    """
    Return (is_high_risk:int, count:int, rationale:str).
    Keyword and phrase heuristic for failure-related log text.
    """
    import re
    t = text.lower()
    # phrase-level red flags
    phrases = [
        r"received exception", r"threw exception", r"writeblock .* exception",
        r"connection (reset|refused|timed out)",
        r"(disk|block) (corrupt|corruption|lost|failure)",
        r"(checksum) (mismatch|failed)"
    ]
    # token-level lexicon
    lex = {
        "error","exception","fail","failed","failure","panic","fatal","critical",
        "timeout","timed","denied","refused","corrupt","corruption","abort","aborted",
        "retry","retries","lost","mismatch","hang","stuck","backoff","recover","recovery"
    }

    phrase_hits = sum(1 for p in phrases if re.search(p, t))
    # split on non-letters for coarse tokens
    toks = re.findall(r"[a-z]+", t)
    token_hits = sum(1 for tok in toks if tok in lex)

    # strong singletons
    strong_singleton = ("exception" in toks) or ("error" in toks)

    hits = phrase_hits + token_hits
    is_high = 1 if (hits >= 2 or strong_singleton) else 0
    why_parts = []
    if phrase_hits: why_parts.append(f"{phrase_hits} phrase hit(s)")
    if token_hits:  why_parts.append(f"{token_hits} keyword hit(s)")
    if strong_singleton and hits < 2: why_parts.append("strong singleton (exception/error)")
    rationale = "Heuristic: " + (", ".join(why_parts) if why_parts else "no failure terms")
    return is_high, hits, rationale


def mine_dynamic_llm(seqs_df, templates_df, db_path="logdb.sqlite", preview_k=6, prompt_template=None, dataset: str | None = None, raw_df=None):
    """Export session prompts to JSON (main.py uses build_session_prompt_items + run_llm_sessions_with_cache)."""
    ensure_dynamic_table(db_path)
    export_llm_prompts_for_sessions(
        seqs_df, templates_df,
        out_path="llm_prompts_sessions.json",
        preview_k=preview_k,
        prompt_template=prompt_template,
        include_schema=False,
        dataset=dataset or "HDFS",
        raw_df=raw_df,
    )


def mine_dynamic_sentiment(templates_df, seqs_df, db_path="logdb.sqlite",
                           sentiment_model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                           neg_threshold=2, min_freq_normal=50):
    import sqlite3
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from collections import Counter, defaultdict

    normal_templates = templates_df[templates_df['label']==0]['EventTemplate']
    freq = Counter(normal_templates)
    frequent_normal = {t for t,c in freq.items() if c >= min_freq_normal}

    tok = AutoTokenizer.from_pretrained(sentiment_model)
    mdl = AutoModelForSequenceClassification.from_pretrained(sentiment_model)
    clf = pipeline("sentiment-analysis", model=mdl, tokenizer=tok, truncation=True, max_length=128)


    tmpl_sent = {}
    for t in templates_df['EventTemplate'].drop_duplicates().tolist():
        tmpl_sent[t] = clf(t)[0]['label'].lower()  # negative/neutral/positive

    neg_counts = defaultdict(int)
    for _, row in templates_df.iterrows():
        t = row['EventTemplate']; sk = row['seq_key']
        if tmpl_sent.get(t, 'neutral') == 'negative' and t not in frequent_normal:
            neg_counts[sk] += 1

    conn = sqlite3.connect(db_path); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS dynamic_patterns(
                   seq_key TEXT, snippet TEXT, count INT, is_high_risk INT)""")
    c.execute("DELETE FROM dynamic_patterns")
    rows = []
    for sk in seqs_df['seq_key']:
        cnt = neg_counts.get(sk, 0)
        rows.append((sk, "sentiment_neg_infrequent", cnt, int(cnt >= neg_threshold)))
    c.executemany("INSERT INTO dynamic_patterns VALUES (?,?,?,?)", rows)
    conn.commit(); conn.close()
    print(f"[DYN] wrote dynamic sentiment flags for {len(rows)} sequences")


def _apply_llm_risk_promotion(is_h, conf, cnt, dataset, promote_negatives=None,
                              promote_min_conf=0.85, promote_min_count=2):
    """Apply LLM risk label mapping during ingestion."""
    if int(is_h) == 1:
        return 1
    if promote_negatives is False:
        return 0
    if promote_negatives is None:
        promote_negatives = str(dataset).strip().upper() != "LIBERTY"
    if not promote_negatives:
        return 0
    if float(conf) >= float(promote_min_conf) and int(cnt) >= int(promote_min_count):
        return 1
    return 0


def ingest_llm_session_results_list(
    results,
    db_path="logdb.sqlite",
    dataset="HDFS",
    run_tag="default",
    upsert=True,
    valid_keys=None,
    promote_negatives=None,
    promote_min_conf=0.85,
    promote_min_count=2,
    quiet=False,
):
    """Persist a list of session LLM JSON objects into dynamic_patterns (KB)."""
    import sqlite3, json

    if isinstance(results, dict):
        results = list(results.values())

    if valid_keys is not None and len(results) > 0:
        valid_set = set(valid_keys) if not isinstance(valid_keys, set) else valid_keys
        result_keys = {str(r["seq_key"]) for r in results if isinstance(r, dict) and "seq_key" in r}
        n_match = len(result_keys & valid_set)
        print(f"[LLM-IO] session results: keys matching current run = {n_match} of {len(result_keys)}")

    ensure_dynamic_table(db_path)
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    verb = "INSERT OR REPLACE" if upsert else "INSERT"
    sql = verb + """
        INTO dynamic_patterns (
            seq_key, dataset, run_tag,
            snippet, count, is_high_risk, rationale,
            summary, core_events, expected_behavior, observed_behavior, deviations, confidence
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    n = 0
    for r in results:
        if not isinstance(r, dict) or "seq_key" not in r:
            continue
        try:
            sk = str(r["seq_key"])
            summary = str(r.get("summary", ""))
            core_events = r.get("core_events", [])
            expected_behavior = str(r.get("expected_behavior", ""))
            observed_behavior = str(r.get("observed_behavior", ""))
            deviations = r.get("deviations", [])
            why = str(r.get("rationale", ""))
            conf = float(r.get("confidence", 0.0))

            core_events_json = (
                json.dumps(core_events, ensure_ascii=False)
                if isinstance(core_events, (list, tuple))
                else json.dumps([str(core_events)], ensure_ascii=False)
            )
            deviations_json = (
                json.dumps(deviations, ensure_ascii=False)
                if isinstance(deviations, (list, tuple))
                else json.dumps([str(deviations)], ensure_ascii=False)
            )

            cnt = int(r.get("count", 0))
            is_h = _apply_llm_risk_promotion(
                int(r.get("is_high_risk", 0)), conf, cnt, dataset,
                promote_negatives=promote_negatives,
                promote_min_conf=promote_min_conf,
                promote_min_count=promote_min_count,
            )
            snip = str(r.get("snippet", ""))

            cur.execute(sql, (
                sk, dataset, run_tag,
                snip, cnt, is_h, why,
                summary, core_events_json, expected_behavior, observed_behavior, deviations_json, conf
            ))
            n += 1
        except Exception as e:
            print(f"[LLM-IO] bad session entry {r}: {e}")

    con.commit()
    n_pos = 0
    try:
        row = cur.execute(
            "SELECT SUM(is_high_risk) FROM dynamic_patterns WHERE dataset=? AND run_tag=?",
            (dataset, run_tag),
        ).fetchone()
        n_pos = int(row[0] or 0)
    except Exception:
        pass
    con.close()
    if not quiet:
        rate = n_pos / max(1, n)
        print(
            f"[LLM-IO] stored {n} session judgments for dataset={dataset}, run_tag={run_tag} "
            f"(is_high_risk=1: {n_pos}/{n}, {rate:.1%})"
        )
    return n


def ingest_llm_session_risks(
    json_path="llm_results_sessions.json",
    db_path="logdb.sqlite",
    dataset="HDFS",
    run_tag="default",
    upsert=True,
    clear_run=False,
    valid_keys=None,
    promote_negatives=None,
    promote_min_conf=0.85,
    promote_min_count=2,
):
    """Ingest session LLM results from a JSON file into dynamic_patterns."""
    import json, os, sqlite3

    if not os.path.exists(json_path):
        print(f"[LLM-IO] session results file not found: {json_path} (skipping)")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    if clear_run:
        ensure_dynamic_table(db_path)
        with sqlite3.connect(db_path) as con:
            con.execute(
                "DELETE FROM dynamic_patterns WHERE dataset=? AND run_tag=?",
                (dataset, run_tag),
            )
            con.commit()

    return ingest_llm_session_results_list(
        results,
        db_path=db_path,
        dataset=dataset,
        run_tag=run_tag,
        upsert=upsert,
        valid_keys=valid_keys,
        promote_negatives=promote_negatives,
        promote_min_conf=promote_min_conf,
        promote_min_count=promote_min_count,
    )

def fetch_dynamic_votes(db_path="logdb.sqlite", dataset="HDFS", run_tag="default"):
    import sqlite3, pandas as pd
    with sqlite3.connect(db_path) as con:
        q = "SELECT seq_key, is_high_risk FROM dynamic_patterns WHERE dataset=? AND run_tag=?"
        df = pd.read_sql(q, con, params=[dataset, run_tag])
    return {str(k): int(v) for k, v in zip(df["seq_key"], df["is_high_risk"])}

def _auto_cluster_summary(cluster: dict, default_top_k: int = 6) -> str:
    """
    Create a short human-friendly summary for a cluster when 'summary' is absent.
    Tries, in order:
      - cluster['top_templates'] ([(eid or tmpl, count), ...])
      - cluster['members'] count
    """
    size = cluster.get("size")
    parts = []
    if size is not None:
        parts.append(f"Cluster size: {size}")

    tops = cluster.get("top_templates") or cluster.get("top_events") or []
    if isinstance(tops, (list, tuple)) and tops:
        # tops is expected like [('E7', 42), ('E10', 30), ...] or [('template text', 5), ...]
        labels = []
        for lab, cnt in tops[:default_top_k]:
            try:
                labels.append(str(lab))
            except Exception:
                labels.append(repr(lab))
        if labels:
            parts.append("Frequent events: " + ", ".join(labels))

    if not parts:
        parts = ["Cluster summary unavailable"]
    return " | ".join(parts)


def make_llm_prompt(cluster_id, bundle, dataset: str = 'HDFS'):
    """
    Build a single-cluster prompt (string) for rule generation.
    Robust to missing 'summary' in cluster.
    """
    import json
    preface = (
        f"You are given log template metadata and a summary of an anomalous cluster in the {dataset} dataset.\n"
        "Derive a concise set of boolean rules that would identify sequences like those in this cluster.\n"
        "Emit STRICT JSON with this schema: {\"rules\": [{\"name\": str, "
        "\"if_any\": [ ... ], \"if_all\": [ ... ], \"explanation\": str, \"confidence\": number }]}\n"
        "Use primitives: contains_ngram (list of row_ids), ordered_subset (list of row_ids), "
        "min_count: {\"event_id\": <EventId>, \"count\": k}, absent_within: {\"event_id\": <EventId>, \"window\": w}.\n"
        "Be minimal and precise (3–10 rules). No prose outside JSON.\n"
    )

    clusters = bundle.get("clusters", {})
    # cluster_id may be a numpy/scalar; normalize to string key lookups too
    cid_str = str(cluster_id)
    cluster = clusters.get(cluster_id) or clusters.get(cid_str) or {}

    # Safe summary
    summary = cluster.get("summary") or _auto_cluster_summary(cluster)

    compact = {
        "meta": bundle.get("meta", {}),
        "cluster_id": cluster_id,
        "cluster_size": cluster.get("size"),
        "summary": summary,
        # keep this small for token budget
        "template_map_sample": (bundle.get("template_map") or [])[:50]
    }
    return preface + "\n=== DATA ===\n" + json.dumps(compact, indent=2)

def load_llm_session_labels(db_path: str, dataset: str, run_tag: str, min_conf: float = 0.0):
    import sqlite3
    out = {}
    with sqlite3.connect(db_path) as con:
        q = """
        SELECT TRIM(seq_key) AS seq_key, is_high_risk, confidence
        FROM dynamic_patterns
        WHERE dataset=? AND run_tag=?
        """
        rows = con.execute(q, (dataset, run_tag)).fetchall()
    for k, y, conf in rows:
        conf = float(conf) if conf is not None else 0.0
        if conf >= float(min_conf):
            out[str(k)] = (int(y), conf)
    return out


def ingest_llm_cluster_rules(json_path="llm_results_clusters.json"):
    """
    Load cluster-derived rules from LLM output.

    Accepts any of the following top-level JSON shapes:
      - [{"rules":[...]} , {"rules":[...]}]
      - {"0":{"rules":[...]}, "3":{"rules":[...]}}
      - [{...rule...}, {...rule...}]   # flat list of rule objects

    Returns: list of rule dicts (not yet standardized to row_ids).
    """
    import json, os
    if not os.path.exists(json_path):
        print(f"[LLM-IO] cluster results file not found: {json_path} (skipping)")
        return []

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    collected = []

    def collect(node):
        # If it's a list, recurse into each item
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        # If it's a dict, either pick its "rules" or treat the dict itself as a rule
        if isinstance(node, dict):
            # Case: {"rules":[...]} holder
            if "rules" in node and isinstance(node["rules"], list):
                for r in node["rules"]:
                    if isinstance(r, dict):
                        collected.append(r)
                return
            # Case: dict of {"cluster_id": {"rules":[...]}}
            # Recurse into values to find nested "rules"
            for v in node.values():
                collect(v)
            # Also: if it *looks like* a rule (has name/if_any/if_all), accept it
            maybe_keys = {"name", "if_all", "if_any"}
            if any(k in node for k in maybe_keys) and not ("rules" in node):
                collected.append(node)
            return
        # Anything else is ignored

    collect(data)
    print(f"[LLM-IO] ingested {len(collected)} rules from {json_path}")
    return collected

def build_session_prompt_items(
    seqs_df,
    templates_df,
    preview_k=6,
    prompt_template: str | None = None,
    include_schema: bool = False,
    dataset: str = "HDFS",
    raw_df=None,
):
    """Build in-memory session prompt items (no file I/O)."""
    import numpy as np

    cols = set(seqs_df.columns)
    ds_upper = (dataset or "").upper()

    key_col = None
    if ds_upper == "HDFS":
        for cand in ["BlockId", "blk_id", "SessionId", "seq_key"]:
            if cand in cols:
                key_col = cand
                break
    elif ds_upper == "BGL":
        for cand in ["BlockId", "Node", "NodeId", "SessionId", "seq_key"]:
            if cand in cols:
                key_col = cand
                break
    else:
        for cand in ["seq_key", "BlockId", "SessionId", "Host"]:
            if cand in cols:
                key_col = cand
                break

    if key_col is None:
        raise ValueError(
            f"Could not find a suitable session key column in seqs_df for dataset={dataset}. "
            f"Available columns: {sorted(cols)}"
        )

    use_raw_log_snippet = (
        ds_upper == "LIBERTY"
        and raw_df is not None
        and hasattr(raw_df, "columns")
        and "_seq_key" in raw_df.columns
        and "log" in raw_df.columns
    )
    raw_log_by_key = {}
    if use_raw_log_snippet:
        for sk, grp in raw_df.groupby("_seq_key", sort=False):
            raw_log_by_key[str(sk)] = grp["log"].astype(str).head(preview_k).tolist()

    rid2tmpl = {}
    if "EventTemplate" in templates_df.columns:
        rid2tmpl = {i: t for i, t in enumerate(templates_df["EventTemplate"].tolist())}
    eid2tmpl = {}
    if {"EventId", "EventTemplate"}.issubset(templates_df.columns):
        eid2tmpl = {str(e): t for e, t in zip(templates_df["EventId"], templates_df["EventTemplate"])}

    def tok2text(tok):
        if isinstance(tok, (int, np.integer)) and tok in rid2tmpl:
            return rid2tmpl[tok]
        s = str(tok)
        if s in eid2tmpl:
            return eid2tmpl[s]
        try:
            x = int(s)
            if x in rid2tmpl:
                return rid2tmpl[x]
        except Exception:
            pass
        return f"[T{tok}]"

    items = []
    for _, row in seqs_df.iterrows():
        seq_key = str(row[key_col])
        if use_raw_log_snippet and seq_key in raw_log_by_key:
            log_lines = raw_log_by_key[seq_key]
            snippet = "\n".join(log_lines[:preview_k]) if log_lines else "(no logs)"
        else:
            seq = row.get("EventSeq") or []
            templs = [tok2text(t) for t in (seq[:preview_k] if hasattr(seq, "__getitem__") else [])]
            snippet = " | ".join(templs) if templs else "(empty)"
        prompt = build_session_prompt(seq_key, snippet, preview_k, prompt_template, dataset=dataset)
        obj = {"seq_key": seq_key, "prompt": prompt, "snippet": snippet}
        if include_schema:
            obj["schema"] = {
                "seq_key": "string",
                "is_high_risk": "int (0 or 1)",
                "rationale": "string",
            }
        items.append(obj)
    return items


def export_llm_prompts_for_sessions(
    seqs_df,
    templates_df,
    out_path="llm_prompts_sessions.json",
    preview_k=6,
    prompt_template: str | None = None,
    include_schema: bool = False,
    dataset: str = "HDFS",
    raw_df=None,
):
    import json

    items = build_session_prompt_items(
        seqs_df, templates_df,
        preview_k=preview_k,
        prompt_template=prompt_template,
        include_schema=include_schema,
        dataset=dataset,
        raw_df=raw_df,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"[dynamic] wrote {len(items)} session prompts to {out_path}")





def build_cluster_prompt_items(bundle, dataset: str = "HDFS"):
    """Return {cluster_id: prompt_text} for in-memory LLM calls."""
    prompts = {}
    for cid in bundle.get("clusters", {}):
        prompts[str(cid)] = make_llm_prompt(cid, bundle, dataset=dataset)
    return prompts


def export_llm_prompts_for_clusters(bundle, out_path="llm_prompts_clusters.json", dataset: str = 'HDFS'):
    import json
    prompts = build_cluster_prompt_items(bundle, dataset=dataset)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)
    print(f"[LLM-IO] wrote cluster prompts → {out_path}")




DEFAULT_SESSION_PROMPT = """You are assessing whether the following {dataset} log *sequence* indicates a HIGH-RISK anomaly.

You MUST return STRICT JSON only, matching exactly this schema:
{{
  "seq_key": "string",
  "expected_behavior": "string",
  "observed_behavior": "string",
  "deviations": ["string", "..."],
  "summary": "string",
  "core_events": ["string", "..."],
  "is_high_risk": 0 or 1,
  "count": "int (number of evidence hits you relied on, >=0)",
  "rationale": "string",
  "confidence": "float in [0,1]"
}}

Guidelines:
- expected_behavior: what a normal sequence would look like for this dataset/task type.
- observed_behavior: what actually happened in THIS snippet.
- deviations: bullet-like strings describing mismatches between expected vs observed.
- core_events: include the *exact* key events/phrases from the snippet that drove the decision.
- summary: 1–2 sentence neutral summary.
- Treat cleanup/deletion events as weak signals unless combined with missing lifecycle completion, failed verification, contradictory ordering, or unresolved repeated failures.
- No extra keys. No prose outside JSON.

seq_key={seq_key}
SNIPPET:
{snippet}
"""



def build_session_prompt(seq_key: str, snippet: str, k: int, prompt_template: str | None = None, dataset: str = 'HDFS') -> str:
    tpl = prompt_template or DEFAULT_SESSION_PROMPT
    return tpl.format(seq_key=seq_key, snippet=snippet, k=k, dataset=dataset)


# ---------------------------------------------------------------------------
# KB-backed live LLM (cache-first; no JSON file required)
# ---------------------------------------------------------------------------

def ensure_cluster_llm_table(db_path="logdb.sqlite"):
    import sqlite3

    con = sqlite3.connect(db_path)
    con.execute("""
    CREATE TABLE IF NOT EXISTS cluster_llm_results (
        cluster_id TEXT,
        dataset TEXT,
        run_tag TEXT,
        response_json TEXT,
        updated_at REAL,
        PRIMARY KEY (cluster_id, dataset, run_tag)
    );
    """)
    con.commit()
    con.close()


def _session_row_has_content(row) -> bool:
    if row is None:
        return False
    rationale = row[6] if len(row) > 6 else ""
    return bool(rationale and str(rationale).strip())


def get_cached_session_row(db_path, seq_key, dataset, run_tag, cross_run=True):
    import sqlite3

    ensure_dynamic_table(db_path)
    sk = str(seq_key)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        row = cur.execute(
            """
            SELECT seq_key, dataset, run_tag, snippet, count, is_high_risk, rationale,
                   summary, core_events, expected_behavior, observed_behavior, deviations, confidence
            FROM dynamic_patterns
            WHERE seq_key=? AND (dataset=? OR (dataset IS NULL AND ? IS NOT NULL))
              AND run_tag=?
            """,
            (sk, dataset, dataset, run_tag),
        ).fetchone()
        if _session_row_has_content(row):
            return row
        if not cross_run:
            return None
        return cur.execute(
            """
            SELECT seq_key, dataset, run_tag, snippet, count, is_high_risk, rationale,
                   summary, core_events, expected_behavior, observed_behavior, deviations, confidence
            FROM dynamic_patterns
            WHERE seq_key=? AND (dataset=? OR dataset IS NULL)
              AND rationale IS NOT NULL AND TRIM(rationale) != ''
            ORDER BY CASE WHEN run_tag=? THEN 0 ELSE 1 END,
                     CASE WHEN dataset=? THEN 0 WHEN dataset IS NULL THEN 1 ELSE 2 END,
                     rowid DESC
            LIMIT 1
            """,
            (sk, dataset, run_tag, dataset),
        ).fetchone()


def _session_row_to_result(row, snippet_fallback=""):
    import json

    if row is None:
        return None
    core_raw = row[8]
    dev_raw = row[11]
    try:
        core_events = json.loads(core_raw) if core_raw else []
    except Exception:
        core_events = []
    try:
        deviations = json.loads(dev_raw) if dev_raw else []
    except Exception:
        deviations = []
    return {
        "seq_key": str(row[0]),
        "snippet": str(row[3] or snippet_fallback),
        "is_high_risk": int(row[5] or 0),
        "count": int(row[4] or 0),
        "rationale": str(row[6] or ""),
        "summary": str(row[7] or ""),
        "core_events": core_events,
        "expected_behavior": str(row[9] or ""),
        "observed_behavior": str(row[10] or ""),
        "deviations": deviations,
        "confidence": float(row[12] if row[12] is not None else 0.0),
    }


def get_cached_cluster_response(db_path, cluster_id, dataset, run_tag, cross_run=True):
    import sqlite3, json

    ensure_cluster_llm_table(db_path)
    with sqlite3.connect(db_path) as con:
        cur = con.cursor()
        row = cur.execute(
            "SELECT response_json FROM cluster_llm_results WHERE cluster_id=? AND dataset=? AND run_tag=?",
            (str(cluster_id), dataset, run_tag),
        ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                pass
        if not cross_run:
            return None
        row = cur.execute(
            """
            SELECT response_json FROM cluster_llm_results
            WHERE cluster_id=? AND dataset=? AND response_json IS NOT NULL AND TRIM(response_json) != ''
            ORDER BY CASE WHEN run_tag=? THEN 0 ELSE 1 END, updated_at DESC
            LIMIT 1
            """,
            (str(cluster_id), dataset, run_tag),
        ).fetchone()
        if row and row[0]:
            try:
                return json.loads(row[0])
            except Exception:
                return None
    return None


def store_cluster_llm_response(db_path, cluster_id, dataset, run_tag, response: dict):
    import sqlite3, json, time

    ensure_cluster_llm_table(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute(
            """
            INSERT OR REPLACE INTO cluster_llm_results
            (cluster_id, dataset, run_tag, response_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (str(cluster_id), dataset, run_tag, json.dumps(response, ensure_ascii=False), time.time()),
        )
        con.commit()


def run_llm_sessions_with_cache(
    prompt_items,
    db_path="logdb.sqlite",
    dataset="HDFS",
    run_tag="default",
    model=None,
    max_items=None,
    sleep_s=0.0,
    use_cache=True,
    cross_run_cache=True,
    llm_enabled=True,
    debug_jsonl_path=None,
    valid_keys=None,
    promote_negatives=None,
    promote_min_conf=0.85,
    promote_min_count=2,
):
    """
    For each session: reuse KB row on cache hit; otherwise call OpenAI and store in dynamic_patterns.
    Returns stats dict {cache_hits, api_calls, failures, stored}.
    """
    import json
    import time as _time
    from logsable.llm_openai import call_llm_json, DEFAULT_MODEL

    ensure_dynamic_table(db_path)
    model = model or DEFAULT_MODEL
    n_total = len(prompt_items) if max_items is None else min(len(prompt_items), max_items)

    stats = {"cache_hits": 0, "api_calls": 0, "failures": 0, "stored": 0, "copied_from_cache": 0}
    to_store = []
    dbg = open(debug_jsonl_path, "w", encoding="utf-8") if debug_jsonl_path else None

    try:
        for i in range(n_total):
            item = prompt_items[i]
            seq_key = str(item["seq_key"])
            snippet = str(item.get("snippet", ""))

            if use_cache:
                cached_row = get_cached_session_row(
                    db_path, seq_key, dataset, run_tag, cross_run=cross_run_cache
                )
                if cached_row is not None:
                    cached_run = str(cached_row[2])
                    if cached_run == run_tag:
                        stats["cache_hits"] += 1
                        if dbg:
                            dbg.write(json.dumps({"seq_key": seq_key, "source": "cache"}) + "\n")
                        continue
                    result = _session_row_to_result(cached_row, snippet_fallback=snippet)
                    if result:
                        to_store.append(result)
                        stats["copied_from_cache"] += 1
                        stats["cache_hits"] += 1
                        if dbg:
                            dbg.write(json.dumps({"seq_key": seq_key, "source": "cache_copy"}) + "\n")
                        continue

            if not llm_enabled:
                stats["failures"] += 1
                continue

            try:
                out = call_llm_json(item["prompt"], model=model)
                if "seq_key" not in out:
                    out["seq_key"] = seq_key
                if snippet and not out.get("snippet"):
                    out["snippet"] = snippet
                to_store.append(out)
                stats["api_calls"] += 1
                if dbg:
                    dbg.write(json.dumps({"seq_key": seq_key, "source": "api", "response": out}) + "\n")
            except Exception as e:
                stats["failures"] += 1
                print(f"[LLM] session {seq_key} failed: {e}")
                if dbg:
                    dbg.write(json.dumps({"seq_key": seq_key, "source": "error", "error": str(e)}) + "\n")

            if sleep_s > 0:
                _time.sleep(sleep_s)
    finally:
        if dbg:
            dbg.close()

    if to_store:
        stats["stored"] = ingest_llm_session_results_list(
            to_store,
            db_path=db_path,
            dataset=dataset,
            run_tag=run_tag,
            valid_keys=valid_keys,
            promote_negatives=promote_negatives,
            promote_min_conf=promote_min_conf,
            promote_min_count=promote_min_count,
        )

    print(
        f"[LLM] sessions dataset={dataset} run_tag={run_tag}: "
        f"total={n_total} cache_hits={stats['cache_hits']} "
        f"(copied={stats['copied_from_cache']}) api_calls={stats['api_calls']} "
        f"failures={stats['failures']} model={model}"
    )
    return stats


def run_llm_clusters_with_cache(
    cluster_prompts: dict,
    db_path="logdb.sqlite",
    dataset="HDFS",
    run_tag="default",
    model=None,
    max_items=None,
    sleep_s=0.0,
    use_cache=True,
    cross_run_cache=True,
    llm_enabled=True,
    debug_jsonl_path=None,
):
    """
    For each cluster: reuse KB on cache hit; otherwise call OpenAI and store in cluster_llm_results.
    Returns list of {cluster_id, response}.
    """
    import time as _time, json
    from logsable.llm_openai import call_llm_json, DEFAULT_MODEL

    ensure_cluster_llm_table(db_path)
    model = model or DEFAULT_MODEL
    items = list(cluster_prompts.items())
    n_total = len(items) if max_items is None else min(len(items), max_items)

    stats = {"cache_hits": 0, "api_calls": 0, "failures": 0}
    results = []
    dbg = open(debug_jsonl_path, "w", encoding="utf-8") if debug_jsonl_path else None

    try:
        for i in range(n_total):
            cid, prompt = items[i]
            cid = str(cid)

            response = None
            if use_cache:
                response = get_cached_cluster_response(
                    db_path, cid, dataset, run_tag, cross_run=cross_run_cache
                )
                if response is not None:
                    stats["cache_hits"] += 1
                    cached_for_run = get_cached_cluster_response(
                        db_path, cid, dataset, run_tag, cross_run=False
                    )
                    if cached_for_run is None:
                        store_cluster_llm_response(db_path, cid, dataset, run_tag, response)
                    results.append({"cluster_id": cid, "response": response})
                    if dbg:
                        dbg.write(json.dumps({"cluster_id": cid, "source": "cache"}) + "\n")
                    continue

            if not llm_enabled:
                stats["failures"] += 1
                continue

            try:
                response = call_llm_json(prompt, model=model)
                store_cluster_llm_response(db_path, cid, dataset, run_tag, response)
                stats["api_calls"] += 1
                results.append({"cluster_id": cid, "response": response})
                if dbg:
                    dbg.write(json.dumps({"cluster_id": cid, "source": "api", "response": response}) + "\n")
            except Exception as e:
                stats["failures"] += 1
                print(f"[LLM] cluster {cid} failed: {e}")
                if dbg:
                    dbg.write(json.dumps({"cluster_id": cid, "source": "error", "error": str(e)}) + "\n")

            if sleep_s > 0:
                _time.sleep(sleep_s)
    finally:
        if dbg:
            dbg.close()

    print(
        f"[LLM] clusters dataset={dataset} run_tag={run_tag}: "
        f"total={n_total} cache_hits={stats['cache_hits']} "
        f"api_calls={stats['api_calls']} failures={stats['failures']} model={model}"
    )
    return results


def ingest_llm_cluster_rules_from_responses(cluster_results):
    """Extract rule dicts from live/cached cluster LLM responses."""
    collected = []

    def collect(node):
        if isinstance(node, list):
            for item in node:
                collect(item)
            return
        if isinstance(node, dict):
            if "rules" in node and isinstance(node["rules"], list):
                for r in node["rules"]:
                    if isinstance(r, dict):
                        collected.append(r)
                return
            for v in node.values():
                collect(v)
            maybe_keys = {"name", "if_all", "if_any"}
            if any(k in node for k in maybe_keys) and "rules" not in node:
                collected.append(node)

    for cr in cluster_results or []:
        resp = cr.get("response", cr) if isinstance(cr, dict) else cr
        collect(resp)

    print(f"[LLM-IO] extracted {len(collected)} cluster rules from {len(cluster_results or [])} cluster responses")
    return collected

