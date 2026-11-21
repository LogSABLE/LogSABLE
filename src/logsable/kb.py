from logsable.common_imports import *

# ===================== KNOWLEDGE BASE (KB) =====================

def kb_init(db_path="logdb.sqlite"):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS discrete_patterns(
      event_id TEXT PRIMARY KEY,
      status   TEXT CHECK(status IN ('normal','anomalous')),
      support  INTEGER
    );
    CREATE TABLE IF NOT EXISTS ngram_patterns(
      ngram TEXT PRIMARY KEY,
      n     INTEGER,
      status TEXT,
      support INTEGER
    );
    CREATE TABLE IF NOT EXISTS semantic_index(
      seq_key TEXT PRIMARY KEY,
      label   INTEGER,
      emb     BLOB
    );
    CREATE TABLE IF NOT EXISTS dynamic_patterns(
      seq_key TEXT,
      snippet TEXT,
      count   INTEGER,
      is_high_risk INTEGER,
      PRIMARY KEY (seq_key, snippet)
    );
    CREATE TABLE IF NOT EXISTS judgments(
      pattern_key  TEXT,
      pattern_type TEXT,
      status       TEXT,
      explanation  TEXT,
      updated_at   TEXT,
      PRIMARY KEY (pattern_key, pattern_type)
    );
    """)
    con.commit()
    return con

def kb_upsert_discrete(con, event_id, status, support):
    con.execute("INSERT OR REPLACE INTO discrete_patterns(event_id,status,support) VALUES (?,?,?)",
                (event_id, status, int(support)))

def kb_upsert_ngram(con, toks, status, support):
    ngram = " ".join(map(str, toks)); n = len(toks)
    con.execute("INSERT OR REPLACE INTO ngram_patterns(ngram,n,status,support) VALUES (?,?,?,?)",
                (ngram, n, status, int(support)))

def kb_upsert_semantic(con, seq_key, label, emb_vec):
    buf = io.BytesIO(); pickle.dump(emb_vec.astype("float32"), buf)
    con.execute("INSERT OR REPLACE INTO semantic_index(seq_key,label,emb) VALUES (?,?,?)",
                (str(seq_key), int(label), sqlite3.Binary(buf.getvalue())))

def kb_upsert_dynamic(con, seq_key, snippet, count, is_high_risk):
    con.execute("INSERT OR REPLACE INTO dynamic_patterns(seq_key,snippet,count,is_high_risk) VALUES (?,?,?,?)",
                (str(seq_key), snippet, int(count), int(is_high_risk)))

def kb_get_judgment(con, pattern_key, pattern_type):
    row = con.execute("SELECT status, explanation FROM judgments WHERE pattern_key=? AND pattern_type=?",
                      (pattern_key, pattern_type)).fetchone()
    return (row[0], row[1]) if row else None

def kb_set_judgment(con, pattern_key, pattern_type, status, explanation=""):
    con.execute("INSERT OR REPLACE INTO judgments(pattern_key,pattern_type,status,explanation,updated_at) VALUES (?,?,?,?,?)",
                (pattern_key, pattern_type, status, explanation, time.strftime("%Y-%m-%d %H:%M:%S")))

import sqlite3, time, json

def ensure_kb_rules(db_path="logdb.sqlite"):
    with sqlite3.connect(db_path) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS kb_rules (
            rule_id    TEXT PRIMARY KEY,
            source     TEXT,        -- 'llm_cluster' | 'auto_mined' | ...
            dataset    TEXT,        -- e.g., 'HDFS'
            model      TEXT,        -- e.g., 'neurallog'
            vocab_from TEXT,        -- e.g., 'EventId'
            scope      TEXT,        -- 'dataset' (default) or 'global'
            rule_json  TEXT,        -- serialized JSON rule
            created_at REAL,
            is_active  INT,
            run_tag    TEXT         
        )
        """)
        try:
            con.execute("ALTER TABLE kb_rules ADD COLUMN run_tag TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists (older DB)


# ---------- EXPLAINABILITY ARTIFACTS ----------

def ensure_explainability_tables(db_path="logdb.sqlite"):
    """Create tables for cluster-level explainability artifacts and rule→cluster mapping."""
    with sqlite3.connect(db_path) as con:
        con.execute("""
        CREATE TABLE IF NOT EXISTS explainability_cluster_artifacts (
            run_tag    TEXT NOT NULL,
            dataset    TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            artifact_json TEXT NOT NULL,
            created_at REAL,
            PRIMARY KEY (run_tag, dataset, cluster_id)
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS explainability_rule_to_cluster (
            rule_id    TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            run_tag    TEXT NOT NULL,
            dataset    TEXT NOT NULL,
            PRIMARY KEY (rule_id, run_tag)
        )
        """)
        con.execute("""
        CREATE TABLE IF NOT EXISTS explainability_seq_cluster (
            seq_key    TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            run_tag    TEXT NOT NULL,
            dataset    TEXT NOT NULL,
            PRIMARY KEY (seq_key, run_tag)
        )
        """)
    return


def store_seq_key_to_cluster_from_bundle(bundle: dict, run_tag: str, dataset: str, db_path: str = "logdb.sqlite"):
    """Persist seq_key -> cluster_id from cluster bundle (for retrieval accuracy)."""
    ensure_explainability_tables(db_path)
    rows = []
    for cid, data in bundle.get("clusters", {}).items():
        for ex in data.get("examples", []):
            sk = ex.get("seq_key")
            if sk is not None:
                rows.append((str(sk), str(cid), run_tag, dataset))
    if not rows:
        return
    with sqlite3.connect(db_path) as con:
        con.executemany("""
            INSERT OR REPLACE INTO explainability_seq_cluster (seq_key, cluster_id, run_tag, dataset)
            VALUES (?, ?, ?, ?)
        """, rows)
    print(f"[KB] stored {len(rows)} seq_key→cluster mappings (run_tag={run_tag!r})")


def store_cluster_artifact(run_tag: str, dataset: str, cluster_id: str, artifact: dict, db_path="logdb.sqlite"):
    """Store one cluster-level explainability artifact (JSON)."""
    import time
    ensure_explainability_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("""
        INSERT OR REPLACE INTO explainability_cluster_artifacts
        (run_tag, dataset, cluster_id, artifact_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """, (run_tag, dataset, str(cluster_id), json.dumps(artifact, ensure_ascii=False), time.time()))
    return


def store_rule_to_cluster_mapping(rule_id: str, cluster_id: str, run_tag: str, dataset: str, db_path="logdb.sqlite"):
    """Record that a rule belongs to a cluster (for instance-level cluster_reference)."""
    ensure_explainability_tables(db_path)
    with sqlite3.connect(db_path) as con:
        con.execute("""
        INSERT OR REPLACE INTO explainability_rule_to_cluster (rule_id, cluster_id, run_tag, dataset)
        VALUES (?, ?, ?, ?)
        """, (str(rule_id), str(cluster_id), run_tag, dataset))
    return


def load_rule_to_cluster_map(run_tag: str, dataset: str, db_path="logdb.sqlite") -> dict:
    """Return dict rule_id -> cluster_id for the given run_tag and dataset."""
    ensure_explainability_tables(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute("""
        SELECT rule_id, cluster_id FROM explainability_rule_to_cluster
        WHERE run_tag = ? AND dataset = ?
        """, (run_tag, dataset)).fetchall()
    return {r[0]: r[1] for r in rows}


def load_cluster_artifacts(run_tag: str, dataset: str, db_path="logdb.sqlite") -> dict:
    """Return dict cluster_id -> artifact (parsed JSON)."""
    ensure_explainability_tables(db_path)
    with sqlite3.connect(db_path) as con:
        rows = con.execute("""
        SELECT cluster_id, artifact_json FROM explainability_cluster_artifacts
        WHERE run_tag = ? AND dataset = ?
        """, (run_tag, dataset)).fetchall()
    return {r[0]: json.loads(r[1]) for r in rows}


def load_seq_key_to_cluster(run_tag: str, dataset: str, db_path="logdb.sqlite") -> dict:
    """Return dict seq_key -> cluster_id for retrieval accuracy."""
    ensure_explainability_tables(db_path)
    try:
        with sqlite3.connect(db_path) as con:
            rows = con.execute("""
                SELECT seq_key, cluster_id FROM explainability_seq_cluster
                WHERE run_tag = ? AND dataset = ?
            """, (run_tag, dataset)).fetchall()
        return {r[0]: r[1] for r in rows}
    except sqlite3.OperationalError:
        return {}


def clear_kb_rules_for_run(
    cfg,
    source,
    run_tag=None,
    db_path="logdb.sqlite",
    scope=None,
):
    """Remove prior KB rows for this dataset/model/run_tag/source (avoids stale OR accumulation)."""
    import sqlite3
    ensure_kb_rules(db_path)
    ds = cfg.data["dataset"]
    mdl = cfg.model["name"]
    voc = cfg.model["vocab_from"]
    tag = run_tag if run_tag is not None else ""
    sql = """
        DELETE FROM kb_rules
        WHERE dataset=? AND model=? AND vocab_from=? AND source=? AND run_tag=?
    """
    params = [ds, mdl, voc, source, tag]
    if scope is not None:
        sql += " AND scope=?"
        params.append(scope)
    with sqlite3.connect(db_path) as con:
        cur = con.execute(sql, params)
        n = int(cur.rowcount if cur.rowcount is not None else 0)
    if n > 0:
        print(
            f"[KB] cleared {n} stale {source!r} rules "
            f"(dataset={ds}, model={mdl}, run_tag={tag!r})"
        )
    return n


def promote_rules_to_kb(
    rules,
    cfg,
    source="llm_cluster",
    scope="dataset",
    run_tag=None,
    db_path="logdb.sqlite",
    clear_existing=False,
):
    import sqlite3, time, json
    from uuid import uuid4
    ensure_kb_rules(db_path)

    ds   = cfg.data["dataset"]
    mdl  = cfg.model["name"]
    voc  = cfg.model["vocab_from"]
    now  = time.time()
    tag  = (run_tag if run_tag is not None else "")

    if clear_existing:
        clear_kb_rules_for_run(cfg, source, run_tag=tag, db_path=db_path, scope=scope)

    rows = []
    for r in rules:
        rid = r.get("name") or f"{source}_{uuid4().hex[:8]}"
        rows.append((rid, source, ds, mdl, voc, scope, json.dumps(r), now, 1, tag))

    with sqlite3.connect(db_path) as con:
        con.executemany("""
            INSERT OR REPLACE INTO kb_rules
            (rule_id,source,dataset,model,vocab_from,scope,rule_json,created_at,is_active,run_tag)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, rows)
    print(
        f"[KB] promoted {len(rows)} rules → kb_rules "
        f"(dataset={ds}, model={mdl}, source={source!r}, run_tag={tag!r}, scope={scope})"
    )
    return len(rows)


def load_active_kb_rules(
    cfg,
    allow_global=False,
    run_tag=None,
    max_rules=None,
    db_path="logdb.sqlite",
    source=None,
):
    import sqlite3, json
    ensure_kb_rules(db_path)
    ds  = cfg.data["dataset"]
    mdl = cfg.model["name"]
    voc = cfg.model["vocab_from"]
    limit_clause = ("ORDER BY created_at DESC LIMIT " + str(int(max_rules))) if max_rules is not None and max_rules > 0 else ""
    source_clause = ""
    source_params = []
    if source is not None:
        if isinstance(source, (list, tuple, set)):
            srcs = list(source)
        else:
            srcs = [str(source)]
        if srcs:
            source_clause = f" AND source IN ({','.join('?' for _ in srcs)})"
            source_params = srcs

    with sqlite3.connect(db_path) as con:
        # When run_tag is set, load only rules from this run (avoids loading 4k+ accumulated rules)
        if run_tag is not None and run_tag != "":
            if allow_global:
                rows = con.execute(
                    f"""
                    SELECT rule_json FROM kb_rules
                    WHERE is_active=1 AND run_tag=? AND
                          ( (dataset=? AND model=? AND vocab_from=?) OR scope='global' )
                          {source_clause}
                    {limit_clause}
                    """,
                    (run_tag, ds, mdl, voc, *source_params),
                ).fetchall()
            else:
                rows = con.execute(
                    f"""
                    SELECT rule_json FROM kb_rules
                    WHERE is_active=1 AND run_tag=? AND dataset=? AND model=? AND vocab_from=?
                          {source_clause}
                    {limit_clause}
                    """,
                    (run_tag, ds, mdl, voc, *source_params),
                ).fetchall()
        else:
            if allow_global:
                rows = con.execute(
                    f"""
                    SELECT rule_json FROM kb_rules
                    WHERE is_active=1 AND
                          ( (dataset=? AND model=? AND vocab_from=?) OR scope='global' )
                          {source_clause}
                    {limit_clause}
                    """,
                    (ds, mdl, voc, *source_params),
                ).fetchall()
            else:
                rows = con.execute(
                    f"""
                    SELECT rule_json FROM kb_rules
                    WHERE is_active=1 AND dataset=? AND model=? AND vocab_from=?
                          {source_clause}
                    {limit_clause}
                    """,
                    (ds, mdl, voc, *source_params),
                ).fetchall()

    rules = [json.loads(rj[0]) for rj in rows]
    src_msg = f", source={source!r}" if source is not None else ""
    print(
        f"[KB] loaded {len(rules)} active rules "
        f"(dataset={ds}, model={mdl}, run_tag={run_tag!r}{src_msg})"
    )
    return rules


def count_active_kb_rules(cfg, run_tag=None, source=None, db_path="logdb.sqlite"):
    import sqlite3
    ensure_kb_rules(db_path)
    ds = cfg.data["dataset"]
    mdl = cfg.model["name"]
    voc = cfg.model["vocab_from"]
    tag = run_tag if run_tag is not None else ""
    source_clause = ""
    source_params = []
    if source is not None:
        if isinstance(source, (list, tuple, set)):
            srcs = list(source)
        else:
            srcs = [str(source)]
        if srcs:
            source_clause = f" AND source IN ({','.join('?' for _ in srcs)})"
            source_params = srcs
    with sqlite3.connect(db_path) as con:
        row = con.execute(
            f"""
            SELECT COUNT(*) FROM kb_rules
            WHERE is_active=1 AND run_tag=? AND dataset=? AND model=? AND vocab_from=?
            {source_clause}
            """,
            (tag, ds, mdl, voc, *source_params),
        ).fetchone()
    return int(row[0]) if row else 0


# ---------- MINERS ----------

def kb_mine_discrete_patterns(seqs_df, label_col="label"):
    """Return dicts: normal_events, anomalous_only_events with supports."""
    # seqs_df columns expected: ['session_key','events','label'] where events is list of event_ids or row_ids
    normals = seqs_df[seqs_df[label_col]==0]["events"]
    abnorms = seqs_df[seqs_df[label_col]==1]["events"]
    c_norm = Counter(e for seq in normals for e in seq)
    c_abn  = Counter(e for seq in abnorms for e in seq)
    normal_set = set(c_norm.keys())
    anomalous_only = {e:c_abn[e] for e in c_abn.keys() if e not in normal_set}
    return c_norm, anomalous_only

def kb_mine_frequent_ngrams(seqs_df, min_bigram=20, min_trigram=15):
    """Use your tokenized sequences (row_ids) to mine frequent ngrams (normal by default)."""
    bigrams = Counter()
    trigrams = Counter()
    for seq in seqs_df["row_ids"]:  # assumes a list of row_ids per session
        for i in range(len(seq)-1):   bigrams[tuple(seq[i:i+2])]  += 1
        for i in range(len(seq)-2):   trigrams[tuple(seq[i:i+3])] += 1
    freq2 = {k:v for k,v in bigrams.items() if v>=min_bigram}
    freq3 = {k:v for k,v in trigrams.items() if v>=min_trigram}
    return freq2, freq3
def build_semantic_index_transformer(seqs_df, db_path="logdb.sqlite",
                                     model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     batch_size=128):
    import sqlite3, pickle
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    # seqs_df needs columns: ['seq_key','label','text']
    texts  = seqs_df["text"].tolist()
    keys   = seqs_df["seq_key"].tolist()
    labels = seqs_df["label"].astype(int).tolist()

    embs = []
    for i in range(0, len(texts), batch_size):
        embs.extend(model.encode(texts[i:i+batch_size], show_progress_bar=False, convert_to_numpy=True))

    conn = sqlite3.connect(db_path); c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS semantic_index(
                   seq_key TEXT PRIMARY KEY, label INT, emb BLOB)""")
    c.execute("DELETE FROM semantic_index")
    rows = [(keys[i], labels[i], pickle.dumps(embs[i], protocol=4)) for i in range(len(keys))]
    c.executemany("INSERT INTO semantic_index VALUES (?,?,?)", rows)
    conn.commit(); conn.close()
    print(f"[SEM] wrote {len(rows)} transformer embeddings to {db_path}")

def kb_build_semantic_index(seqs_df, text_col="text", label_col="label", db_con=None):
    """Simple TF-IDF embeddings per session; store in KB."""
    corpus = seqs_df[text_col].tolist()
    labels = seqs_df[label_col].tolist()
    keys   = seqs_df["session_key"].tolist()
    vect = TfidfVectorizer(min_df=2, max_df=0.9)
    X = vect.fit_transform(corpus).astype("float32").toarray()  # dense for simplicity
    for k, y, row in zip(keys, labels, X):
        kb_upsert_semantic(db_con, k, y, row)
    db_con.commit()
    return vect  # return vectorizer so you can reuse if needed

def kb_detect_dynamic_for_seq(seq_row_ids, freq2_set, freq3_set, neg_tokens=("error","exception","fail"), window=3, thresh=2, raw_text=""):
    """Count rare ngrams + presence of negative hints in windowed scan."""
    rare_hits = 0
    for i in range(len(seq_row_ids)-1):
        if tuple(seq_row_ids[i:i+2]) not in freq2_set: rare_hits += 1
    for i in range(len(seq_row_ids)-2):
        if tuple(seq_row_ids[i:i+3]) not in freq3_set: rare_hits += 1
    neg_hit = any(tok in raw_text.lower() for tok in neg_tokens)
    is_high = int( (rare_hits >= thresh) and neg_hit )
    return rare_hits, is_high


def ensure_discrete_view(db_path="logdb.sqlite"):
    import sqlite3
    con = sqlite3.connect(db_path); c = con.cursor()
    c.execute("""
    CREATE VIEW IF NOT EXISTS discrete_patterns AS
    SELECT
      t.event_id AS event_id,
      CASE
        WHEN COALESCE(n0.cnt,0)=0 AND COALESCE(n1.cnt,0)>0 THEN 'anomalous'
        ELSE 'normal'
      END AS status,
      COALESCE(n0.cnt,0) + COALESCE(n1.cnt,0) AS support
    FROM templates t
    LEFT JOIN unigrams_by_label n0 ON t.row_id = n0.row_id AND n0.label = 0
    LEFT JOIN unigrams_by_label n1 ON t.row_id = n1.row_id AND n1.label = 1
    """)
    con.commit(); con.close()

def load_semantic_index(db_path="logdb.sqlite"):
    import sqlite3, pickle, numpy as np
    conn = sqlite3.connect(db_path); c = conn.cursor()
    rows = list(c.execute("SELECT seq_key, label, emb FROM semantic_index"))
    keys, labels, embs = [], [], []
    for k,l,b in rows:
        keys.append(k); labels.append(int(l)); embs.append(pickle.loads(b))
    conn.close()
    return keys, np.array(labels), np.vstack(embs)

def fit_nn_index(X, n_neighbors=5):
    from sklearn.neighbors import NearestNeighbors
    nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    nn_index.fit(X)
    return nn_index

def retrieve_neighbors(nn_index, X_all, keys_all, q_emb, k=5):
    dist, idx = nn_index.kneighbors(q_emb.reshape(1,-1), n_neighbors=k)
    return [(keys_all[i], float(1 - dist[0][j])) for j, i in enumerate(idx[0])]

def aggregate_neighbor_rules(neighbor_keys, rules_by_seq, topn=3):
    from collections import Counter
    bag = Counter()
    for k, _score in neighbor_keys:
        for r in rules_by_seq.get(k, []):
            bag[r] += 1
    return [r for r,_ in bag.most_common(topn)]

def kb_lookup_cached(con, pattern_key, pattern_type):
    con.execute("""CREATE TABLE IF NOT EXISTS judgments(
                       pattern_key TEXT, pattern_type TEXT, status INT, explanation TEXT,
                       PRIMARY KEY(pattern_key, pattern_type))""")
    return con.execute("SELECT status, explanation FROM judgments WHERE pattern_key=? AND pattern_type=?",
                       (pattern_key, pattern_type)).fetchone()

def kb_set_judgment(con, pattern_key, pattern_type, status, explanation):
    con.execute("INSERT OR REPLACE INTO judgments VALUES (?,?,?,?)",
                (pattern_key, pattern_type, int(status), str(explanation)))
    con.commit()

def explain_with_cache(con, pattern_key, pattern_type, llm_explain_fn):
    cached = kb_lookup_cached(con, pattern_key, pattern_type)
    if cached:
        return cached[0], cached[1], True
    status, explanation = llm_explain_fn(pattern_key, pattern_type)
    kb_set_judgment(con, pattern_key, pattern_type, status, explanation)
    return status, explanation, False
