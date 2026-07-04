from logsable.common_imports import *

class WindowDataset(Dataset):
    def __init__(self, X_ids: np.ndarray, y: np.ndarray, blk_ids: np.ndarray,
                 llm_map: dict | None = None, default_llm: int = 0):
        assert len(X_ids) == len(y) == len(blk_ids)
        self.X_ids = X_ids
        self.y = y
        self.blk = blk_ids
        self.llm_map = llm_map or {}
        self.default_llm = int(default_llm)

    def __len__(self):
        return len(self.X_ids)

    def __getitem__(self, idx):
        x_ids = torch.from_numpy(self.X_ids[idx]).long()
        y = int(self.y[idx])
        blk = str(self.blk[idx])

        llm_y = int(self.llm_map.get(blk, self.default_llm))

        return x_ids, torch.tensor(y).long(), blk, torch.tensor(llm_y).long()

def _download_file(url: str, dest_path: str) -> None:
    """Download a file from url to dest_path. Works on Windows and macOS (no wget required)."""
    import urllib.request
    dest_path = os.path.abspath(dest_path)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, dest_path)


def ensure_data(data_dir="hdfs_data"):
    os.makedirs(data_dir, exist_ok=True)
    base_url = "https://zenodo.org/record/7439296/files"
    struct_path = os.path.join(data_dir, "HDFS_100k.log_structured.csv")
    label_path = os.path.join(data_dir, "HDFS.anomaly_label.csv")
    if not os.path.exists(struct_path):
        _download_file(f"{base_url}/HDFS_100k.log_structured.csv?download=1", struct_path)
    if not os.path.exists(label_path):
        _download_file(f"{base_url}/HDFS.anomaly_label.csv?download=1", label_path)



def load_hdfs_sequences(data_dir="hdfs_data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import pandas as pd, os
    struct_path = os.path.join(data_dir, "HDFS_100k.log_structured.csv")
    label_path  = os.path.join(data_dir, "HDFS.anomaly_label.csv")

    hdfs   = pd.read_csv(struct_path, engine="c", na_filter=False, memory_map=True)
    labels = pd.read_csv(label_path)
    hdfs["BlockId"] = hdfs["Content"].str.extract(r'(blk_-?\d+)')
    hdfs = hdfs.dropna(subset=["BlockId"])

    templates_df = hdfs[["EventId", "EventTemplate"]].drop_duplicates().reset_index(drop=True)
    eid2row = {eid: i for i, eid in enumerate(templates_df["EventId"].tolist())}

    seqs_df = (
        hdfs.groupby("BlockId")["EventId"]
        .apply(lambda s: [eid2row.get(e) for e in s if e in eid2row])
        .reset_index(name="EventSeq")
    )

    label_map = dict(zip(labels["BlockId"], (labels["Label"] == "Anomaly").astype(int)))
    seqs_df["Label"] = seqs_df["BlockId"].map(label_map).fillna(0).astype(int)

    return hdfs, templates_df, seqs_df


def make_windows_from_sequences(seqs_df: pd.DataFrame, window_size=50, stride=1,
                                filter_len_col: str | None = None,
                                content_col: str | None = None,
                                pad_short_sequences: bool = False):

    flen = filter_len_col if filter_len_col and filter_len_col in seqs_df.columns else "EventSeq"
    ccol = content_col if content_col and content_col in seqs_df.columns else "EventSeq"

    if not pad_short_sequences:
        seqs_df = seqs_df[seqs_df[flen].apply(lambda s: len(s or []) >= window_size)].reset_index(drop=True)

    X_win, y_cls, blk_ids = [], [], []
    for blk, seq, lbl in zip(seqs_df["BlockId"].values, seqs_df[ccol].values, seqs_df["Label"].values):
        seq = list(seq) if seq is not None else []
        L = len(seq)
        if L < window_size:
            if pad_short_sequences:
                # Pad to window_size so every session yields at least one window.
                # For empty (L=0): use token 0 - preserves anomaly blocks where content became empty after masking.
                pad_token = seq[-1] if seq else 0
                seq = seq + [pad_token] * (window_size - L)
                L = len(seq)
            else:
                continue  # skip empty or short (when pad_short_sequences=False)
        for i in range(0, L - window_size + 1, stride):   # inclusive end
            X_win.append(seq[i:i+window_size])
            y_cls.append(lbl)
            blk_ids.append(blk)
    X_win = np.array(X_win, np.int32)
    y_cls = np.array(y_cls, np.int32)
    blk_ids = np.array(blk_ids)  # array of strings ok in numpy
    return X_win, y_cls, blk_ids




def make_group_splits(
    X,
    y,
    groups,
    val_size,
    test_size,
    seed,
    stratify_groups=True,
    train_size=None,
):
    """
    Split windows by unique group ids (e.g., BlockId) so the same session
    never leaks across splits.

    train_size: if None, train fraction = 1 - val_size - test_size (legacy).
        If set, train/val/test are fractions of all sessions (need not sum to 1.0;
        remainder is unused).

    stratify_groups: if True, stratify by session-level label so anomalies
        are distributed across train/val/test. Uses max(y) per group as label.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    val_size = float(val_size)
    test_size = float(test_size)

    if train_size is not None:
        train_size = float(train_size)
        total = train_size + val_size + test_size
        if total > 1.0 + 1e-5:
            print(
                f"[split] warn: train+val+test={total:.4f} > 1.0; "
                f"val/test will use all of the non-train pool"
            )

    def _explicit_group_split(grp_idx, labels, tr_frac, va_frac, te_frac, stratify):
        """train/val/test as fractions of all groups; leftover groups unused."""
        strat = labels if stratify else None
        pool_frac = max(0.0, min(1.0, 1.0 - tr_frac))
        tr_idx, pool_idx = train_test_split(
            grp_idx, test_size=pool_frac, stratify=strat, random_state=seed
        )
        if va_frac + te_frac <= 0 or len(pool_idx) == 0:
            return tr_idx, np.array([], dtype=int), np.array([], dtype=int)
        pool_frac = max(pool_frac, 1e-9)
        used_in_pool = min(1.0, (va_frac + te_frac) / pool_frac)
        pool_strat = labels[pool_idx] if stratify and np.unique(labels[pool_idx]).size >= 2 else None
        if used_in_pool < 1.0 - 1e-9:
            pool_used_idx, _ = train_test_split(
                pool_idx, test_size=1.0 - used_in_pool, stratify=pool_strat, random_state=seed
            )
        else:
            pool_used_idx = pool_idx
        if len(pool_used_idx) == 0:
            return tr_idx, np.array([], dtype=int), np.array([], dtype=int)
        te_frac_rel = te_frac / (va_frac + te_frac) if (va_frac + te_frac) > 0 else 0.0
        used_strat = labels[pool_used_idx] if stratify and np.unique(labels[pool_used_idx]).size >= 2 else None
        va_idx, te_idx = train_test_split(
            pool_used_idx, test_size=te_frac_rel, stratify=used_strat, random_state=seed
        )
        return tr_idx, va_idx, te_idx

    # Build group-level labels: group_label = max(window_label in group)
    grp2lbl = {}
    for i, (g, lbl) in enumerate(zip(groups, y)):
        g = str(g)
        grp2lbl[g] = max(grp2lbl.get(g, 0), int(lbl))

    unique_groups = np.array(list(grp2lbl.keys()))
    group_labels = np.array([grp2lbl[g] for g in unique_groups])

    # Group index -> window indices
    grp_to_win_idx = {}
    for i, g in enumerate(groups):
        g = str(g)
        grp_to_win_idx.setdefault(g, []).append(i)

    use_stratify = (
        stratify_groups
        and np.unique(group_labels).size >= 2
        and (group_labels == 1).sum() >= 1
        and (group_labels == 0).sum() >= 1
    )

    if use_stratify:
        try:
            grp_idx = np.arange(len(unique_groups))
            if train_size is not None:
                tr_grp_idx, va_grp_idx, te_grp_idx = _explicit_group_split(
                    grp_idx, group_labels, train_size, val_size, test_size, use_stratify
                )
            else:
                # test first, then val from remainder; train = rest
                trv_grp_idx, te_grp_idx = train_test_split(
                    grp_idx, test_size=test_size, stratify=group_labels, random_state=seed
                )
                trv_labels = group_labels[trv_grp_idx]
                val_frac = val_size / (1.0 - test_size)
                tr_grp_idx, va_grp_idx = train_test_split(
                    trv_grp_idx, test_size=val_frac, stratify=trv_labels, random_state=seed
                )
            tr_groups = set(unique_groups[tr_grp_idx])
            va_groups = set(unique_groups[va_grp_idx])
            te_groups = set(unique_groups[te_grp_idx])
        except ValueError:
            use_stratify = False

    if not use_stratify:
        # non-stratified GroupShuffleSplit
        all_idx = np.arange(len(y))
        if train_size is not None:
            # Map groups to windows via explicit session fractions
            grp_idx = np.arange(len(unique_groups))
            tr_gi, va_gi, te_gi = _explicit_group_split(
                grp_idx, group_labels, train_size, val_size, test_size, False
            )
            tr_groups = set(unique_groups[tr_gi])
            va_groups = set(unique_groups[va_gi])
            te_groups = set(unique_groups[te_gi])
            tr_idx = np.array([i for g in tr_groups for i in grp_to_win_idx.get(g, [])])
            va_idx = np.array([i for g in va_groups for i in grp_to_win_idx.get(g, [])])
            te_idx = np.array([i for g in te_groups for i in grp_to_win_idx.get(g, [])])
            return (X[tr_idx], y[tr_idx], groups[tr_idx]), (X[va_idx], y[va_idx], groups[va_idx]), (X[te_idx], y[te_idx], groups[te_idx])
        else:
            gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
            trv_idx, te_idx = next(gss1.split(all_idx, y, groups=groups))
            gss2 = GroupShuffleSplit(
                n_splits=1,
                test_size=val_size / (1.0 - test_size),
                random_state=seed,
            )
            tr_idx, va_idx = next(gss2.split(trv_idx, y[trv_idx], groups=groups[trv_idx]))
        TR = (X[tr_idx], y[tr_idx], groups[tr_idx])
        VA = (X[va_idx], y[va_idx], groups[va_idx])
        TE = (X[te_idx], y[te_idx], groups[te_idx])
        return TR, VA, TE

    # Map group membership to window indices
    tr_idx = np.array([i for g in tr_groups for i in grp_to_win_idx.get(g, [])])
    va_idx = np.array([i for g in va_groups for i in grp_to_win_idx.get(g, [])])
    te_idx = np.array([i for g in te_groups for i in grp_to_win_idx.get(g, [])])

    TR = (X[tr_idx], y[tr_idx], groups[tr_idx])
    VA = (X[va_idx], y[va_idx], groups[va_idx])
    TE = (X[te_idx], y[te_idx], groups[te_idx])
    return TR, VA, TE


def load_bgl_sequences(data_dir="bgl_data") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import os, pandas as pd

    struct_path = os.path.join(data_dir, "BGL_2k.log_structured.csv")
    tmpl_path   = os.path.join(data_dir, "BGL_2k.log_templates.csv")
    if not (os.path.exists(struct_path) and os.path.exists(tmpl_path)):
        raise FileNotFoundError(
            f"BGL files not found under {data_dir}. Expected:\n"
            f"  - {os.path.basename(struct_path)}\n"
            f"  - {os.path.basename(tmpl_path)}"
        )

    df = pd.read_csv(struct_path, engine="c", na_filter=False, memory_map=True)
    # Templates from structured to keep alignment with the actual run
    templates_df = df[["EventId", "EventTemplate"]].drop_duplicates().reset_index(drop=True)
    eid2row = {eid: i for i, eid in enumerate(templates_df["EventId"].tolist())}

    key_col = "Node" if "Node" in df.columns else ("NodeRepeat" if "NodeRepeat" in df.columns else None)
    if key_col is None:
        raise KeyError("Neither 'Node' nor 'NodeRepeat' columns were found in BGL structured CSV.")

    df["SeqKey"] = df[key_col].astype(str)
    # Preserve timestamp order within each session so EventSeq is chronological (for inference)
    if "Timestamp" in df.columns:
        _ts = pd.to_numeric(df["Timestamp"], errors="coerce")
        df = df.assign(_ts_num=_ts).sort_values(by=["SeqKey", "_ts_num"]).drop(columns=["_ts_num"]).reset_index(drop=True)
    else:
        df = df.sort_values(by="SeqKey").reset_index(drop=True)

    # Build sequences; order within group = row order (timestamp order when Timestamp exists)
    seqs_df = (
        df.groupby("SeqKey", sort=False)["EventId"]
          .apply(lambda s: [eid2row.get(e) for e in s if e in eid2row])
          .reset_index(name="EventSeq")
          .rename(columns={"SeqKey": "BlockId"})
    )

    # Label policy: '-' → normal(0); anything else → anomaly(1).
    if "Label" in df.columns:
        df["_lbl"] = (df["Label"].astype(str) != "-").astype(int)
    else:
            # infer label from severity levels when Label column is missing
        df["_lbl"] = df.get("Level", pd.Series([""]*len(df))).astype(str).str.contains(r"(ERROR|SEVERE|FATAL)", case=False, regex=True).astype(int)

    labels = (df.groupby("SeqKey")["_lbl"].max()
                .reset_index()
                .rename(columns={"SeqKey": "BlockId", "_lbl": "Label"}))

    seqs_df = seqs_df.merge(labels, on="BlockId", how="left").fillna({"Label": 0})
    seqs_df["Label"] = seqs_df["Label"].astype(int)

    return df, templates_df, seqs_df


def load_liberty_sequences(
    data_dir: str,
    bucket_sec: int = 600,
    filename: str = "liberty_150k.csv",
    event_id_source: str = "log",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    import os
    import pandas as pd

    csv_path = os.path.join(data_dir, filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Liberty file not found: {csv_path}. Expected {filename}"
        )

    raw_df = pd.read_csv(csv_path, engine="c", na_filter=False, memory_map=True)

    # Label already 0/1
    raw_df["Label"] = raw_df["Label"].astype(int)

    src = str(event_id_source).strip().lower()
    if src == "label_token":
        tokens_unique = (
            raw_df["LabelToken"].astype(str).drop_duplicates().sort_values().reset_index(drop=True)
        )
        templates_df = pd.DataFrame({
            "EventId": [f"E{i}" for i in range(len(tokens_unique))],
            "EventTemplate": tokens_unique.tolist(),
        })
        token2eid = {str(t): f"E{i}" for i, t in enumerate(tokens_unique)}
        raw_df["EventId"] = raw_df["LabelToken"].astype(str).map(token2eid)
    elif src == "log":
        tokens_unique = raw_df["log"].astype(str).drop_duplicates().reset_index(drop=True)
        templates_df = pd.DataFrame({
            "EventId": [f"E{i}" for i in range(len(tokens_unique))],
            "EventTemplate": tokens_unique.tolist(),
        })
        token2eid = {str(t): f"E{i}" for i, t in enumerate(tokens_unique)}
        raw_df["EventId"] = raw_df["log"].astype(str).map(token2eid)
    else:
        raise ValueError(f"Liberty event_id_source must be 'log' or 'label_token', got {event_id_source!r}")

    eid2rid = {eid: i for i, eid in enumerate(templates_df["EventId"].tolist())}

    # Timestamp from ts column
    raw_df["_ts"] = pd.to_numeric(raw_df["ts"], errors="coerce").fillna(0).astype(int)

    # Host from log: "2005.09.02 ln215 Sep 2 ..." -> ln215 (index 1)
    def _host_from_log(log_str):
        parts = str(log_str).strip().split(None, 3)
        return str(parts[1]) if len(parts) >= 2 else "GLOBAL"

    raw_df["_host"] = raw_df["log"].apply(_host_from_log)

    # Session key: time-bucketing per host
    raw_df["_seq_key"] = raw_df["_host"] + "::" + (raw_df["_ts"] // bucket_sec).astype(str)

    raw_df = raw_df.sort_values(by=["_seq_key", "_ts"]).reset_index(drop=True)

    rows = []
    for sess, g in raw_df.groupby("_seq_key", sort=False):
        ev_ids = g["EventId"].astype(str).tolist()
        ev_row_ids = [eid2rid[e] for e in ev_ids if e in eid2rid]
        y = int((g["Label"] == 1).any())
        rows.append({
            "BlockId": str(sess),
            "seq_key": str(sess),
            "Label": y,
            "EventSeq": ev_row_ids,
        })
    seqs_df = pd.DataFrame(rows)

    return raw_df, templates_df, seqs_df


def load_sequences(data_dir: str, dataset: str = "HDFS", data_cfg: dict | None = None):
    ds = str(dataset).upper()
    data_cfg = data_cfg or {}
    if ds == "HDFS":
        return load_hdfs_sequences(data_dir)
    elif ds == "BGL":
        return load_bgl_sequences(data_dir)
    elif ds in ("THUNDERBIRD", "TBD", "TBIRD"):
        bucket_sec = int(data_cfg.get("thunderbird_bucket_sec", 600))
        chunk_size = int(data_cfg.get("thunderbird_chunk_size", 300))
        raw_df, templates_df, seqs_df = load_thunderbird_sequences(
            data_dir, bucket_sec=bucket_sec, chunk_size=chunk_size
        )
        return raw_df, templates_df, seqs_df
    elif ds == "SPIRIT":
        bucket_sec = int(data_cfg.get("spirit_bucket_sec", 600))
        chunk_size = int(data_cfg.get("spirit_chunk_size", 300))
        return load_spirit_sequences(data_dir, bucket_sec=bucket_sec, chunk_size=chunk_size)
    elif ds == "LIBERTY":
        bucket_sec = int(data_cfg.get("liberty_bucket_sec", 600))
        filename = str(data_cfg.get("liberty_filename", "liberty_150k.csv"))
        return load_liberty_sequences(data_dir, bucket_sec=bucket_sec, filename=filename)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

