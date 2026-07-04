from logsable.common_imports import *
import torch
import torch.nn.functional as F

def make_loaders(X_win, y_cls, blk_ids, batch_size, val_size, test_size, random_state):
    X_train, X_tmp, y_train, y_tmp, blk_train, blk_tmp = train_test_split(
        X_win, y_cls, blk_ids,
        test_size=(val_size + test_size),
        random_state=random_state, stratify=y_cls
    )
    rel_val = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, blk_val, blk_test = train_test_split(
        X_tmp, y_tmp, blk_tmp,
        test_size=(1 - rel_val),
        random_state=random_state, stratify=y_tmp
    )
    return (X_train, y_train, blk_train), (X_val, y_val, blk_val), (X_test, y_test, blk_test)


def dataset_stats(split_name, y):
    unique, counts = np.unique(y, return_counts=True)
    d = dict(zip(unique.tolist(), counts.tolist()))
    total = len(y)
    pos = d.get(1, 0); neg = d.get(0, 0)
    print(f"[{split_name}] windows={total}  neg={neg}  pos={pos}  pos_rate={pos/total if total>0 else 0:.3f}")

def class_weights_from(y):
    classes, counts = np.unique(y, return_counts=True)
    # If the training set contains only a single class, return None so callers
    # can choose not to pass a weight tensor to CrossEntropyLoss (which
    # requires a weight per class index).
    if len(classes) < 2:
        return None
    total = counts.sum()
    weights = torch.tensor([total / c for c in counts], dtype=torch.float32)
    # normalize so mean weight ~1 (preserve relative class importance)
    return weights / weights.sum() * len(classes)


def train_epoch(model, loader, criterion, optim, device="cpu", distill_cfg=None):
    model.train()
    total_loss = 0.0

    # distill config defaults
    distill_cfg = distill_cfg or {}
    distill_on = bool(distill_cfg.get("enabled", False))
    lam = float(distill_cfg.get("lambda", 0.3))         # how much to weight LLM loss
    use_soft = bool(distill_cfg.get("use_soft", False)) # later: soft labels/probs
    # (hard distillation only)

    for batch in loader:
        # Backward-compatible batch unpacking
        if len(batch) == 3:
            x, y, blk = batch
            llm_y = None
        else:
            x, y, blk, llm_y = batch

        x = x.to(device)
        y = y.to(device)
        if llm_y is not None:
            llm_y = llm_y.to(device)

        optim.zero_grad()
        logits = model(x)  # shape: [B, 2] (assuming binary)

        loss_sup = criterion(logits, y)

        # Distillation: add an auxiliary loss that makes model match LLM label
        if distill_on and (llm_y is not None) and (lam > 0):
            # hard distillation: treat LLM label as another target
            loss_llm = F.cross_entropy(logits, llm_y)
            loss = (1.0 - lam) * loss_sup + lam * loss_llm
        else:
            loss = loss_sup

        loss.backward()
        optim.step()
        total_loss += float(loss.item())

    return total_loss / max(1, len(loader))


def aggregate_to_session(y_true_win, p1_win, blk_ids, policy="mean", th=0.5, k=2):
    """Aggregate window scores to session/block labels.
    policy: 'mean' (avg score >= th), 'max' (max score >= th), 'kofn' (>= k windows above th).
    """
    sess_true, sess_sum, sess_cnt, sess_poswins, sess_max = {}, {}, {}, {}, {}
    for yt, p, b in zip(y_true_win, p1_win, blk_ids):
        sess_true[b] = max(sess_true.get(b, 0), int(yt))
        sess_sum[b]  = sess_sum.get(b, 0.0) + float(p)
        sess_cnt[b]  = sess_cnt.get(b, 0) + 1
        sess_max[b]  = max(sess_max.get(b, 0.0), float(p))
        if p >= th:
            sess_poswins[b] = sess_poswins.get(b, 0) + 1
    keys = list(sess_true.keys())
    y_true = np.array([sess_true[k] for k in keys], dtype=int)

    if policy == "kofn":
        wins  = np.array([sess_poswins.get(k, 0) for k in keys])
        y_pred = (wins >= k).astype(int)
    elif policy == "max":
        scores = np.array([sess_max.get(k, 0.0) for k in keys])
        y_pred = (scores >= th).astype(int)
    else:  # "mean"
        scores = np.array([sess_sum[k] / sess_cnt[k] for k in keys])
        y_pred = (scores >= th).astype(int)

    return keys, y_true, y_pred


def tune_session_threshold(y_true_win, p1_win, blk_ids, policy="mean", k=2):
    """Resolve session aggregation threshold on the validation split."""
    best_th, best_f1 = 0.5, -1.0
    for th in np.linspace(0.05, 0.95, 19):
        _, ys, yp = aggregate_to_session(y_true_win, p1_win, blk_ids, policy=policy, th=th, k=k)
        _, _, f1, _ = anomaly_f1(ys, yp)
        if f1 > best_f1:
            best_th, best_f1 = float(th), float(f1)
    return best_th

def collect_window_scores(model, loader, device="cpu", dropout_at_eval=False):
    model.train() if dropout_at_eval else model.eval()
    ys, ps, blks = [], [], []

    with torch.no_grad():
        for batch in loader:
            # Works for (X, y, blk) OR (X, y, blk, llm_y) OR even more later
            Xb, yb, blkb = batch[0], batch[1], batch[2]

            Xb = Xb.to(device)
            yb = yb.to(device)

            logits = model(Xb)
            prob = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()

            ys.append(yb.detach().cpu().numpy())
            ps.append(prob)

            # keep block ids as plain python strings
            if isinstance(blkb, (list, tuple)):
                blks.extend([str(x) for x in blkb])
            else:
                # if it's a tensor/np array
                blks.extend([str(x) for x in list(blkb)])

    y = np.concatenate(ys, axis=0)
    p = np.concatenate(ps, axis=0)
    b = np.array(blks, dtype=object)
    return y, p, b


def agg_session_preds(y_true_win, p1_win, blk_ids, policy="mean", th=0.5, k=2):
    sess_true, sess_sum, sess_cnt, sess_poswins = {}, {}, {}, {}
    for yt, p, b in zip(y_true_win, p1_win, blk_ids):
        sess_true[b] = max(sess_true.get(b, 0), int(yt))
        sess_sum[b]  = sess_sum.get(b, 0.0) + float(p)
        sess_cnt[b]  = sess_cnt.get(b, 0) + 1
        if p >= th:
            sess_poswins[b] = sess_poswins.get(b, 0) + 1
    keys = list(sess_true.keys())
    y_true = np.array([sess_true[k] for k in keys], dtype=int)
    if policy == "kofn":
        wins = np.array([sess_poswins.get(k, 0) for k in keys])
        y_pred = (wins >= k).astype(int)
    else:  # "mean"
        scores = np.array([sess_sum[k] / sess_cnt[k] for k in keys])
        y_pred = (scores >= th).astype(int)
    return keys, y_true, y_pred


def anomaly_f1(y_true, y_pred):
    """
    Precision/Recall/F1 for the anomaly class (label=1).
    """
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return p, r, f1, cm