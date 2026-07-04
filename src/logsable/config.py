
from __future__ import annotations
from logsable.common_imports import *
from dataclasses import dataclass, field
from typing import Any, Dict

@dataclass
class Config:
    raw: Dict[str, Any]
    run: Dict[str, Any]
    data: Dict[str, Any]
    model: Dict[str, Any]
    train: Dict[str, Any]
    eval: Dict[str, Any]
    cluster: Dict[str, Any] = field(default_factory=dict)
    logrobust: Dict[str, Any] = field(default_factory=dict)
    distill: Dict[str, Any] = field(default_factory=dict)
    llm: Dict[str, Any] = field(default_factory=dict)

    @property
    def device(self) -> str:
        return resolve_torch_device(self.run.get("device", "auto"))


def _mps_available() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_built() and mps.is_available())


def resolve_torch_device(requested: str = "auto") -> str:
    """
    Pick PyTorch device string: cuda:0 > mps > cpu when requested is 'auto'.
    Explicit values: 'cpu', 'mps', 'cuda', 'cuda:0', etc.
    """
    d = str(requested or "auto").strip().lower()
    if d == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if _mps_available():
            return "mps"
        return "cpu"
    if d in ("cuda", "gpu"):
        return "cuda:0" if torch.cuda.is_available() else (
            "mps" if _mps_available() else "cpu"
        )
    if d == "mps" and not _mps_available():
        print("[device] WARNING: mps requested but not available; falling back to cpu")
        return "cpu"
    return requested if requested != "mps" else "mps"


def describe_torch_devices() -> Dict[str, Any]:
    """Runtime device availability summary (CUDA / MPS / CPU)."""
    info = {
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": bool(
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_built()
        ),
        "mps_available": _mps_available(),
    }
    try:
        if info["mps_available"]:
            t = torch.zeros(1, device="mps")
            info["mps_tensor_ok"] = str(t.device)
    except Exception as e:
        info["mps_tensor_ok"] = f"failed: {e}"
    info["resolved_auto"] = resolve_torch_device("auto")
    return info


def _as_float(x):
    # robust: handles numbers, strings, scientific notation
    if isinstance(x, (float, int)):
        return float(x)
    try:
        return float(str(x))
    except Exception:
        raise ValueError(f"Expected float-like value, got {x!r}")

def _as_int(x):
    if isinstance(x, int):
        return x
    try:
        return int(str(x))
    except Exception:
        raise ValueError(f"Expected int-like value, got {x!r}")
    
def _as_bool(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "y", "t")

def load_config(path: str) -> Config:
    with open(path, "r") as f:
        y = yaml.safe_load(f)

    y.setdefault("cluster", {})
    y.setdefault("logrobust", {})
    y.setdefault("distill", {})
    y.setdefault("llm", {})
    m = y.get("model", {})
    m.setdefault("logbert", {})

    # ---- cast TRAIN fields
    y["train"]["lr"] = _as_float(os.getenv("LR", y["train"]["lr"]))
    y["train"]["weight_decay"] = _as_float(os.getenv("WEIGHT_DECAY", y["train"].get("weight_decay", 0.0)))
    y["train"]["epochs"] = _as_int(os.getenv("EPOCHS", y["train"]["epochs"]))
    y["train"]["batch_size"] = _as_int(os.getenv("BATCH_SIZE", y["train"]["batch_size"]))
    y["train"]["num_workers"] = _as_int(os.getenv("NUM_WORKERS", y["train"].get("num_workers", 0)))
    pin = os.getenv("PIN_MEMORY", y["train"].get("pin_memory", False))
    y["train"]["pin_memory"] = _as_bool(pin)

    # ---- cast DATA fields
    y["data"]["window"] = _as_int(os.getenv("WINDOW", y["data"]["window"]))
    y["data"]["stride"] = _as_int(os.getenv("STRIDE", y["data"]["stride"]))
    y["data"]["val_size"] = _as_float(os.getenv("VAL_SIZE", y["data"]["val_size"]))
    y["data"]["test_size"] = _as_float(os.getenv("TEST_SIZE", y["data"]["test_size"]))
    _ts = os.getenv("TRAIN_SIZE", y["data"].get("train_size"))
    if _ts is None or str(_ts).strip().lower() in ("", "null", "none"):
        y["data"]["train_size"] = None
    else:
        y["data"]["train_size"] = _as_float(_ts)
    y["data"]["data_dir"] = os.getenv("DATA_DIR", y["data"]["data_dir"])

    # ---- cast RUN fields
    y["run"]["seed"] = _as_int(os.getenv("SEED", y["run"]["seed"]))
    y["run"]["device"] = os.getenv("DEVICE", y["run"].get("device", "auto"))

    # ---- CLUSTER defaults
    c = y["cluster"]
    if "CLUSTER_ENABLED" in os.environ:
        c["enabled"] = _as_bool(os.environ["CLUSTER_ENABLED"])
    c.setdefault("enabled", False)
    c.setdefault("source", "all")
    c.setdefault("vectorizer", "tfidf-eid")
    c.setdefault("min_cluster_size", 10)
    c.setdefault("min_samples", None)
    c.setdefault("metric", "cosine")
    c.setdefault("select_k", 3)

    try: c["min_cluster_size"] = _as_int(c["min_cluster_size"])
    except Exception: c["min_cluster_size"] = 10
    if c["min_samples"] is not None:
        try: c["min_samples"] = _as_int(c["min_samples"])
        except Exception: c["min_samples"] = None
    try: c["select_k"] = _as_int(c["select_k"])
    except Exception: c["select_k"] = 3

    # ---- DISTILL defaults
    d = y["distill"]
    d.setdefault("enabled", False)
    d.setdefault("lambda", 0.3)
    d.setdefault("source", "session")
    d.setdefault("threshold", 0.5)
    d.setdefault("min_conf", 0.0)
    d.setdefault("use_soft", False)

    # ---- LLM defaults (live OpenAI + KB cache)
    llm = y["llm"]
    llm.setdefault("enabled", True)
    llm.setdefault("model", os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini-2024-07-18"))
    llm.setdefault("session_preview_k", 20)
    llm.setdefault("max_sessions", None)
    llm.setdefault("max_clusters", None)
    llm.setdefault("sleep_s", 0.0)
    llm.setdefault("use_cache", True)
    llm.setdefault("cross_run_cache", True)
    llm.setdefault("export_prompts", False)

    return Config(
        raw=y,
        run=y["run"],
        data=y["data"],
        model=y["model"],
        train=y["train"],
        eval=y["eval"],
        cluster=y["cluster"],
        logrobust=y.get("logrobust", {}),
        distill=y.get("distill", {}),
        llm=y.get("llm", {}),
    )


