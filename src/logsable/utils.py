from logsable.common_imports import *


def seed_everything(seed: int = 43):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_banner(cfg):
    from logsable.config import describe_torch_devices
    dev_info = describe_torch_devices()
    resolved = cfg.device
    print("\n=== DEVICE ===")
    print(f"  requested: {cfg.run.get('device', 'auto')}")
    print(f"  resolved:  {resolved}")
    print(f"  PyTorch {dev_info['pytorch_version']} | CUDA={dev_info['cuda_available']} "
          f"MPS built={dev_info['mps_built']} available={dev_info['mps_available']}")
    if dev_info.get("mps_tensor_ok"):
        print(f"  MPS smoke test: {dev_info['mps_tensor_ok']}")
    print("=== RUN CONFIG ===")
    # concise one-liner
    print(json.dumps({
        "run":   {**cfg.run, "device_resolved": resolved},
        "data":  {k: v for k, v in cfg.data.items() if k not in ("_paths",)},
        "model": {"name": cfg.model["name"], "vocab_from": cfg.model.get("vocab_from")},
        "train": cfg.train,
        "eval":  cfg.eval
    }, indent=2, default=str))
    print("==================\n")