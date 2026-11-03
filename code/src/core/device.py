"""
Central device resolver for GPU-agnostic operation.

Priority order:
  1. CUDA_VISIBLE_DEVICES env var — if set, the process sees ONLY those GPUs.
     CUDA renumbers them from 0, so we always use cuda:0 in this case.
     This is the hard isolation guarantee: other physical GPUs are invisible.
  2. CUDA_DEVICE env var / config.DEFAULT_CUDA_DEVICE (default: 1)
  3. "cpu" if CUDA is not available at all

To lock to physical GPU 1 and prevent any access to GPUs 0 and 2:
    export CUDA_VISIBLE_DEVICES=1
    python scripts/run_idiom_rag.py ...

For Ollama, start it in the same environment:
    CUDA_VISIBLE_DEVICES=1 ollama serve

Usage:
    from src.core.device import get_device
    embeddings = HuggingFaceEmbeddings(model_kwargs={"device": get_device()})
"""

import os
import config


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_device() -> str:
    """
    Return a torch-compatible device string: "cuda:N" or "cpu".

    If CUDA_VISIBLE_DEVICES is set, this process can only see the listed
    physical GPUs — CUDA renumbers them starting at 0, so we return "cuda:0".
    This is the safest isolation: the kernel will refuse access to any other
    device regardless of what the code requests.
    """
    if not _cuda_available():
        return "cpu"

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        # CUDA_VISIBLE_DEVICES is set — only those GPUs exist in this process.
        # The first (and typically only) visible device is always cuda:0.
        return "cuda:0"

    # No CUDA_VISIBLE_DEVICES restriction — use the configured device index.
    idx = _configured_index()
    try:
        import torch
        n = torch.cuda.device_count()
        if idx >= n:
            print(f"[Device] Requested GPU {idx} not available "
                  f"({n} device(s) found). Falling back to GPU 0.")
            idx = 0
    except Exception:
        idx = 0
    return f"cuda:{idx}"


def get_device_index() -> int:
    """Return the logical CUDA device index (0-based) as an int."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        return 0
    return _configured_index()


def list_available_devices() -> list[dict]:
    """Return a list of compute devices visible to this process."""
    devices = [{"label": "CPU (no GPU)", "value": "cpu"}]
    if not _cuda_available():
        return devices
    try:
        import torch
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            devices.append({"label": f"GPU {i} — {name}", "value": str(i)})
    except Exception:
        devices.append({"label": "GPU 0 (name unavailable)", "value": "0"})
    return devices


def _configured_index() -> int:
    """
    Read device index from SQLite preference or CUDA_DEVICE env var.
    Raises if neither is set — no silent fallback to GPU 0.
    """
    try:
        from src.db.sqlite_manager import get_setting, init_db
        init_db()
        stored = get_setting("cuda_device", default="")
        if stored:
            return int(stored)
    except Exception:
        pass

    if config.DEFAULT_CUDA_DEVICE is None:
        raise EnvironmentError(
            "[Device] No GPU selected. Set CUDA_VISIBLE_DEVICES=<n> (recommended) "
            "or CUDA_DEVICE=<n> in your environment before running."
        )
    return int(config.DEFAULT_CUDA_DEVICE)
