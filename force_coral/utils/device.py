"""Backend / device selection helpers for cpu, mps, and cuda."""

from __future__ import annotations

from typing import Optional


def resolve_device(preferred: Optional[str] = None) -> str:
    """Resolve the best available device for semantic/model components."""
    preferred = (preferred or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu"

    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        return "mps" if has_mps else "cpu"
    if preferred == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if has_mps:
        return "mps"
    return "cpu"
