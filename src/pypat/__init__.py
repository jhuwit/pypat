"""Python tools for pretrained activity transformer (PAT) models."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FineTuneResult",
    "PATConfig",
    "attention_to_timeline",
    "download_weights",
    "fine_tune_pat",
    "load_nhanes_weekly_accelerometry",
    "augment_all_weekly_cycles",
    "rotate_nhanes_weekly_accelerometry",
]


def __getattr__(name: str) -> Any:
    """Load optional package functionality only when it is requested."""
    if name in {"FineTuneResult", "fine_tune_pat"}:
        from .finetune import FineTuneResult, fine_tune_pat

        return {"FineTuneResult": FineTuneResult, "fine_tune_pat": fine_tune_pat}[name]
    if name == "PATConfig":
        from .model import PATConfig

        return PATConfig
    if name == "attention_to_timeline":
        from .explain import attention_to_timeline

        return attention_to_timeline
    if name == "download_weights":
        from .weights import download_weights

        return download_weights
    if name == "load_nhanes_weekly_accelerometry":
        from .datasets import load_nhanes_weekly_accelerometry

        return load_nhanes_weekly_accelerometry
    if name == "augment_all_weekly_cycles":
        from .datasets import augment_all_weekly_cycles

        return augment_all_weekly_cycles
    if name == "rotate_nhanes_weekly_accelerometry":
        from .datasets import rotate_nhanes_weekly_accelerometry

        return rotate_nhanes_weekly_accelerometry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
