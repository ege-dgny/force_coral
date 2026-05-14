"""Compatibility shim for the canonical FORTE controller.

FORTE implementation lives in `FORTE/run_forte.py`.
This module remains only for backward import compatibility.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

from FORTE.run_forte import TASK_NAME, run_forte as _canonical_run_forte


def run_forte(
    *,
    task_name: str = TASK_NAME,
    init_idx: int = 0,
    problem_folder: str = "my_suite",
    num_steps: int = 150,
    use_vlm: bool = False,
    show: bool = False,
    save_video: bool = True,
    device: str = "auto",
    pose_source: str = "ground_truth",
    eta: float = 0.005,
    min_eigenvalue: float = 0.1,
    horizon: int = 10,
    num_samples: int = 64,
    noise_scale: float = 1.0,
    review_interval: int = 10,
    num_workers: Optional[int] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    warnings.warn(
        "force_coral.controllers.run_forte is deprecated. "
        "Use FORTE.run_forte.run_forte instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    if device != "auto":
        warnings.warn(
            "Parameter 'device' is ignored by the canonical FORTE stack.",
            stacklevel=2,
        )
    if kwargs:
        warnings.warn(f"Ignoring unsupported legacy kwargs: {sorted(kwargs.keys())}", stacklevel=2)

    return _canonical_run_forte(
        task_name=task_name,
        init_idx=init_idx,
        problem_folder=problem_folder,
        num_steps=num_steps,
        use_vlm=use_vlm,
        show=show,
        save_video=save_video,
        pose_source=pose_source,
        eta=eta,
        min_eigenvalue=min_eigenvalue,
        horizon=horizon,
        num_samples=num_samples,
        noise_scale=noise_scale,
        review_interval=review_interval,
        num_workers=num_workers,
    )


__all__ = ["TASK_NAME", "run_forte"]


if __name__ == "__main__":
    run_forte()
