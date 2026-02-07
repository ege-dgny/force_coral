"""
Utilities to load initialization "bundles" for tasks: (object overrides + MuJoCo state).

This enables per-initialization geometry / mass configuration together with a
matching simulator state, so evaluation code can be concise and reproducible.

Expected files per task (under force_coral/data/init_files/<problem_folder>):
- <task>.pruned_init              : list[flattened_mj_state]
- <task>.pruned_init.meta.json    : metadata with overrides_per_state, placements

Primary API:
- make_env_for_init(task_suite, task_id, init_idx, base_env_args) -> env, obs
  Constructs OffScreenRenderEnv with overrides for init_idx, sets simulator
  state from .pruned_init, returns env and the initial observations.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import torch

import force_coral
from libero.libero.envs.env_wrapper import OffScreenRenderEnv


def _paths_for_task(task_suite, task_id: int) -> Dict[str, str]:
    """Resolve BDDL path, init-states path, and sidecar meta path for a task."""
    bddl_path = task_suite.get_task_bddl_file_path(task_id)

    task = task_suite.get_task(task_id)
    init_root = force_coral.get_data_path("init_states")
    init_dir = os.path.join(init_root, task.problem_folder)
    init_path = os.path.join(init_dir, task.init_states_file)
    meta_path = init_path + ".meta.json"
    return {"bddl": bddl_path, "init": init_path, "meta": meta_path}


def _load_states(init_path: str) -> List[Any]:
    return torch.load(init_path, weights_only=False, map_location="cpu")


def _load_meta(meta_path: str) -> Dict[str, Any]:
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r") as f:
        return json.load(f)


def _overrides_for_idx(meta: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if "overrides_per_state" in meta:
        ops = meta["overrides_per_state"]
        if 0 <= idx < len(ops):
            return ops[idx] or {}
    if "overrides" in meta and isinstance(meta["overrides"], dict):
        return meta["overrides"]
    return {}


def get_num_inits(task_suite, task_id: int) -> int:
    """Return how many initialization states are stored for this task."""
    paths = _paths_for_task(task_suite, task_id)
    states = _load_states(paths["init"])
    return len(states)


def load_init_bundle(task_suite, task_id: int, init_idx: int) -> Tuple[Dict[str, Any], Any]:
    """Load (overrides, state) for the given task and init index."""
    paths = _paths_for_task(task_suite, task_id)
    states = _load_states(paths["init"])
    assert 0 <= init_idx < len(states), (
        f"init_idx {init_idx} out of range (n={len(states)}) for {paths['init']}"
    )
    meta = _load_meta(paths["meta"])
    overrides = _overrides_for_idx(meta, init_idx)
    return overrides, states[init_idx]


def make_env_for_init(
    task_suite,
    task_id: int,
    init_idx: int,
    base_env_args: Dict[str, Any] | None = None,
):
    if base_env_args is None:
        base_env_args = {}

    paths = _paths_for_task(task_suite, task_id)
    overrides, state = load_init_bundle(task_suite, task_id, init_idx)

    env_kwargs = {
        "bddl_file_name": paths["bddl"],
        **base_env_args,
    }
    if overrides:
        env_kwargs["object_overrides"] = overrides
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(state)
    return env, obs


def iter_envs_for_task(
    task_suite,
    task_id: int,
    base_env_args: Dict[str, Any] | None = None,
):
    """Generator yielding (env, obs, idx) for each stored initialization."""
    n = get_num_inits(task_suite, task_id)
    for i in range(n):
        env, obs = make_env_for_init(task_suite, task_id, i, base_env_args)
        yield env, obs, i


# ── Name-based helpers (no suite instance required) ───────────────────────

def _paths_for_task_name(problem_folder: str, task_name: str) -> Dict[str, str]:
    bddl = os.path.join(
        force_coral.get_data_path("bddl_files"),
        problem_folder,
        f"{task_name}.bddl",
    )
    init_dir = os.path.join(
        force_coral.get_data_path("init_states"),
        problem_folder,
    )
    init_path = os.path.join(init_dir, f"{task_name}.pruned_init")
    meta_path = init_path + ".meta.json"
    return {"bddl": bddl, "init": init_path, "meta": meta_path}


def load_init_bundle_by_name(
    problem_folder: str, task_name: str, init_idx: int
) -> Tuple[Dict[str, Any], Any]:
    paths = _paths_for_task_name(problem_folder, task_name)
    states = _load_states(paths["init"])
    assert 0 <= init_idx < len(states), (
        f"init_idx {init_idx} out of range (n={len(states)}) for {paths['init']}"
    )
    meta = _load_meta(paths["meta"])
    overrides = _overrides_for_idx(meta, init_idx)
    return overrides, states[init_idx]


def make_env_for_task_name(
    problem_folder: str,
    task_name: str,
    init_idx: int,
    base_env_args: Dict[str, Any] | None = None,
):
    if base_env_args is None:
        base_env_args = {}
    paths = _paths_for_task_name(problem_folder, task_name)
    overrides, state = load_init_bundle_by_name(problem_folder, task_name, init_idx)
    env_kwargs = {
        "bddl_file_name": paths["bddl"],
        **base_env_args,
    }
    if overrides:
        env_kwargs["object_overrides"] = overrides
    env = OffScreenRenderEnv(**env_kwargs)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(state)
    return env, obs
