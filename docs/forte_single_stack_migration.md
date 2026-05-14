# FORTE Single-Stack Migration

## Decision

`FORTE/` is the only canonical controller stack.

## What Changed

- `force_coral/controllers/run_forte.py` is now a compatibility shim that delegates to `FORTE.run_forte.run_forte`.
- Active controller logic lives in:
  - `FORTE/run_forte.py`
  - `FORTE/env.py`
  - `FORTE/forte_wrapper.py`
  - `FORTE/mppi.py`
  - `FORTE/semantic.py`
  - `FORTE/estimator.py`
  - `FORTE/vlm.py`

## Migration Notes

- Existing imports of `force_coral.controllers.run_forte.run_forte` continue to work but emit a deprecation warning.
- New development should only target `FORTE/`.
- Runtime contract remains stable: wrapper `configure_runtime()` -> MPPI `compute_control()` -> wrapper `rollout_cost()`.
