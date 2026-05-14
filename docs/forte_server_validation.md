# FORTE Server Validation Protocol

Run this on the server (no local execution).

## 1) Pipeline Parity Check

Run a short GT rollout and inspect logs:

```bash
python -m FORTE.run_forte --steps 40 --samples 64 --horizon 12 --no-video
```

Verify in `summary.json` / step logs:

- `observed_pose_source` exists each step.
- `selected_contact_hypothesis` and `top_contact_hypotheses` exist each step.
- `measured_force_task_pre` and `measured_force_task_post` both exist.
- `cost_total == task + energy + force_upper + force_lower` from logged fields.

## 2) FoundationPose Inner-Update Check

```bash
python -m FORTE.run_forte --pose-source foundationpose --steps 40 --samples 64 --horizon 12 --no-video
```

Acceptance:

- Most steps report `observed_pose_source=foundationpose`, otherwise `ground_truth_fallback`.
- No runtime crash when FP is unavailable; fallback path works.

## 3) Control Invariant Checks

Confirm from logs/code:

- Worker and execution both use same `action_multiplier`.
- `sync_robot_from_real` then `update_inner_from_pose` order is preserved.
- Monitor uses post-action force (`measured_force_task_post`).

## 4) Ablations

Run three conditions and compare:

1. Baseline (new contact inference enabled)
2. Single-contact baseline (disable hypothesis perturbation)
3. Fallback disabled (force always aggressive action scale)

Metrics:

- success rate
- median steps to reach target height
- mean wall-normal force in lift phase
- lateral drift at terminal state

## 5) Output Artifacts

Collect and archive:

- `run_config.json`
- `summary.json`
- step-level records (`records.jsonl` or equivalent artifact output)
- phase plots from `ArtifactManager`
