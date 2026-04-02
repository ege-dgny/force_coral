# FORTE Phase 1 Design Log

## Research Anchors

- **Theory anchor:** `/Users/egedoganay/Desktop/force_coral/FORTE_Methodology_and_Outline.pdf`
- **Architecture anchor:** `/Users/egedoganay/Desktop/force_coral/RSS2026-CoRAL`

## Intended Phase-1 Deliverable

Phase 1 delivers the final pipeline shape:

1. one-shot semantic initialization
2. bounded outer semantic supervision
3. inner FORTE control loop
4. wall-assisted vertical lift task
5. structured artifacts: logs, plots, and video

## Module Mapping

- `force_coral/dynamics/estimator.py`
  - theory anchor: translational SPD stiffness estimation
  - implementation note: phase-1 uses SPD(3) exponential-map updates

- `force_coral/perception/vlm_interface.py`
  - architecture anchor: semantic task conditioning
  - implementation note: bounded semantic package instead of free-form code generation

- `force_coral/perception/semantic_manager.py`
  - architecture anchor: outer semantic loop
  - implementation note: constrained supervisor that updates only high-level parameters

- `force_coral/controllers/run_forte.py`
  - architecture anchor: CoRAL-style receding-horizon execution loop
  - theory anchor: task-aware energy-aware MPPI with force bands

- `force_coral/controllers/forte_support.py`
  - implementation note: task monitor, artifacts, wall-lift task frame, and replay generation

## Wall-Lift Task Rationale

The wall-assisted vertical lift was selected because it makes force-to-effect
coupling central to success:

- too little normal force -> the object loses support and falls
- too much normal force -> friction / sticking prevents upward progress
- correct force band -> stable upward motion

This is a stronger phase-1 study task than the previous wall-flip demo because
it directly exposes contact-maintenance and over-force failure modes.

## Progress Log

### Milestone 1

- Refactored package bootstrap so lightweight unit tests can import FORTE
  modules without requiring the full robotics stack.
- Replaced the old 6x6 stiffness estimator with a phase-1 3x3 translational
  estimator using the SPD exponential map.
- Introduced a bounded semantic package and semantic manager.
- Added a new phase-1 FORTE controller path with:
  - task monitor
  - structured telemetry
  - plot generation
  - MP4 replay export
- Added the wall-lift BDDL task and benchmark registration.

### Milestone 2

- Added a bounded semantic supervisor and updated tests around:
  - semantic revisions
  - force-band penalties
  - monitor logic
  - artifact generation
- Pinned the setup files to LIBERO-compatible `robosuite==1.4.0` and
  `bddl==1.0.1`, then created the dedicated `forte_phase1` conda environment.
- Added runtime cache-dir bootstrap (`NUMBA_CACHE_DIR`, `MPLCONFIGDIR`,
  `XDG_CACHE_HOME`) so Apple Silicon imports do not fail on unwritable default
  cache paths.
- Tightened success semantics so phase-1 success now requires:
  - target height reached
  - wall contact still present
- Normalized the wall-lift success predicate to use the **top of the box**
  rather than the box center, so varying cube sizes remain comparable.

## Open Follow-Ups

- Verify the new wall-lift task parameters in simulation and tune the force band.
- Add per-contact MuJoCo force validation if the net body wrench proves noisy.
- Reintegrate optional FoundationPose only after the simulation-first study is stable.
- In this Codex desktop session, full MuJoCo environment construction on macOS
  still hangs after the robotics stack imports, with Apple GUI-service messages
  during env creation. The package and controller import successfully in the
  dedicated `forte_phase1` env, but an end-to-end rollout could not be fully
  validated inside this sandboxed session.
