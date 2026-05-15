"""Paper-grade figures aggregated across multiple FORTE runs.

Reads each run's ``forte_log.csv`` plus ``summary.json`` and produces:

  • stiffness_convergence_sweep.{pdf,png}  — Σ̂(t) for {k=50, 200, 800}
    overlaid on the matching ground-truth k_true lines.
  • in_band_ratio_sweep.{pdf,png}          — cumulative %-in-band per run.
  • force_band_sweep.{pdf,png}             — F_n(t) vs band per k value.

Usage:
    python -m FORTE.paper_figures \
        --runs my_runs/forte_press_the_spring_button/20260515_181557 \
               my_runs/forte_press_the_spring_button/20260515_180619 \
               my_runs/forte_press_the_spring_button/20260515_182036 \
        --out paper_figures/

Each --runs entry is a run directory; the script reads forte_log.csv and
summary.json from it. Output is a directory containing one .pdf + .png per
figure plus a small summary.csv with per-run aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


# Paper styling — bigger fonts, vector output, journal palette.
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"]


@dataclass
class RunData:
    label: str           # human label, e.g. "k = 200 N/m"
    color: str
    k_true: Optional[float]
    steps: np.ndarray
    sigma_max: np.ndarray
    force_normal: np.ndarray
    in_band: np.ndarray   # cumulative percentage 0..100
    band_lower: np.ndarray
    band_upper: np.ndarray
    summary: Dict[str, Any]


def _load_run(run_dir: str, color: str, label_override: Optional[str] = None) -> RunData:
    csv_path = os.path.join(run_dir, "forte_log.csv")
    summary_path = os.path.join(run_dir, "summary.json")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    with open(summary_path) as f:
        summary = json.load(f)

    steps: List[int] = []
    sigma_max: List[float] = []
    force_normal: List[float] = []
    band_lo: List[float] = []
    band_hi: List[float] = []
    in_band_cum = []
    running = 0
    k_true: Optional[float] = None

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            steps.append(int(row["step"]))
            try:
                eigs = json.loads(row["sigma_eigenvalues"])
                sigma_max.append(float(max(eigs)) if eigs else float("nan"))
            except (KeyError, ValueError, TypeError):
                sigma_max.append(float("nan"))
            force_normal.append(float(row.get("measured_force_normal") or 0.0))
            band_lo.append(float(row.get("force_band_lower") or 0.0))
            band_hi.append(float(row.get("force_band_upper") or 0.0))
            in_band_flag = (
                str(row.get("in_band", "")).lower() == "true"
                or str(row.get("regime", "")) == "in-band"
            )
            if in_band_flag:
                running += 1
            in_band_cum.append(100.0 * running / float(i + 1))
            if k_true is None:
                bs = row.get("button_stiffness")
                if bs and bs not in {"None", "null"}:
                    try:
                        k_true = float(bs)
                    except ValueError:
                        pass

    label = label_override or (
        f"k = {int(k_true)} N/m" if k_true else os.path.basename(run_dir)
    )
    return RunData(
        label=label,
        color=color,
        k_true=k_true,
        steps=np.asarray(steps),
        sigma_max=np.asarray(sigma_max),
        force_normal=np.asarray(force_normal),
        in_band=np.asarray(in_band_cum),
        band_lower=np.asarray(band_lo),
        band_upper=np.asarray(band_hi),
        summary=summary,
    )


def _save_pair(fig: plt.Figure, out_dir: str, stem: str) -> None:
    fig.savefig(os.path.join(out_dir, f"{stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=200, bbox_inches="tight")


def plot_stiffness_sweep(runs: List[RunData], out_dir: str) -> str:
    """The headline figure — Σ̂_max(t) for each k_true."""
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    for r in runs:
        ax.plot(r.steps, r.sigma_max, color=r.color, label=r.label)
        if r.k_true is not None:
            ax.axhline(
                r.k_true, color=r.color, linestyle="--", linewidth=1.0, alpha=0.55,
            )
            ax.axhline(0.9 * r.k_true, color=r.color, linestyle=":", linewidth=0.6, alpha=0.4)
            ax.axhline(1.1 * r.k_true, color=r.color, linestyle=":", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("control step")
    ax.set_ylabel(r"$\hat{\Sigma}_{\max}(t)$  (N/m)")
    ax.set_yscale("log")
    ax.set_title("Riemannian stiffness estimator convergence")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save_pair(fig, out_dir, "stiffness_convergence_sweep")
    plt.close(fig)
    return os.path.join(out_dir, "stiffness_convergence_sweep.pdf")


def plot_in_band_sweep(runs: List[RunData], out_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for r in runs:
        ax.plot(r.steps, r.in_band, color=r.color, label=r.label)
    ax.set_xlabel("control step")
    ax.set_ylabel("% of steps with $F_n \\in [F_{min}, F_{max}]$")
    ax.set_ylim(0, 100)
    ax.set_title("Force-band tracking (cumulative)")
    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    _save_pair(fig, out_dir, "in_band_ratio_sweep")
    plt.close(fig)
    return os.path.join(out_dir, "in_band_ratio_sweep.pdf")


def plot_force_band_sweep(runs: List[RunData], out_dir: str) -> str:
    """One column per run: F_n(t) with [F_min, F_max] shaded."""
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 3.6), sharey=False)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, runs):
        # Shade the band where it's meaningful (skip the wide [0, 100] approach).
        tight = (r.band_upper - r.band_lower) <= 20.0
        if np.any(tight):
            lo = np.where(tight, r.band_lower, np.nan)
            hi = np.where(tight, r.band_upper, np.nan)
            ax.fill_between(r.steps, lo, hi, alpha=0.18, color=r.color, label="target band")
        ax.plot(r.steps, r.force_normal, color=r.color, linewidth=1.4)
        ax.set_title(r.label)
        ax.set_xlabel("step")
        ax.set_ylabel("$F_n$  (N)")
    fig.tight_layout()
    _save_pair(fig, out_dir, "force_band_sweep")
    plt.close(fig)
    return os.path.join(out_dir, "force_band_sweep.pdf")


def write_summary_csv(runs: List[RunData], out_dir: str) -> str:
    path = os.path.join(out_dir, "summary.csv")
    fields = [
        "label", "k_true", "num_steps", "success", "time_in_band",
        "max_normal_force", "num_revisions", "contact_latched_at_step",
        "final_sigma_max",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in runs:
            final_sigma = float(r.sigma_max[-1]) if len(r.sigma_max) else float("nan")
            s = r.summary
            w.writerow({
                "label": r.label,
                "k_true": r.k_true,
                "num_steps": s.get("num_steps"),
                "success": s.get("success"),
                "time_in_band": s.get("time_in_band"),
                "max_normal_force": s.get("max_normal_force"),
                "num_revisions": s.get("num_revisions"),
                "contact_latched_at_step": s.get("contact_latched_at_step"),
                "final_sigma_max": final_sigma,
            })
    return path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="One or more run directories.")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Optional label per run (defaults to 'k = <k_true>').")
    p.add_argument("--out", default="paper_figures",
                   help="Output directory.")
    args = p.parse_args()

    if args.labels and len(args.labels) != len(args.runs):
        raise SystemExit("--labels must match --runs length")

    os.makedirs(args.out, exist_ok=True)
    runs: List[RunData] = []
    for i, run_dir in enumerate(args.runs):
        label = args.labels[i] if args.labels else None
        color = _PALETTE[i % len(_PALETTE)]
        runs.append(_load_run(run_dir, color=color, label_override=label))

    # Sort by k_true so the legend reads small→large.
    runs.sort(key=lambda r: (r.k_true if r.k_true is not None else float("inf")))
    for i, r in enumerate(runs):
        r.color = _PALETTE[i % len(_PALETTE)]

    stem_k = plot_stiffness_sweep(runs, args.out)
    stem_b = plot_in_band_sweep(runs, args.out)
    stem_f = plot_force_band_sweep(runs, args.out)
    summary_path = write_summary_csv(runs, args.out)

    print(f"[paper_figures] {len(runs)} runs aggregated → {args.out}/")
    for s in (stem_k, stem_b, stem_f, summary_path):
        print(f"  - {s}")


if __name__ == "__main__":
    main()
