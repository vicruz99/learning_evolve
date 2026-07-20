"""Cross-run analysis helpers for ICL experiment results.

Load the per-run `progress.csv` / `summary.json` / `index.csv` written by
:class:`results.tracker.ExperimentTracker` and plot best-so-far curves and success rates across
experiments. Designed to be called from a notebook (see `notebooks/analyze_runs.ipynb`) but usable
as a plain module. Requires the `analysis` extra: `pip install -e ".[analysis]"`.
"""
from __future__ import annotations

import glob
import json
import os

import pandas as pd
import matplotlib.pyplot as plt


def load_index(runs_dir: str = "runs") -> pd.DataFrame:
    """Cross-experiment index. Uses index.csv if present, else rebuilds it from each summary.json."""
    idx_path = os.path.join(runs_dir, "index.csv")
    if os.path.exists(idx_path):
        return pd.read_csv(idx_path)
    rows = []
    for s in sorted(glob.glob(os.path.join(runs_dir, "*", "summary.json"))):
        d = json.load(open(s))
        rows.append({
            "run": os.path.basename(os.path.dirname(s)),
            "problem": d.get("problem"), "strategy": d.get("strategy"),
            "n_context": d.get("n_context"), "group_size": d.get("group_size"),
            "groups_per_batch": d.get("groups_per_batch"), "num_generations": d.get("num_generations"),
            "generations_done": len(d.get("per_generation", [])),
            "best_score": (d.get("best") or {}).get("score"), "status": d.get("status"),
        })
    return pd.DataFrame(rows)


def load_progress(runs_dir: str = "runs", runs: list[str] | None = None) -> pd.DataFrame:
    """Concatenate every run's progress.csv, adding a `run` column."""
    frames = []
    for pcsv in sorted(glob.glob(os.path.join(runs_dir, "*", "progress.csv"))):
        run = os.path.basename(os.path.dirname(pcsv))
        if runs and run not in runs:
            continue
        df = pd.read_csv(pcsv)
        if df.empty:
            continue
        df["run"] = run
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_solutions(runs_dir: str, run: str) -> pd.DataFrame:
    """The solutions manifest for one run (sol_id, state, parent, raw_score, value, gen)."""
    p = os.path.join(runs_dir, run, "solutions", "manifest.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.DataFrame([json.loads(line) for line in open(p)])


def _plot_metric(runs_dir, runs, column, ylabel, title, ax, ylim=None):
    df = load_progress(runs_dir, runs)
    if df.empty:
        print(f"No runs with progress.csv found under {runs_dir!r}.")
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    for run, g in df.groupby("run"):
        ax.plot(g["generation"], g[column], marker="o", markersize=3, label=run)
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="best")
    return ax


def plot_best_so_far(runs_dir: str = "runs", runs: list[str] | None = None, ax=None):
    """Best-so-far score vs generation, one line per run.

    Note: for minimize problems (erdos, ac1) lower is better, so those curves trend DOWN.
    """
    return _plot_metric(runs_dir, runs, "best_so_far_score",
                        "best so far (score)", "Best-so-far by generation", ax)


def plot_success_rate(runs_dir: str = "runs", runs: list[str] | None = None, ax=None):
    """Fraction of valid candidates per generation, one line per run."""
    return _plot_metric(runs_dir, runs, "success_rate",
                        "valid fraction", "Success rate by generation", ax, ylim=(0, 1.02))
