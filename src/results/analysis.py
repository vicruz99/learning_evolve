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


# ------------------------------------------------------------------ nested campaign loaders
# The loaders above assume a flat ``runs/<run>/``. Sweeps launched with `run_sweep.py` nest one
# level deeper -- ``runs/<group>/<run>/`` -- so these three read that layout instead. They are
# additive; nothing above changed.

def _arm_of(group: str, run: str, cfg: dict) -> str:
    """Short arm label: the campaign plus, for ICL runs, the context strategy and size.

    ``bon_gptoss/cp26_s1`` -> ``bon_gptoss``;
    ``ctx_gptoss/cp26_n10_s1_best`` -> ``ctx_gptoss:best``;
    ``ac1_gptoss_rest/n05_s1_random`` -> ``ac1_gptoss_rest:random``.
    """
    if cfg.get("n_context"):
        return f"{group}:{cfg.get('context_strategy')}"
    return group


def _seed_of(cfg: dict) -> int | None:
    return cfg.get("seed")


def load_grouped_index(runs_dir: str = "runs", groups: list[str] | None = None) -> pd.DataFrame:
    """One row per run across every ``runs/<group>/<run>/summary.json``.

    Joins in the knobs and the host from ``config.json`` -- ``host`` matters because these
    campaigns were spread over several machines, and eval-time comparisons are only valid within
    one host (see ``docs/BOSCH_CLUSTER.md``).
    """
    rows = []
    for s in sorted(glob.glob(os.path.join(runs_dir, "*", "*", "summary.json"))):
        run_dir = os.path.dirname(s)
        group, run = os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)
        if groups and group not in groups:
            continue
        d = json.load(open(s))
        cfg_path = os.path.join(run_dir, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        meta = cfg.get("_meta", {})
        n_events = sum(1 for _ in open(os.path.join(run_dir, "events.jsonl"))) \
            if os.path.exists(os.path.join(run_dir, "events.jsonl")) else 0
        rows.append({
            "group": group, "run": run, "arm": _arm_of(group, run, cfg),
            "problem": d.get("problem"), "model": cfg.get("model_name"),
            "host": (meta.get("host") or "").split(".")[0],
            "strategy": d.get("strategy"), "n_context": d.get("n_context"),
            "seed": _seed_of(cfg), "group_size": d.get("group_size"),
            "groups_per_batch": d.get("groups_per_batch"),
            "num_generations": d.get("num_generations"),
            "generations_done": len(d.get("per_generation") or []),
            "candidates": n_events,
            "best_score": (d.get("best") or {}).get("score"),
            "metric_name": d.get("metric_name"), "maximize": d.get("maximize"),
            "status": d.get("status"),
            "started_at": d.get("started_at"), "updated_at": d.get("updated_at"),
            "wall_hours": sum(g.get("wall_seconds") or 0
                              for g in (d.get("per_generation") or [])) / 3600.0,
            "path": run_dir,
        })
    return pd.DataFrame(rows)


def load_events(runs_dir: str, group: str, run: str) -> pd.DataFrame:
    """Every candidate of one run, in completion order, from ``events.jsonl``.

    Adds two derived columns the raw file does not have: ``valid`` (the definition used everywhere
    -- ``correctness == 1.0`` *and* a non-null ``raw_score``; note ``failure_type`` is ``''`` rather
    than null for a success) and ``k``, the 1-based candidate index used as the budget axis.
    """
    p = os.path.join(runs_dir, group, run, "events.jsonl")
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.DataFrame([json.loads(line) for line in open(p)])
    if df.empty:
        return df
    df["valid"] = (df["correctness"] == 1.0) & df["raw_score"].notna()
    df["failure_type"] = df["failure_type"].replace("", None)
    df["k"] = range(1, len(df) + 1)
    return df


def load_grouped_progress(runs_dir: str = "runs", groups: list[str] | None = None) -> pd.DataFrame:
    """Per-generation rows from every ``summary.json``'s ``per_generation`` list.

    Richer than ``progress.csv``: keeps the ``failure_types`` breakdown (flattened to
    ``fail_<type>`` columns) and all four latency percentile blocks (``eval_p50`` ...).
    """
    rows = []
    for s in sorted(glob.glob(os.path.join(runs_dir, "*", "*", "summary.json"))):
        run_dir = os.path.dirname(s)
        group, run = os.path.basename(os.path.dirname(run_dir)), os.path.basename(run_dir)
        if groups and group not in groups:
            continue
        d = json.load(open(s))
        cfg_path = os.path.join(run_dir, "config.json")
        cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        for g in d.get("per_generation") or []:
            row = {"group": group, "run": run, "arm": _arm_of(group, run, cfg),
                   "problem": d.get("problem"), "seed": _seed_of(cfg),
                   "host": (cfg.get("_meta", {}).get("host") or "").split(".")[0]}
            row.update({k: v for k, v in g.items()
                        if k not in ("failure_types", "usage", "decode_percentiles",
                                     "eval_percentiles", "queue_percentiles", "grade_percentiles")})
            for ftype, n in (g.get("failure_types") or {}).items():
                row[f"fail_{ftype}"] = n
            for block, prefix in (("eval_percentiles", "eval"), ("grade_percentiles", "grade"),
                                  ("queue_percentiles", "queue"), ("decode_percentiles", "decode")):
                for pct, v in (g.get(block) or {}).items():
                    row[f"{prefix}_{pct}"] = v
            for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "truncated"):
                row[key] = (g.get("usage") or {}).get(key)
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in [c for c in df.columns if c.startswith("fail_")]:
        df[c] = df[c].fillna(0).astype(int)
    return df


RunSelection = list[str] | dict[str, str] | None


def _resolve_runs(runs) -> dict[str, str] | None:
    """Normalize a run selection into an ordered {run_dir: legend_label} mapping.

    Accepts ``None`` (all runs, labelled by their directory name), a list of run
    directory names (label == name), or a dict mapping run dir -> custom legend label.
    """
    if runs is None:
        return None
    if isinstance(runs, dict):
        return dict(runs)
    return {r: r for r in runs}


def _finish(ax, ylabel, title, ylim, save_path):
    """Common axis styling + optional PNG save. Returns the axis."""
    ax.set_xlabel("generation")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    if save_path:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"saved -> {save_path}")
    return ax


def _plot_progress_metric(runs_dir, runs, column, ylabel, title, ax, ylim=None,
                          save_path=None, scale=1.0, figsize=(8, 5)):
    """Plot a per-generation column from each run's progress.csv, one line per run."""
    labels = _resolve_runs(runs)
    df = load_progress(runs_dir, list(labels) if labels else None)
    if df.empty:
        print(f"No runs with progress.csv found under {runs_dir!r}.")
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    order = list(labels) if labels else sorted(df["run"].unique())
    for run in order:
        g = df[df["run"] == run].sort_values("generation")
        if g.empty:
            print(f"warning: no progress rows for run {run!r}; skipping.")
            continue
        label = labels[run] if labels else run
        ax.plot(g["generation"], g[column] * scale, marker="o", markersize=3, label=label)
    return _finish(ax, ylabel, title, ylim, save_path)


def plot_best_so_far(runs_dir: str = "runs", runs: RunSelection = None, ax=None,
                     save_path: str | None = None, title: str = "Best-so-far by generation"):
    """Best-so-far score vs generation, one line per run.

    ``runs`` selects and labels the runs to draw. It may be a list of run directory
    names, or a ``{run_dir: legend_label}`` dict for custom legend names. Pass
    ``save_path`` (e.g. ``"best.png"``) to also write a PNG.

    Note: for minimize problems (erdos, ac1) lower is better, so those curves trend DOWN.
    """
    return _plot_progress_metric(runs_dir, runs, "best_so_far_score",
                                 "best so far (score)", title, ax, save_path=save_path)


def plot_valid_percentage(runs_dir: str = "runs", runs: RunSelection = None, ax=None,
                          save_path: str | None = None,
                          title: str = "Valid solutions (%) by generation"):
    """Percentage of valid candidates per generation, one line per run.

    Same ``runs`` selection / ``save_path`` semantics as :func:`plot_best_so_far`.
    """
    return _plot_progress_metric(runs_dir, runs, "success_rate", "valid solutions (%)",
                                 title, ax, ylim=(0, 102), save_path=save_path, scale=100.0)


# Backwards-compatible alias: same curve as plot_valid_percentage but on a 0-1 fraction.
def plot_success_rate(runs_dir: str = "runs", runs: RunSelection = None, ax=None,
                      save_path: str | None = None):
    """Fraction (0-1) of valid candidates per generation, one line per run."""
    return _plot_progress_metric(runs_dir, runs, "success_rate", "valid fraction",
                                 "Success rate by generation", ax, ylim=(0, 1.02),
                                 save_path=save_path)


def load_avg_valid_value(runs_dir: str = "runs", runs: RunSelection = None) -> pd.DataFrame:
    """Per-generation mean ``value`` over the valid solutions of each run.

    Reads each run's ``solutions/manifest.jsonl`` (which stores only valid solutions),
    and returns a long-form frame with columns ``generation``, ``avg_valid_value``, ``run``.
    """
    labels = _resolve_runs(runs)
    if labels is not None:
        names = list(labels)
    else:
        names = sorted(
            os.path.basename(os.path.dirname(os.path.dirname(m)))
            for m in glob.glob(os.path.join(runs_dir, "*", "solutions", "manifest.jsonl"))
        )
    frames = []
    for run in names:
        sols = load_solutions(runs_dir, run)
        if sols.empty:
            continue
        valid = sols[sols["correctness"] == 1.0] if "correctness" in sols else sols
        agg = (valid.groupby("gen")["value"].mean().reset_index()
               .rename(columns={"gen": "generation", "value": "avg_valid_value"}))
        agg["run"] = run
        frames.append(agg)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_avg_valid_value(runs_dir: str = "runs", runs: RunSelection = None, ax=None,
                         save_path: str | None = None,
                         title: str = "Average value of valid solutions by generation"):
    """Mean ``value`` over each generation's valid solutions, one line per run.

    Same ``runs`` selection / ``save_path`` semantics as :func:`plot_best_so_far`.
    """
    labels = _resolve_runs(runs)
    df = load_avg_valid_value(runs_dir, runs)
    if df.empty:
        print(f"No solution manifests found under {runs_dir!r}.")
        return None
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    order = list(labels) if labels else sorted(df["run"].unique())
    for run in order:
        g = df[df["run"] == run].sort_values("generation")
        if g.empty:
            print(f"warning: no valid solutions for run {run!r}; skipping.")
            continue
        label = labels[run] if labels else run
        ax.plot(g["generation"], g["avg_valid_value"], marker="o", markersize=3, label=label)
    return _finish(ax, "avg value of valid solutions", title, None, save_path)
