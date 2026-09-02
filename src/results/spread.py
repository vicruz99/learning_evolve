"""Signal-vs-noise analysis for the ICL campaign: is an arm difference resolvable at all?

Two different noise quantities are computed here, because they answer two different questions
and mixing them up is the easiest way to over-claim:

``sigma_seed`` — **the problem's noise unit.**
    At generation 0 every arm whose ``parent_source`` is ``initial`` / ``best`` / ``puct`` runs a
    *byte-identical* prompt: the buffer is empty, so the ICL context block renders to "" (see
    ``context.selection.build_context_block``), and the parent is the seed solution regardless of
    selection rule. Generation 0 is therefore the same experiment repeated once per run, differing
    only in RNG seed and temperature-1.0 sampling. The standard deviation of ``gen_best_score``
    over those runs is a direct, assumption-free estimate of how much a single generation of
    ``groups_per_batch * group_size`` candidates moves by luck alone.

    Use it to ask *"does this problem have enough headroom to be worth running?"* — express each
    arm's total improvement in units of sigma_seed and see how much of the axis the choice of
    method actually buys. ``parent_source="none"`` (archive-only) is excluded from the estimate,
    since its generation-0 prompt genuinely differs (no parent solution at all).

``sigma_within`` — **the run-to-run noise of a finished run.**
    Pooled standard deviation of the *final* best score within each arm, over arms with >= 2 seeds.
    This is the right denominator for comparing two arms, because it is the scatter the comparison
    actually has to beat. It is not the same as sigma_seed: 25 generations of search partially
    average out generation-0 luck, so sigma_within is usually the smaller of the two — and where it
    is *not*, outcomes stay path-dependent all the way to the end.

    Use it for effect sizes, for the minimum detectable effect at a given seed count, and to size
    how many seeds an experiment needs.

**Why the baseline is a per-problem constant, not each run's own generation 0.** Two reasons.
(1) Subtracting a constant cancels out of every comparison — it fixes where the axis origin sits and
changes no ranking, no effect size and no p-value. (2) Subtracting each run's *own* generation-0 best
is the classic change-score trap: ``Var(final - g0) = Var(final) + Var(g0) - 2*Cov``, so it only pays
when generation-0 luck predicts the finish. Measured on this campaign it does not — corr(g0, final)
is +0.41 (ac1), -0.11 (ac2), +0.48 (erdos), -0.18 (trimul_b200) — and since sigma_seed > sigma_within
on three of four problems, pairing roughly doubles the within-arm sd (ac1 0.00250 -> 0.00544, ac2
0.00364 -> 0.00688, erdos 2.54e-5 -> 5.43e-5, trimul 940 -> 1011). ANCOVA on generation 0 is the
principled middle ground and also loses on all four. ``--baseline`` re-runs the comparison so the
claim can be re-checked when more seeds land.

Significance is reported by an exact/Monte-Carlo permutation test on the two arms' seed values —
no normality assumption, which matters at n = 2-6. Note the resolution floor: with n_a vs n_b
seeds the smallest achievable p-value is 1 / C(n_a + n_b, n_a), so e.g. 3-vs-6 bottoms out at
p ~ 0.012 and 3-vs-3 at p ~ 0.05. An arm with one seed has no p-value at all.

Run it:  ``python -m results.spread``  (from ``src/``), or ``--runs-dir``/``--problem`` to narrow.
"""
from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import math
import os
import random
import statistics as st

# Problems this analysis covers, and whether the metric is maximised.
PROBLEMS: dict[str, bool] = {
    "ac1": False,
    "ac2": True,
    "erdos": False,
    "trimul_b200": False,
    "trimul_h100": False,
}

# Arms whose generation-0 prompt is identical, and therefore usable as the noise probe.
# ``none`` is excluded: it shows no parent solution, so its generation 0 is a different experiment.
_IDENTICAL_GEN0 = {"initial", "best", "puct"}

# Runs only pool with runs that ran the SAME protocol. Filtering on generation count alone is a
# trap: the 1x5 `like_shinka` arms reach 145-187 generations on a third of the candidate budget, so
# a `generations >= 25` filter silently folds them into the 6x16 PUCT arm and drags its mean down.
# Batch shape is part of the protocol, not a nuisance parameter — compare shapes deliberately, in
# their own analysis, against a matched candidate budget.
PROTOCOL_SHAPE = "6x16"
PROTOCOL_CANDIDATES = 2400


def arm_of(cfg: dict) -> str:
    """Canonical arm name. Reads ``parent_source`` — never infer the arm from ``n_context``."""
    source = cfg.get("parent_source")
    strategy = cfg.get("context_strategy")
    n = cfg.get("n_context") or 0
    if source == "initial":
        return "BoN"
    if source == "best":
        return "greedy"
    if source == "none":
        return f"archive n{n}"
    return "PUCT" if n == 0 else f"PUCT+n{n} {strategy}"


def load_runs(runs_dir: str) -> list[dict]:
    """Every run under ``runs_dir`` that has both a config and a summary, at any nesting depth."""
    runs = []
    for summary_path in sorted(glob.glob(os.path.join(runs_dir, "**", "summary.json"), recursive=True)):
        d = os.path.dirname(summary_path)
        config_path = os.path.join(d, "config.json")
        if not os.path.exists(config_path):
            continue
        cfg = json.load(open(config_path))
        summary = json.load(open(summary_path))
        per_gen = summary.get("per_generation") or []
        if not per_gen:
            continue
        totals = summary.get("totals") or {}
        runs.append({
            "rel": os.path.relpath(d, runs_dir),
            "problem": cfg.get("problem"),          # config is authoritative, not the directory
            "arm": arm_of(cfg),
            "parent_source": cfg.get("parent_source"),
            "seed": cfg.get("seed"),
            "shape": f"{cfg.get('groups_per_batch')}x{cfg.get('group_size')}",
            "generations": len(per_gen),
            "candidates": totals.get("candidates") or 0,
            "gen0_best": per_gen[0].get("gen_best_score"),
            "final_best": (summary.get("best") or {}).get("score"),
        })
    return runs


def sigma_seed(runs: list[dict]) -> tuple[float | None, float | None, int]:
    """(sd, median, n) of generation-0 best score over the identically-prompted runs."""
    values = [
        r["gen0_best"] for r in runs
        if r["parent_source"] in _IDENTICAL_GEN0 and r["gen0_best"] is not None
    ]
    if len(values) < 2:
        return None, None, len(values)
    return st.stdev(values), st.median(values), len(values)


def sigma_within(by_arm: dict[str, list[float]]) -> tuple[float | None, int]:
    """Pooled within-arm sd of final score, over arms with >= 2 seeds. Returns (sd, dof)."""
    squares, dof = 0.0, 0
    for values in by_arm.values():
        if len(values) < 2:
            continue
        mean = st.mean(values)
        squares += sum((v - mean) ** 2 for v in values)
        dof += len(values) - 1
    if dof == 0:
        return None, 0
    return math.sqrt(squares / dof), dof


def permutation_p(a: list[float], b: list[float], maximize: bool, iters: int = 200_000) -> tuple[float, float]:
    """One-sided p for "a beats b". Exact when the split count is small, else Monte-Carlo."""
    sign = 1 if maximize else -1
    observed = (st.mean(a) - st.mean(b)) * sign
    pool = a + b
    n_a = len(a)
    n_splits = math.comb(len(pool), n_a)
    if n_splits <= iters:                                   # exact
        hits = sum(
            1 for idx in itertools.combinations(range(len(pool)), n_a)
            if (st.mean([pool[i] for i in idx])
                - st.mean([pool[i] for i in range(len(pool)) if i not in idx])) * sign >= observed - 1e-15
        )
        return observed, hits / n_splits
    rng = random.Random(0)                                  # Monte-Carlo
    hits = 0
    for _ in range(iters):
        rng.shuffle(pool)
        if (st.mean(pool[:n_a]) - st.mean(pool[n_a:])) * sign >= observed:
            hits += 1
    return observed, (hits + 1) / (iters + 1)


def min_detectable_effect(sd: float, n_per_arm: int) -> float:
    """Rough two-sample MDE at alpha=0.05 / power=0.8: 2.8 * sd * sqrt(2/n)."""
    return 2.8 * sd * math.sqrt(2.0 / n_per_arm)


def half_width(sd: float, n: int) -> float:
    """Half-width of one arm's uncertainty span: 1.4 * sd * sqrt(2/n).

    Chosen so that two arms with equal n are separable exactly when their spans stop overlapping
    (the two half-widths then sum to the two-sample MDE above). For unequal n it is mildly
    conservative. n=1 gives 1.98*sd, which is the honest way to draw "no error information".
    """
    return 1.4 * sd * math.sqrt(2.0 / n)


def adjust(runs: list[dict], baseline: str) -> dict[str, float]:
    """Per-run adjusted score under one of the three baselines. Keyed by ``rel``."""
    if baseline == "constant":
        return {r["rel"]: r["final_best"] for r in runs}
    if baseline == "paired":
        return {r["rel"]: r["final_best"] - r["gen0_best"] for r in runs}
    xs = [r["gen0_best"] for r in runs]
    ys = [r["final_best"] for r in runs]
    mx, my = st.mean(xs), st.mean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom if denom else 0.0
    return {r["rel"]: r["final_best"] - slope * (r["gen0_best"] - mx) for r in runs}


def report(runs_dir: str, only: str | None = None, baseline: str = "constant") -> None:
    runs = [r for r in load_runs(runs_dir) if r["problem"] in PROBLEMS]
    complete = [r for r in runs
                if r["shape"] == PROTOCOL_SHAPE and r["candidates"] >= PROTOCOL_CANDIDATES]
    off_protocol = [r for r in runs if r not in complete]

    print(f"{len(runs)} runs found, {len(complete)} on protocol "
          f"({PROTOCOL_SHAPE}, >= {PROTOCOL_CANDIDATES} candidates)\n")
    if off_protocol:
        print("off protocol — reported separately, never pooled with the arms below:")
        for r in sorted(off_protocol, key=lambda r: r["rel"]):
            print(f"   {r['rel']:<40} {r['problem']:<12} {r['arm']:<22} shape={r['shape']:<6}"
                  f" gens={r['generations']:>3} cands={r['candidates']:>5}")
        print()
    mismatched = [r for r in runs if not r["rel"].startswith(str(r["problem"]).split("_")[0])]
    if mismatched:
        print("directory / config.problem mismatches (config wins):")
        for r in mismatched:
            print(f"   {r['rel']:<44} config.problem={r['problem']}")
        print()

    for problem, maximize in PROBLEMS.items():
        if only and problem != only:
            continue
        group = [r for r in complete if r["problem"] == problem]
        if len(group) < 2:
            continue
        sd_seed, median_gen0, n_gen0 = sigma_seed(group)
        scored = [r for r in group if r["final_best"] is not None and r["gen0_best"] is not None]
        adjusted = adjust(scored, baseline)
        by_arm: dict[str, list[float]] = collections.defaultdict(list)
        for r in scored:
            by_arm[r["arm"]].append(adjusted[r["rel"]])
        sd_within, dof = sigma_within(by_arm)
        if sd_seed is None or sd_within is None:
            continue

        sign = 1 if maximize else -1
        arm_mean = {a: st.mean(v) for a, v in by_arm.items()}
        # Centre on the median of the *adjusted* scores, so the column stays readable under every
        # baseline. The origin cancels out of spread, d and p — it only fixes where the axis starts.
        origin = median_gen0 if baseline == "constant" else st.median(list(adjusted.values()))
        in_sigma = {a: (m - origin) * sign / sd_seed for a, m in arm_mean.items()}
        order = sorted(by_arm, key=lambda a: -in_sigma[a])
        spread = in_sigma[order[0]] - in_sigma[order[-1]]

        print("=" * 96)
        print(f"{problem}   ({'maximize' if maximize else 'minimize'})   {len(group)} runs, {len(by_arm)} arms")
        print(f"  sigma_seed   = {sd_seed:<12.6g} (sd of generation-0 best over {n_gen0} identically-prompted runs)")
        print(f"  median gen-0 = {median_gen0:<12.6g}")
        print(f"  sigma_within = {sd_within:<12.6g} (pooled within-arm sd of final best, {dof} dof)"
              f"   ratio to sigma_seed: {sd_within / sd_seed:.2f}x")
        print(f"  ARM SPREAD   = {spread:.2f} sigma_seed"
              f"   <-- how much of the axis the choice of method buys")
        print(f"  MDE at n=3   = {min_detectable_effect(sd_within, 3) / sd_seed:.2f} sigma_seed"
              f"   ({min_detectable_effect(sd_within, 3):.4g} raw) "
              f"| seeds needed for the full spread: "
              f"{math.ceil(2 * (2.8 * sd_within / (spread * sd_seed)) ** 2) if spread > 0 else '-'}")

        baseline = "PUCT" if "PUCT" in by_arm else order[0]
        print(f"\n  {'arm':<24}{'n':>2}{'mean':>14}{'in sigma_seed':>15}{'+/- (sigma_w)':>15}"
              f"{'d vs ' + baseline:>14}{'perm p':>9}")
        for a in order:
            values = by_arm[a]
            delta = (arm_mean[a] - arm_mean[baseline]) * sign
            d = delta / sd_within
            if a == baseline or len(values) < 2 or len(by_arm[baseline]) < 2:
                p_str = "-"
            else:
                _, p = permutation_p(values, by_arm[baseline], maximize)
                p_str = f"{p:.3f}"
            mark = "  <-- resolvable" if p_str != "-" and float(p_str) < 0.05 else ""
            hw = half_width(sd_within, len(values)) / sd_within
            print(f"  {a:<24}{len(values):>2}{arm_mean[a]:>14.6g}{in_sigma[a]:>15.2f}"
                  f"{hw:>15.2f}{d:>+14.2f}{p_str:>9}{mark}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--problem", default=None, help="restrict to one problem")
    parser.add_argument("--baseline", choices=("constant", "paired", "ancova"), default="constant",
                        help="what to subtract before comparing arms; see the module docstring. "
                             "'constant' (default) subtracts the per-problem generation-0 median, "
                             "'paired' subtracts each run's own generation-0 best, "
                             "'ancova' subtracts the fitted regression on generation 0.")
    args = parser.parse_args()
    report(args.runs_dir, args.problem, args.baseline)


if __name__ == "__main__":
    main()
