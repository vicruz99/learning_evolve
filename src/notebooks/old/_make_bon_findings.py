"""Generate notebooks/bon_findings.ipynb from a cell list (safer than hand-writing JSON)."""
import json, sys

C = []
def md(s): C.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
def code(s): C.append({"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": s.strip("\n").splitlines(True)})

md(r"""
# Best-of-N on circle_packing_26 — what 4,800 samples buy

`runs/bon_gptoss`: gpt-oss-120b, 4 seeds x 1,200 candidates, no past experience at all
(`parent-source initial`, `n-context 0`). This is the floor every other arm has to beat.

Because Best-of-N has no dependency between candidates, the x-axis here is **candidate count k**,
not generation. The other arms' generation *g* corresponds to k = 80(g+1), so they drop straight
onto these plots when they finish.

**The four findings:** sampling saturates at k~160 &middot; the score space is a handful of discrete
constructions &middot; the 4-seed noise floor is wider than the headroom left &middot; "valid" does not
mean "good".
""")

code(r"""
import json, os, glob, collections
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

RUNS = "../runs" if os.path.basename(os.getcwd()) == "notebooks" else "runs"
TARGET = 2.636          # envs/circle_packing.py:124 — the value the prompt asks the model to beat
BUDGET = 1200           # candidates per run, identical in every arm

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.autolayout": True})
CSEED = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]   # per-seed
CMEAN, CREF = "#22223B", "#B0B0B0"


def load_arm(campaign):
    # {run: value-or-nan per candidate, in completion order} from each run's events.jsonl
    out = {}
    for f in sorted(glob.glob(os.path.join(RUNS, campaign, "*", "events.jsonl"))):
        run, vals, fails = os.path.basename(os.path.dirname(f)), [], []
        for line in open(f):
            e = json.loads(line)
            ok = e.get("correctness") == 1.0 and e.get("raw_score") is not None
            vals.append(e["raw_score"] if ok else np.nan)
            fails.append("valid" if ok else (e.get("failure_type") or "unknown"))
        out[run] = {"value": np.array(vals, float), "failure": fails}
    return out


def best_so_far(v):
    # running max ignoring failures; 0 until the first valid candidate
    return np.fmax.accumulate(np.nan_to_num(v, nan=-np.inf)).clip(min=0.0)


BON = load_arm("bon_gptoss")
SEEDS = list(BON)
V = {s: BON[s]["value"] for s in SEEDS}                      # per-candidate scores
B = {s: best_so_far(V[s]) for s in SEEDS}                    # best-so-far curves
POOL = np.concatenate([V[s] for s in SEEDS])                 # 4,800 candidates, i.i.d.
VALID = POOL[~np.isnan(POOL)]

print(f"{len(SEEDS)} runs x {len(V[SEEDS[0]])} candidates = {POOL.size} total, "
      f"{VALID.size} valid ({VALID.size / POOL.size:.1%})")
""")

md(r"""
## 1. Sampling saturates at ~160 candidates

87 % of the budget buys almost nothing. Two of the four seeds never improved after k=320 and k=80.
""")

code(r"""
mean_at = lambda k: np.mean([B[s][k - 1] for s in SEEDS])
KNEE, gain_tail = 160, mean_at(BUDGET) - mean_at(160)

rng = np.random.default_rng(0)
KS = np.unique(np.round(np.logspace(0, np.log10(2 * BUDGET), 22)).astype(int))
DRAW = np.nan_to_num(POOL, nan=0.0)                   # a failed candidate scores 0, same as best_so_far
boot = {k: np.array([DRAW[rng.choice(DRAW.size, k, replace=(k > DRAW.size))].max()
                     for _ in range(1500)]) for k in KS}      # k > 4800 needs replacement

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.7))

for c, s in zip(CSEED, SEEDS):
    axL.step(np.arange(1, BUDGET + 1), B[s], where="post", lw=1.1, color=c, alpha=.85,
             label=f"seed {s[-1]}")
axL.step(np.arange(1, BUDGET + 1), np.mean([B[s] for s in SEEDS], axis=0), where="post",
         lw=2.4, color=CMEAN, label="mean of 4 seeds")
axL.axhline(TARGET, ls="--", lw=1, color=CREF)
axL.text(1.5, TARGET, f" target {TARGET}", va="bottom", fontsize=8, color="#666")
axL.axvline(KNEE, ls=":", lw=1, color="#999")
axL.annotate(f"k={KNEE}\nlast {BUDGET - KNEE} candidates: {gain_tail:+.4f}",
             xy=(KNEE, 2.585), xytext=(300, 2.55), fontsize=8, color="#444",
             arrowprops=dict(arrowstyle="->", lw=.8, color="#999"))
axL.set(xscale="log", xlim=(1, BUDGET), ylim=(2.40, 2.645),
        xlabel="candidates drawn (k)", ylabel="best sum of radii found",
        title="Best-so-far per run")
axL.xaxis.set_major_formatter(ScalarFormatter())
axL.legend(fontsize=8, loc="lower right")

m = np.array([boot[k].mean() for k in KS])
axR.fill_between(KS, [np.percentile(boot[k], 5) for k in KS],
                 [np.percentile(boot[k], 95) for k in KS], color=CMEAN, alpha=.15,
                 label="5-95 % of draws")
axR.plot(KS, m, lw=2, color=CMEAN, marker="o", ms=3, label="E[best of k]")
axR.axhline(TARGET, ls="--", lw=1, color=CREF)
axR.axvline(BUDGET, ls=":", lw=1, color="#999")
axR.text(BUDGET * .93, 2.47, "budget", ha="right", fontsize=8, color="#666", rotation=90)
axR.set(xscale="log", xlim=(1, 2 * BUDGET), ylim=(2.40, 2.645), xlabel="candidates drawn (k)",
        title="Expected best of k (bootstrap over the pooled 4,800)")
axR.xaxis.set_major_formatter(ScalarFormatter())
axR.legend(fontsize=8, loc="lower right")
fig.savefig("bon_saturation.png", dpi=200, bbox_inches="tight")

print(f"{'k':>6}  {'mean best':>10}  {'seed spread':>12}  {'E[best of k]':>13}")
for k in (16, 80, 160, 320, 640, 1200):
    sp = max(B[s][k - 1] for s in SEEDS) - min(B[s][k - 1] for s in SEEDS)
    kb = KS[np.argmin(abs(KS - k))]
    print(f"{k:>6}  {mean_at(k):>10.4f}  {sp:>12.4f}  {boot[kb].mean():>13.4f}")
print(f"\ndoubling the budget to 2400 would add {boot[KS[-1]].mean() - boot[KS[np.argmin(abs(KS-BUDGET))]].mean():+.4f}")
""")

md(r"""
**Finding 1.** The mean best-so-far reaches 2.618 by k=160 and 2.624 by k=1,200 — the last
1,040 candidates per run are worth **+0.006**. The bootstrap says another doubling to 2,400 would
add ~+0.002. More sampling is not the lever.
""")

md(r"""
## 2. Why: the model re-draws a handful of known constructions

The score distribution is not a landscape, it is a set of atoms — closed-form packings reproduced
from memory. The top tail is 12 candidates out of 3,331.
""")

code(r"""
atoms = collections.Counter(np.round(VALID, 9)).most_common(5)
attractor = 2.6180682559          # the local optimum every seed finds independently

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.5),
                               gridspec_kw={"width_ratios": [1.35, 1]})

axL.hist(VALID, bins=120, color="#4C72B0", alpha=.85)
axL.set(yscale="log", ylim=(.8, 1500), xlabel="sum of radii",
        ylabel="valid candidates (log)", title=f"All {VALID.size} valid scores")
for v, n in atoms:                                   # tick each atom, list them in one box
    axL.plot([v], [n * 1.35], "v", ms=4, color="#22223B")
axL.text(.03, .97, "most common scores\n" + "\n".join(f"{v:<9g} x{n}" for v, n in atoms),
         transform=axL.transAxes, va="top", ha="left", fontsize=7.5, family="monospace",
         color="#333", bbox=dict(fc="white", ec="#CCC", lw=.6, pad=3.5))
axL.axvline(TARGET, ls="--", lw=1, color=CREF)

tail = np.sort(VALID[VALID >= 2.55])
axR.plot(tail, np.arange(len(tail), 0, -1), lw=1.6, color=CMEAN)
for c, s in zip(CSEED, SEEDS):
    axR.plot(B[s][-1], 1.25, "v", ms=8, color=c, label=f"seed {s[-1]} final")
axR.axvline(TARGET, ls="--", lw=1, color=CREF)
axR.text(TARGET, 3, f"target {TARGET} ", ha="right", fontsize=7.5, color="#666", rotation=90)
axR.axvline(attractor, ls=":", lw=1, color="#999")
axR.annotate(f"{attractor:.4f}\nfound by all 4 seeds", xy=(attractor, 17),
             xytext=(2.5615, 40), fontsize=7.5, color="#444",
             arrowprops=dict(arrowstyle="->", lw=.8, color="#999"))
axR.set(yscale="log", xlim=(2.55, 2.6405), ylim=(.8, 260), xlabel="sum of radii",
        ylabel="candidates at least this good", title="The top tail")
axR.legend(fontsize=7.5, loc="lower left")
fig.savefig("bon_atoms.png", dpi=200, bbox_inches="tight")

print("most common valid scores:", "  ".join(f"{v:g} (x{n})" for v, n in atoms))
for thr in (2.55, 2.60, 2.62, 2.63, TARGET):
    print(f"  >= {thr:.3f}: {(VALID >= thr).sum():>5} of {VALID.size}")
print(f"\nbest single candidate in 4,800: {VALID.max():.4f}   (target {TARGET}, "
      f"gap {TARGET - VALID.max():.4f})")
""")

md(r"""
**Finding 2.** The five most common scores are hand-derivable constructions — 2.5 appears 339
times, 2.541421 (`= 2 + something/sqrt2`-style) 156 times. The 2.6180682559 configuration is found
independently by **all four seeds**, agreeing to 12 digits. Nothing in 4,800 candidates beat 2.6299,
and none reached 2.63. Extra samples mostly re-draw the same packings, which is exactly why the
curve in section 1 is flat.
""")

md(r"""
## 3. The consequence: this comparison cannot resolve the arms

The seed-to-seed noise is of the same size as the entire distance left to the target.
""")

code(r"""
fin = np.array([B[s][-1] for s in SEEDS])
sd, se = fin.std(ddof=1), fin.std(ddof=1) / np.sqrt(len(fin))
se_diff = se * np.sqrt(2)
MDE = 3.35 * se_diff        # (t_.025 + t_.20) at df=6 = 2.447 + 0.906; normality is approximate here
head = TARGET - fin.mean()

# resolution of alternative summary statistics, same 4 seeds
tops = {s: np.sort(V[s][~np.isnan(V[s])])[::-1] for s in SEEDS}
cands = {"max (best-of-1200)": fin,
         "mean top-10": np.array([tops[s][:10].mean() for s in SEEDS]),
         "mean top-50": np.array([tops[s][:50].mean() for s in SEEDS]),
         "AUC of best-so-far": np.array([B[s].mean() for s in SEEDS])}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.4),
                               gridspec_kw={"width_ratios": [1, 1.15]})

axL.axhspan(fin.mean() - se, fin.mean() + se, color=CMEAN, alpha=.13, label="mean $\\pm$ SE")
axL.axhline(fin.mean(), lw=1.6, color=CMEAN)
for i, (c, s) in enumerate(zip(CSEED, SEEDS)):
    axL.plot(i, fin[i], "o", ms=9, color=c)
axL.axhline(TARGET, ls="--", lw=1, color=CREF)
axL.annotate("", xy=(4.1, fin.mean()), xytext=(4.1, TARGET),
             arrowprops=dict(arrowstyle="<->", lw=1, color="#C44E52"))
axL.text(3.95, (fin.mean() + TARGET) / 2, f"headroom\n{head:.4f}", ha="right", va="center",
         fontsize=8, color="#C44E52")
axL.annotate("", xy=(-1.9, fin.mean()), xytext=(-1.9, fin.mean() + MDE),
             arrowprops=dict(arrowstyle="<->", lw=1, color="#4C72B0"))
axL.text(-1.78, fin.mean() + MDE / 2, f"smallest detectable\nwin {MDE:.4f}", va="center",
         ha="left", fontsize=8, color="#4C72B0")
axL.set(xlim=(-2.2, 4.4), ylim=(2.610, 2.6405), xticks=range(4),
        xticklabels=[f"s{s[-1]}" for s in SEEDS], ylabel="final best sum of radii",
        title="Final score, 4 seeds")
axL.legend(fontsize=8, loc="lower left")

names = list(cands)
ses = np.array([cands[n].std(ddof=1) / 2 for n in names]) * 1e3      # per-mille, for readable ticks
axR.barh(names, ses, color=["#4C72B0" if n == "max (best-of-1200)" else "#B9C6D9" for n in names])
for i, v in enumerate(ses):
    axR.text(v + .06, i, f"{v:.2f}", va="center", fontsize=8)
axR.invert_yaxis()
axR.set(xlim=(0, max(ses) * 1.22),
        xlabel="standard error over 4 seeds ($\\times 10^{-3}$, lower = more sensitive)",
        title="No summary statistic rescues it")
axR.grid(axis="y", visible=False)
fig.savefig("bon_resolution.png", dpi=200, bbox_inches="tight")

print(f"final scores: {np.array2string(fin, precision=4)}")
print(f"mean {fin.mean():.4f}  sd {sd:.4f}  SE {se:.4f}  SE of a 2-arm difference {se_diff:.4f}")
print(f"smallest detectable win ~{MDE:.4f}   vs headroom to target {head:.4f}")
print(f"=> at 8 seeds the threshold falls to ~{MDE / np.sqrt(2):.4f}")
""")

md(r"""
**Finding 3.** The 4-seed SE is 0.003, so a two-arm difference needs roughly **0.015** to be
detectable — while Best-of-N sits only **0.012** from the target. An arm that solved cp26 outright
would still not separate from this baseline. Switching summary statistic does not help
(mean-top-10 is marginally the most sensitive, and the max is already close).

**So cp26 is the wrong problem to discriminate arms on** — it is nearly solved by brute sampling.
The discriminating result has to come from cp32 or ac1, where the headroom is larger. Going to
8 seeds halves the threshold and is cheap: Best-of-N is GPU-bound and PUCT leaves the GPU idle, so
extra seeds can be co-scheduled on the same Ray head.
""")

md(r"""
## 4. "Valid" does not mean "good" — a warning for the context arms
""")

code(r"""
ORDER = ["valid", "process_crash", "invalid_result", "eval_timeout", "no_code"]
COL = dict(zip(ORDER, ["#55A868", "#C44E52", "#DD8452", "#8172B3", "#937860"]))
cnt = {s: collections.Counter(BON[s]["failure"]) for s in SEEDS}

fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 3.2),
                               gridspec_kw={"width_ratios": [1.15, 1]})

left = np.zeros(len(SEEDS))
for k in ORDER:
    w = np.array([cnt[s].get(k, 0) for s in SEEDS], float)
    axL.barh([f"s{s[-1]}" for s in SEEDS], w, left=left, color=COL[k],
             label=f"{k} ({int(w.sum())})")
    left += w
axL.invert_yaxis()
axL.set(xlabel="candidates", xlim=(0, 1700), xticks=[0, 400, 800, 1200],
        title="Outcome of every candidate")
axL.legend(fontsize=7.5, loc="center right")

frac = np.array([(VALID < t).mean() for t in (0.001, 1.0, 2.0, 2.5)])
axR.bar(["= 0", "< 1.0", "< 2.0", "< 2.5"], frac * 100, color="#DD8452", alpha=.9)
for i, f in enumerate(frac):
    axR.text(i, f * 100 + 1, f"{f:.0%}", ha="center", fontsize=8)
axR.set(ylabel="% of VALID solutions", ylim=(0, 82),
        title="How bad a 'valid' solution can be")
fig.savefig("bon_validity.png", dpi=200, bbox_inches="tight")

print(f"valid: {VALID.size / POOL.size:.1%} of {POOL.size};  "
      f"of those, {(VALID == 0).sum()} score exactly 0 and {(VALID < 2.0).mean():.0%} score < 2.0")
""")

md(r"""
**Finding 4.** 69 % of candidates are valid, and failures are stable across seeds
(`process_crash` ~180/run, `invalid_result` ~180/run, `eval_timeout` only ~6/run — the 530 s cap is
no longer censoring). But **41 % of *valid* solutions score below 2.0** and 324 score exactly 0.

That is a design problem for `sweeps/ctx_gptoss.yaml`: a `random` context strategy will fill the
prompt with junk about half the time, so "best vs random" would measure example **quality**, not
example **diversity**. Sample random contexts from the valid-and-above-threshold pool instead.
""")

md(r"""
## Summary

| | |
|---|---|
| Best-of-N ceiling, 1,200 candidates | **2.6236** mean of 4 seeds (best single 2.6299, target 2.636) |
| Saturation point | **k ~ 160**; the remaining 87 % of the budget is worth +0.006 |
| Why | scores are discrete memorised constructions; one local optimum found by all 4 seeds |
| Noise floor | SE 0.003 &rarr; smallest detectable win **~0.015** vs **0.012** of headroom |
| Actions | discriminate on cp32/ac1, not cp26; 8 seeds not 4; fix `random` context sampling |

The other two arms drop onto section 1's plots unchanged — `load_arm("puct_gptoss")` and
`load_arm("ctx_gptoss")`, with generation *g* read at k = 80(g+1).
""")

json.dump({"cells": C,
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                       "name": "python3"},
                        "language_info": {"name": "python"}},
           "nbformat": 4, "nbformat_minor": 5},
          open(sys.argv[1], "w"), indent=1)
print(f"wrote {sys.argv[1]} with {len(C)} cells")
