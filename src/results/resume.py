"""Where a run can honestly be resumed from — derived from its artifacts, never from its status field.

``--resume`` used to read exactly one thing, ``summary.json``'s ``status``, and that field lies in
every way that matters:

  * delete a run's generations and the summary left behind still says ``complete``, so the sweep skips
    the run instead of redoing it;
  * a run resumed once had its summary REWRITTEN from an empty per-generation list (the tracker built
    no prior state), so it claimed fewer generations than it had done — the next resume then rewound
    to that wrong point, or declared a 3-of-15 run finished;
  * a generation whose LLM requests all failed is recorded as an ordinary generation, because
    ``icl.loop._run_group`` catches the error, records an empty group and returns — the run walks on
    with a generation that never reached the model.

So every number here comes from the per-generation artifacts, and a generation counts as done only
when all of them are present, parseable and whole:

    generations/gen_NNNN/meta.json         written once, by end_generation -> the generation finished
      .parents[].children                  one record per candidate; a missing or short group means
                                           that group never came back from the LLM
    buffer/puct_sampler_step_NNNNNN.json   the buffer ``--resume-step N`` actually reloads
    buffer/context_pool.jsonl              the ICL context; a short pool silently shrinks the prompts
                                           of every generation after the resume

``inspect_run`` reports the longest trustworthy prefix; ``rewind`` moves everything after a given
generation into ``stale_<timestamp>/`` so a relaunch appends to consistent files instead of interleaving
with the attempt it replaces; ``prior_state`` rebuilds the counters the tracker needs to keep writing
ONE cumulative summary when a run IS continued.

Who uses which: ``run_sweep.py --resume`` restarts every incomplete run of a sweep from generation 0
(``rewind(run_dir, 0)``) — it uses this module to decide *whether* a run is finished, not where to
resume it, because a run built by two processes with an interruption buried inside is not the clean
lineage the arms are compared on. ``run_icl.py --resume-step`` is the mid-run path, for continuing one
long run by hand; that is where ``resume_step`` and ``prior_state`` earn their keep.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime

from results.tracker import USAGE_KEYS

_GEN_DIR_RE = re.compile(r"gen_(\d+)$")
_SNAPSHOT_RE = re.compile(r"_step_(\d+)\.json$")
_SOL_RE = re.compile(r"sol_(\d+)$")


# --------------------------------------------------------------------------------------------------
# reading a run directory
# --------------------------------------------------------------------------------------------------
def _read_json(path: str):
    """Parse ``path``, or None if it is missing, truncated or not JSON (a killed run leaves both)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _count_lines(path: str) -> int:
    try:
        with open(path) as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# Verdict cache for snapshot files, keyed by identity (path, mtime_ns, size) rather than by path, so
# a rewritten file misses. A snapshot holds the whole buffer -- every state with its full program code
# -- and `run_sweep --status` re-checks every snapshot of every run of the sweep, on a 60 s timer,
# inside the same loop that reaps and launches runs. Measured on a synthetic 12-run x 15-generation
# sweep: 268 MB re-read and re-parsed per render (0.94 s warm on local disk, and these run dirs are
# routinely on NFS). Snapshots are immutable once written, so the second parse never learns anything.
_SNAPSHOT_CACHE: dict[tuple[str, int, int], bool] = {}


def _snapshot_holds_states(path: str) -> bool:
    """Does this snapshot parse and hold states? Files are checked, not trusted: a snapshot killed
    mid-write is a truncated JSON file."""
    try:
        st = os.stat(path)
        key = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        return False
    cached = _SNAPSHOT_CACHE.get(key)
    if cached is None:
        store = _read_json(path)
        cached = isinstance(store, dict) and bool(store.get("states"))
        _SNAPSHOT_CACHE[key] = cached
    return cached


def _snapshot_steps(run_dir: str) -> set[int]:
    """Steps whose PUCT snapshot is present AND holds states — the only steps ``--resume-step`` can
    load."""
    steps = set()
    for path in glob.glob(os.path.join(run_dir, "buffer", "puct_sampler_step_*.json")):
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        if m and _snapshot_holds_states(path):
            steps.add(int(m.group(1)))
    return steps


def _generation_dirs(run_dir: str) -> dict[int, str]:
    out = {}
    for path in glob.glob(os.path.join(run_dir, "generations", "gen_*")):
        m = _GEN_DIR_RE.search(os.path.basename(path))
        if m and os.path.isdir(path):
            out[int(m.group(1))] = path
    return out


def _manifest_rows(run_dir: str) -> list[dict]:
    """Solution records, one per valid candidate ever written. This is the per-solution ground truth:
    it carries the generation, so it survives a mangled summary."""
    rows = []
    path = os.path.join(run_dir, "solutions", "manifest.jsonl")
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue                       # a torn last line, nothing more
    except OSError:
        pass
    return rows


def _check_generation(gen: int, gen_dir: str, groups: int | None, group_size: int | None
                      ) -> tuple[dict | None, str | None]:
    """Return (meta, damage) for one generation. ``meta`` is None whenever the generation may not be
    resumed past, and ``damage`` says why in one line."""
    meta = _read_json(os.path.join(gen_dir, "meta.json"))
    if meta is None:
        # end_generation writes meta.json last, so its absence means the generation never finished —
        # this is the normal shape of the generation a killed or timed-out run died inside.
        return None, f"gen {gen}: no meta.json (the generation never finished)"
    if meta.get("generation") != gen:
        return None, (f"gen {gen}: meta.json belongs to generation {meta.get('generation')} "
                      f"(the run dir was written by two processes)")
    parents = meta.get("parents")
    if not isinstance(parents, list) or not parents:
        return None, f"gen {gen}: meta.json records no parent groups"
    empty = [p.get("slot") for p in parents if not p.get("children")]
    if empty:
        # record_group(..., [], []) — icl.loop swallowed an LLM failure for these slots. The
        # generation is real on disk but the model never answered for part of it.
        return None, (f"gen {gen}: {len(empty)} of {len(parents)} groups recorded no candidates "
                      f"(slots {empty}) — the LLM was unreachable for them")
    if groups and len(parents) != groups:
        return None, (f"gen {gen}: {len(parents)} groups on disk, {groups} configured "
                      f"— the LLM failed before a group was recorded")
    if group_size:
        short = {p.get("slot"): len(p.get("children") or []) for p in parents
                 if len(p.get("children") or []) != group_size}
        if short:
            return None, (f"gen {gen}: groups {short} returned fewer than {group_size} candidates "
                          f"— part of the generation never came back from the LLM")
    return meta, None


@dataclass
class RunProgress:
    """What one run dir is really worth. ``good_generations`` counts trustworthy generations from 0;
    ``resume_step`` is what to hand ``--resume-step`` (0 = nothing survives, start over)."""
    run_dir: str
    num_generations: int | None = None
    good_generations: int = 0
    # Generation directories present at all, damaged or not. ``good_generations`` is what may be
    # resumed FROM; this is what a restart actually throws away, and the two diverge exactly when a
    # generation in the middle is damaged — which is when saying the right number matters most.
    generations_on_disk: int = 0
    resume_step: int = 0
    complete: bool = False
    damage: list[str] = field(default_factory=list)
    snapshots: list[int] = field(default_factory=list)   # steps with a loadable PUCT snapshot
    # summary.json's own account, for display and for spotting summaries a past resume mangled
    summary_status: str | None = None
    summary_generations: int = 0
    best: float | None = None
    updated_at: str | None = None
    wall: float | None = None
    tok_s: int | None = None

    @property
    def exists(self) -> bool:
        return os.path.isdir(self.run_dir)

    @property
    def has_tail(self) -> bool:
        """True when there is data after ``resume_step`` that a relaunch would otherwise interleave
        with (a partial generation, an orphaned snapshot, extra progress rows)."""
        return tail_exists(self.run_dir, self.resume_step)

    def describe(self) -> str:
        want = self.num_generations if self.num_generations is not None else "?"
        if self.complete:
            return f"complete ({self.good_generations}/{want})"
        if self.resume_step == 0:
            return f"start over (nothing resumable of {self.good_generations}/{want})"
        return f"resume at generation {self.resume_step}/{want}"


def tail_exists(run_dir: str, keep: int) -> bool:
    """Is there anything on disk from generation ``keep`` onwards? That is data a relaunch starting at
    ``keep`` would append on top of — the shape that gave one run dir two interleaved runs."""
    if not os.path.isdir(run_dir):
        return False
    if any(g >= keep for g in _generation_dirs(run_dir)):
        return True
    if any(s > keep for s in _snapshot_steps(run_dir)):
        return True
    if glob.glob(os.path.join(run_dir, "buffer", "*.tmp.*")):
        return True
    summary = _read_json(os.path.join(run_dir, "summary.json")) or {}
    if len(summary.get("per_generation") or []) > keep:
        return True
    return any(isinstance(r.get("gen"), int) and r["gen"] >= keep for r in _manifest_rows(run_dir))


def inspect_run(run_dir: str, num_generations: int | None = None) -> RunProgress:
    """Verify a run dir and report the furthest point it may be resumed from.

    ``num_generations`` overrides the run's own config (the sweep manifest knows it even when the run
    never got far enough to write a config).
    """
    cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
    want = num_generations if num_generations is not None else cfg.get("num_generations")
    prog = RunProgress(run_dir=run_dir, num_generations=want)

    summary = _read_json(os.path.join(run_dir, "summary.json")) or {}
    per_gen = summary.get("per_generation") or []
    prog.summary_status = summary.get("status")
    prog.summary_generations = len(per_gen)
    prog.best = (summary.get("best") or {}).get("score")
    prog.updated_at = summary.get("updated_at")
    if per_gen:
        last = per_gen[-1]
        prog.wall = last.get("wall_seconds")
        completion = (last.get("usage") or {}).get("completion_tokens") or 0
        prog.tok_s = round(completion / prog.wall) if prog.wall else None

    if not os.path.isdir(run_dir):
        prog.damage.append("run directory does not exist")
        return prog

    # --- how far the generations themselves are trustworthy ---------------------------------------
    gen_dirs = _generation_dirs(run_dir)
    prog.generations_on_disk = len(gen_dirs)
    groups, group_size = cfg.get("groups_per_batch"), cfg.get("group_size")
    metas: list[dict] = []
    gen = 0
    while gen in gen_dirs:
        meta, damage = _check_generation(gen, gen_dirs[gen], groups, group_size)
        if meta is None:
            prog.damage.append(damage)
            break
        metas.append(meta)
        gen += 1
    good = len(metas)
    if good == 0 and not gen_dirs:
        prog.damage.append("no generations on disk")

    # --- the ICL context pool must cover exactly those generations --------------------------------
    # The pool is appended once per generation with that generation's valid solutions, so after
    # generation g it holds sum(valid_candidates[0..g]) lines. Fewer lines than that means the file
    # was truncated or deleted, and resuming on it would quietly shrink every later prompt.
    valid_per_gen = [int(((m.get("stats") or {}).get("valid_candidates")) or 0) for m in metas]
    pool_path = os.path.join(run_dir, "buffer", "context_pool.jsonl")
    pool_lines = _count_lines(pool_path)
    n_context = cfg.get("n_context")
    if good and (n_context is None or n_context > 0):
        covered, total = 0, 0
        for i, v in enumerate(valid_per_gen):
            total += v
            if total > pool_lines:
                break
            covered = i + 1
        if covered < good:
            prog.damage.append(
                f"buffer/context_pool.jsonl holds {pool_lines} solutions, enough for {covered} "
                f"generation(s) of the {good} on disk"
                + ("" if os.path.exists(pool_path) else " (file is missing)"))
            good = covered

    # --- a run that graded nothing valid is not a finished run ---------------------------------
    # Every check above is structural: full groups, full complement of children, meta.json written.
    # A run whose evaluator was broken end to end passes all of them — its candidates came back and
    # failed, which is what an ordinary generation looks like from the outside — so it verified as
    # COMPLETE and --resume skipped it. `icl.loop` now stops such a run at --max-empty-generations,
    # but that does not help the run dirs already on disk, and yield is cheap to check here.
    barren = good > 0 and sum(valid_per_gen[:good]) == 0
    if barren:
        prog.damage.append(
            f"none of the {good} generation(s) on disk produced a single valid candidate — the "
            f"evaluator was failing, so this run holds no result whatever its summary says")

    prog.good_generations = good

    # --- the buffer snapshot is what --resume-step actually loads ---------------------------------
    steps = _snapshot_steps(run_dir)
    prog.snapshots = sorted(steps)
    if good in steps:
        prog.resume_step = good
    else:
        usable = [s for s in steps if s < good]
        prog.resume_step = max(usable) if usable else 0
        if good:
            have = ", ".join(str(s) for s in sorted(steps)) or "none"
            prog.damage.append(
                f"no usable PUCT snapshot for step {good} (have: {have}) — resuming at "
                f"{prog.resume_step} instead")

    if barren:
        prog.complete = False                # never skip a run that holds nothing
    elif want is not None:
        prog.complete = prog.good_generations >= want
    else:
        # No configured target (no config.json, and the caller did not know either): the only claim
        # left is the summary's, and it counts only where the artifacts reach as far as it says.
        prog.complete = (prog.summary_status == "complete"
                         and prog.summary_generations > 0
                         and prog.good_generations >= prog.summary_generations)

    if prog.summary_generations != prog.good_generations:
        prog.damage.append(
            f"summary.json records {prog.summary_generations} generation(s), {prog.good_generations} "
            f"are verifiable on disk")
    return prog


# --------------------------------------------------------------------------------------------------
# rewinding: move the discarded tail aside instead of appending on top of it
# --------------------------------------------------------------------------------------------------
def _stale_dir(run_dir: str, stamp: str | None = None) -> str:
    stamp = stamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(run_dir, f"stale_{stamp}")


def _move(src: str, dest: str) -> None:
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest):
        shutil.rmtree(dest) if os.path.isdir(dest) else os.remove(dest)
    shutil.move(src, dest)


def _split_lines(path: str, keep_pred, dest: str) -> int:
    """Rewrite ``path`` with only the lines ``keep_pred`` accepts; the rest go to ``dest``. Returns
    how many lines moved. The rewrite is atomic, so an interrupted rewind cannot shorten the run."""
    if not os.path.exists(path):
        return 0
    kept, dropped = [], []
    with open(path) as f:
        for line in f:
            (kept if keep_pred(line) else dropped).append(line)
    if not dropped:
        return 0
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "a") as f:
        f.writelines(dropped)
    tmp = path + ".rewind.tmp"
    with open(tmp, "w") as f:
        f.writelines(kept)
    os.replace(tmp, path)
    return len(dropped)


def _keep_by_json_field(field_name: str, keep: int):
    def pred(line: str) -> bool:
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return False                       # unparseable: it belongs to the tail, not the prefix
        value = rec.get(field_name)
        return isinstance(value, int) and value < keep
    return pred


def rewind(run_dir: str, keep: int, *, stamp: str | None = None) -> list[str]:
    """Move everything belonging to generation >= ``keep`` into ``<run_dir>/stale_<timestamp>/``.

    Nothing is deleted: the tail keeps its layout under the stale dir, so a rewind can be inspected
    (or reversed by hand) afterwards. Returns one human-readable line per thing moved.
    """
    if not os.path.isdir(run_dir):
        return []
    stale = _stale_dir(run_dir, stamp)
    moved: list[str] = []

    # 1. whole generation directories, and the solutions those generations produced
    rows = _manifest_rows(run_dir)
    tail_sols = {r.get("sol") for r in rows if isinstance(r.get("gen"), int) and r["gen"] >= keep}
    for gen, path in sorted(_generation_dirs(run_dir).items()):
        if gen >= keep:
            _move(path, os.path.join(stale, "generations", os.path.basename(path)))
            moved.append(f"generations/gen_{gen:04d}/")
    for sol in sorted(s for s in tail_sols if s):
        src = os.path.join(run_dir, "solutions", f"{sol}.py")
        if os.path.exists(src):
            _move(src, os.path.join(stale, "solutions", f"{sol}.py"))
    if tail_sols:
        moved.append(f"{len(tail_sols)} solution file(s)")

    # 2. buffer snapshots for steps past the resume point (a step-N snapshot IS generation N-1's end)
    for path in sorted(glob.glob(os.path.join(run_dir, "buffer", "puct_sampler_step_*.json"))):
        m = _SNAPSHOT_RE.search(os.path.basename(path))
        if m and int(m.group(1)) > keep:
            _move(path, os.path.join(stale, "buffer", os.path.basename(path)))
            moved.append(f"buffer/{os.path.basename(path)}")
    for path in glob.glob(os.path.join(run_dir, "buffer", "*.tmp.*")):
        _move(path, os.path.join(stale, "buffer", os.path.basename(path)))
        moved.append(f"buffer/{os.path.basename(path)} (torn write)")

    # 3. the append-only files: keep the prefix, move the tail's lines out
    n = _split_lines(os.path.join(run_dir, "events.jsonl"),
                     _keep_by_json_field("generation", keep),
                     os.path.join(stale, "events.jsonl"))
    if n:
        moved.append(f"{n} line(s) of events.jsonl")
    n = _split_lines(os.path.join(run_dir, "solutions", "manifest.jsonl"),
                     _keep_by_json_field("gen", keep),
                     os.path.join(stale, "solutions", "manifest.jsonl"))
    if n:
        moved.append(f"{n} line(s) of solutions/manifest.jsonl")

    def _keep_progress_row(line: str) -> bool:
        first = line.split(",", 1)[0].strip()
        if first == "generation":
            return True                                  # the header stays
        try:
            return int(first) < keep
        except ValueError:
            return False
    n = _split_lines(os.path.join(run_dir, "progress.csv"), _keep_progress_row,
                     os.path.join(stale, "progress.csv"))
    if n:
        moved.append(f"{n} row(s) of progress.csv")

    # 4. the context pool: trim to exactly the solutions the kept generations produced, so the
    #    resumed run's prompts see the same context they would have seen without the interruption.
    metas = [_read_json(os.path.join(run_dir, "generations", f"gen_{g:04d}", "meta.json"))
             for g in range(keep)]
    want_lines = sum(int(((m.get("stats") or {}).get("valid_candidates")) or 0) for m in metas if m)
    pool = os.path.join(run_dir, "buffer", "context_pool.jsonl")
    # Only trim when every kept generation reported its count: without them all, ``want_lines`` would
    # undercount and the trim would throw away context the kept generations DID produce.
    if os.path.exists(pool) and all(m for m in metas):
        with open(pool) as f:
            lines = [line for line in f if line.strip()]
        if len(lines) > want_lines:
            os.makedirs(stale, exist_ok=True)
            with open(os.path.join(stale, "context_pool.jsonl"), "a") as f:
                f.writelines(lines[want_lines:])
            tmp = pool + ".rewind.tmp"
            with open(tmp, "w") as f:
                f.writelines(lines[:want_lines])
            os.replace(tmp, pool)
            moved.append(f"{len(lines) - want_lines} line(s) of buffer/context_pool.jsonl")

    # 5. summary.json: keep a copy of what it claimed, then trim it to the kept generations. The
    #    relaunched run rewrites it from prior_state(); this only matters if no relaunch follows.
    summary_path = os.path.join(run_dir, "summary.json")
    summary = _read_json(summary_path)
    if summary is not None and len(summary.get("per_generation") or []) > keep:
        os.makedirs(stale, exist_ok=True)
        shutil.copy2(summary_path, os.path.join(stale, "summary.json"))
        prior = prior_state(run_dir, keep)
        summary["per_generation"] = prior.per_generation
        summary["totals"] = {**(summary.get("totals") or {}), **prior.totals}
        summary["best"], summary["worst_valid"] = prior.best, prior.worst_valid
        summary["status"] = "rewound"
        summary["rewound_to_generation"] = keep
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        moved.append(f"summary.json trimmed to {keep} generation(s)")

    if moved:
        moved.append(f"-> {os.path.relpath(stale, run_dir)}/")
    return moved


# --------------------------------------------------------------------------------------------------
# prior state: what the tracker must reload so a resumed run keeps ONE cumulative summary
# --------------------------------------------------------------------------------------------------
@dataclass
class PriorState:
    """Everything ``ExperimentTracker`` needs to continue a run's books at generation ``keep``.

    Rebuilt from the per-generation ``meta.json`` files and ``solutions/manifest.jsonl`` rather than
    from summary.json: those are written once each and are what a rewind has already made consistent,
    while the summary is the file a previous broken resume overwrote.
    """
    per_generation: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=lambda: dict.fromkeys(USAGE_KEYS, 0))
    totals: dict = field(default_factory=dict)
    candidates: int = 0
    succeeded: int = 0
    failed: int = 0
    failure_types: dict = field(default_factory=dict)
    sol_seq: int = 0
    state_to_sol: dict = field(default_factory=dict)
    best: dict | None = None
    worst_valid: dict | None = None
    started_at: str | None = None


def prior_state(run_dir: str, keep: int) -> PriorState:
    prior = PriorState()
    if keep <= 0:
        return prior

    for gen in range(keep):
        meta = _read_json(os.path.join(run_dir, "generations", f"gen_{gen:04d}", "meta.json"))
        stats = (meta or {}).get("stats")
        if not stats:
            continue
        prior.per_generation.append(stats)
        prior.succeeded += int(stats.get("valid_candidates") or 0)
        prior.failed += int(stats.get("failed_candidates") or 0)
        for k, v in (stats.get("failure_types") or {}).items():
            prior.failure_types[k] = prior.failure_types.get(k, 0) + int(v)
        for k in USAGE_KEYS:                    # run totals stay the sum of the per-generation ones
            prior.usage[k] += int((stats.get("usage") or {}).get(k) or 0)
    prior.candidates = prior.succeeded + prior.failed

    for row in _manifest_rows(run_dir):
        gen, sol = row.get("gen"), row.get("sol")
        if not isinstance(gen, int) or gen >= keep or not sol:
            continue
        m = _SOL_RE.match(sol)
        if m:
            prior.sol_seq = max(prior.sol_seq, int(m.group(1)))
        if row.get("state_id"):
            prior.state_to_sol[row["state_id"]] = sol
        value, score = row.get("value"), row.get("raw_score")
        if value is None or score is None:
            continue
        entry = {"score": score, "rank_value": value, "sol": sol, "generation": gen}
        if prior.best is None or value > prior.best["rank_value"]:
            prior.best = entry
        if prior.worst_valid is None or value < prior.worst_valid["rank_value"]:
            prior.worst_valid = dict(entry)

    prior.totals = {
        "candidates": prior.candidates,
        "succeeded": prior.succeeded,
        "failed": prior.failed,
        "success_rate": round(prior.succeeded / prior.candidates, 4) if prior.candidates else 0.0,
        "unique_solutions": prior.sol_seq,
        "failure_types": dict(prior.failure_types),
    }
    cfg = _read_json(os.path.join(run_dir, "config.json")) or {}
    prior.started_at = (cfg.get("_meta") or {}).get("created_at")
    if prior.started_at is None:
        prior.started_at = (_read_json(os.path.join(run_dir, "summary.json")) or {}).get("started_at")
    return prior


def main(argv: list[str] | None = None) -> int:
    """``python -m results.resume <run_dir> [--num-generations N] [--rewind]`` — inspect one run dir
    (``run_sweep.py --status`` does this for a whole sweep)."""
    import argparse

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("run_dir")
    p.add_argument("--num-generations", type=int, default=None,
                   help="Override the run's configured generation count.")
    p.add_argument("--rewind", action="store_true",
                   help="Move everything after the resume point into stale_<timestamp>/.")
    a = p.parse_args(argv)

    prog = inspect_run(a.run_dir, a.num_generations)
    print(f"{a.run_dir}: {prog.describe()}")
    print(f"  verified generations : {prog.good_generations}")
    print(f"  resume-step          : {prog.resume_step}")
    print(f"  summary.json says    : {prog.summary_status} ({prog.summary_generations} generations)")
    for line in prog.damage:
        print(f"  ! {line}")
    if a.rewind and not prog.complete:
        for line in rewind(a.run_dir, prog.resume_step):
            print(f"  moved {line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
