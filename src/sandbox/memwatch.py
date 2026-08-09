"""Measure the job's memory the way the batch system does, so a run can see a TERM_MEMLIMIT coming.

Why this exists
---------------
LSF at this site runs with ``LSB_RESOURCE_ENFORCE="cpu memory gpu"`` and ``LSF_PROCESS_TRACKING=Y``:
it sums the RSS of your whole process tree itself and kills the job on the total. There is no cgroup
file carrying that number, which has two consequences:

  * ``memory.max`` reads ``max``, so Ray falls back to ``/proc/meminfo`` and sizes itself for the
    whole node. Ray's own memory monitor watches that same wrong denominator, so on a 1.5 T box with
    a 488 G job ceiling it can never fire first. Ray will not save the job.
  * Nothing in the process reports the number LSF is about to kill on. Job 12669131 died at
    ``MAX MEM 488 G`` after 28h45m and 14 generations with ``AVG MEM 78.4 G`` -- a 6x ramp that
    nobody watched, and that had to be reconstructed afterwards from ``bhist``.

So this module computes the same sum LSF does and hands it back once per generation. The cheap win is
not the auto-stop, it is the single logged line: a run that prints its own RSS every generation turns
the next failure from forensics into a glance.

Which processes count
---------------------
Not just our descendants. ``ray start --head`` daemonizes, so the raylet, the GCS and every eval
worker are reparented away from the driver and a plain process-tree walk misses precisely the
processes most likely to be accumulating. Under LSF we instead match ``LSB_JOBID`` in each process's
environment, which is exactly the set LSF accounts against the ceiling; off LSF we fall back to our
own tree plus any Ray component we own.

Summing RSS double-counts pages shared between processes (the object store mapped by N workers counts
N times). That is deliberate: LSF's accounting does the same, and the point here is to predict its
kill decision, not to measure the machine honestly.
"""
from __future__ import annotations

import os

from sandbox.ray_head import GiB, _RAY_PROC_MARKERS, _cgroup_memory_limit, _lsf_memlimit

PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")


def _rss(pid: int) -> int:
    """Resident bytes of one pid, 0 if it is gone or unreadable.

    ``statm`` rather than ``status``: one small read and a split, which matters when this walks a few
    hundred processes on every generation boundary.
    """
    try:
        with open(f"/proc/{pid}/statm") as fh:
            return int(fh.read().split()[1]) * PAGE_SIZE
    except (OSError, ValueError, IndexError):
        return 0


def _environ_has(pid: int, needle: bytes) -> bool:
    try:
        with open(f"/proc/{pid}/environ", "rb") as fh:
            return needle in fh.read()
    except OSError:
        return False


def _is_ray_component(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as fh:
            cmd = fh.read().decode("utf-8", "replace")
    except OSError:
        return False
    return any(marker in cmd for marker in _RAY_PROC_MARKERS)


def _descendants(root: int) -> set[int]:
    """``root`` and every process reachable from it by ppid.

    Built by inverting the ppid map in one pass rather than by recursing per pid: a walk that reads
    ``/proc/<pid>/stat`` once per candidate per level is quadratic in the worker count.
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                ppid = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(entry))

    out, stack = set(), [root]
    while stack:
        pid = stack.pop()
        if pid in out:
            continue
        out.add(pid)
        stack.extend(children.get(pid, ()))
    return out


def job_pids() -> tuple[set[int], str]:
    """The processes the batch system will charge to this job, and how we identified them."""
    jobid = os.environ.get("LSB_JOBID")
    if jobid:
        needle = f"LSB_JOBID={jobid}".encode()
        pids = {int(e) for e in os.listdir("/proc")
                if e.isdigit() and _environ_has(int(e), needle)}
        if pids:
            return pids, f"LSB_JOBID={jobid} in /proc/*/environ"

    # Off LSF (or LSB_JOBID not exported into the daemonised Ray processes): our own tree, plus any
    # Ray component this uid owns. The Ray half can over-count a co-tenant sweep of our own on the
    # same box; that is the honest failure mode and it errs toward reporting too much, not too little.
    pids = _descendants(os.getpid())
    uid = os.getuid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in pids:
            continue
        try:
            if os.stat(f"/proc/{pid}").st_uid != uid:
                continue
        except OSError:
            continue
        if _is_ray_component(pid):
            pids.add(pid)
    return pids, "own process tree + Ray components owned by this uid"


def ceiling() -> tuple[int | None, str]:
    """Bytes this job may use before it is killed, and where the number came from.

    Prefers the LSF ceiling over the cgroup one: where the two disagree at this site it is because
    LSF is polling and killing on a number no cgroup carries, and that is the one that ends the run.
    """
    lsf, lsf_where = _lsf_memlimit()
    if lsf:
        return lsf, lsf_where
    cg, cg_where = _cgroup_memory_limit()
    if cg:
        return cg, cg_where
    return None, "no LSF or cgroup memory ceiling detectable"


def sample(limit: int | None = None) -> dict:
    """One measurement of the job's memory against its ceiling.

    ``limit`` overrides detection, which matters because the sweep launcher has already resolved the
    ceiling once and re-deriving it per generation would shell out to ``bjobs`` each time.

    Reports TWO totals, because with several runs sharing one Ray head they answer different
    questions and only one of them is per-run:

      * ``job_rss_gb`` -- everything the batch system charges to this job, which is what it kills on.
        Every concurrent run measures the same figure, most of it the shared head and its eval
        workers. Correct for "are we about to be killed"; meaningless as "this run's memory".
      * ``own_rss_gb`` -- this driver's own process tree only. Small (the sandboxed evals run in the
        shared head's workers, not here) but genuinely attributable, so it is the one to compare
        across runs when asking which configuration is heavy.
    """
    pids, how = job_pids()
    total = sum(_rss(pid) for pid in pids)
    own = sum(_rss(pid) for pid in _descendants(os.getpid()))
    if limit is None:
        limit, where = ceiling()
    else:
        where = "supplied by the caller"

    out = {
        "job_rss_gb": round(total / GiB, 2),
        "own_rss_gb": round(own / GiB, 2),
        "n_procs": len(pids),
        "source": how,
        "ceiling_gb": round(limit / GiB, 2) if limit else None,
        "ceiling_source": where,
        "job_rss_pct": round(100.0 * total / limit, 1) if limit else None,
    }
    return out


def claim_shed(sweep_dir: str, run_name: str, cooldown: float = 900.0) -> tuple[bool, str]:
    """Decide whether THIS run is the one that stops, when several share a memory ceiling.

    Runs sharing a Ray head all measure the same ``job_rss_gb`` against the same ceiling, so without
    arbitration they all cross the threshold and all stop -- losing a whole sweep where shedding one
    run would have freed enough to let the rest finish. This is the arbitration: the first run to
    reach a generation boundary above the threshold writes a claim file and stops; siblings that see
    a fresh claim keep going, because the memory that run is about to release is exactly the relief
    they need.

    ``cooldown`` seconds after a claim, the next run above the threshold may shed too -- so a job that
    is still growing sheds runs one at a time instead of all at once or only ever one. It has to be
    long enough for the shedding run's driver and its in-flight evals to actually exit.

    Filesystem-based rather than an actor because the runs are separate processes that may not share
    a Ray namespace, and because a claim has to survive the claiming run's exit. ``O_CREAT | O_EXCL``
    makes the first writer unambiguous; on a shared filesystem this is a single-node sweep's
    directory, so ordinary POSIX create semantics are all that is required.

    Returns ``(should_stop, note)``.
    """
    import errno
    import json
    import time

    path = os.path.join(sweep_dir, ".memory_shed.json")
    now = time.time()
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except OSError as e:
        if e.errno != errno.EEXIST:
            # Cannot arbitrate (unwritable dir, etc.). Stopping is the safe default: being killed by
            # the batch system mid-generation is strictly worse than stopping one run too many.
            return True, f"could not arbitrate against sibling runs ({e!r}); stopping this run"
        try:
            with open(path) as fh:
                prev = json.load(fh)
        except Exception:
            prev = {}
        age = now - float(prev.get("at") or 0)
        if age < cooldown:
            return False, (f"{prev.get('run', 'another run')} is already shedding "
                           f"({age:.0f}s ago, {cooldown:.0f}s cooldown) — continuing, its exit is "
                           "what frees the headroom this run needs")
        # Stale claim: the previous shed did not bring the job under the threshold. Take over.
        try:
            with open(path, "w") as fh:
                json.dump({"run": run_name, "at": now}, fh)
        except OSError:
            pass
        return True, (f"the previous shed ({prev.get('run', '?')}, {age:.0f}s ago) did not bring the "
                      "job under its ceiling; shedding this run too")

    with os.fdopen(fd, "w") as fh:
        json.dump({"run": run_name, "at": now}, fh)
    return True, "first run over the threshold — shedding this one so its siblings can continue"


def describe(s: dict) -> str:
    """One log line. Deliberately terse -- it prints every generation for the whole run."""
    head = (f"rss {s['job_rss_gb']:.1f}G across {s['n_procs']} procs "
            f"(this driver's own tree {s['own_rss_gb']:.1f}G)")
    if s.get("ceiling_gb"):
        return f"{head} | {s['job_rss_pct']:.0f}% of the {s['ceiling_gb']:.0f}G ceiling"
    return f"{head} | no ceiling detected, so nothing to compare against"
