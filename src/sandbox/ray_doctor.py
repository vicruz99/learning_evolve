"""Check that Ray + the ``cpu_scheduler`` actor are set up the way the sandbox needs.

Every failure mode this looks for is SILENT — the runs keep going and just grade slowly, or grade
at half the box, or (worst) two runs oversubscribe the same cores while Ray reports everything fine.
So this is meant to be run once on a new machine, and again whenever grading looks slower than the
core count says it should be.

    cd src
    python -m sandbox.ray_doctor              # read-only: safe on a live box
    python -m sandbox.ray_doctor --exec 8     # + actually run 8 sandboxed tasks end to end
    python -m sandbox.ray_doctor --exec 0     # --exec with 0 = saturate (one task per CPU group)

Read-only mode touches nothing: it connects, reads cluster resources, and asks the existing actor
for its stats. It never CREATES the actor — creating it is what a real run does, and doing it here
would paper over "the actor is missing" and bake in a group size the sweep did not ask for.

Exit status is 0 if every check passed, 1 if any FAIL was printed.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time

RAY_NAMESPACE = "icl"

_fails = 0
_warns = 0


def ok(msg: str) -> None:
    print(f"  \033[32mOK  \033[0m {msg}")


def warn(msg: str) -> None:
    global _warns
    _warns += 1
    print(f"  \033[33mWARN\033[0m {msg}")


def fail(msg: str) -> None:
    global _fails
    _fails += 1
    print(f"  \033[31mFAIL\033[0m {msg}")


def info(msg: str) -> None:
    print(f"       {msg}")


def section(title: str) -> None:
    print(f"\n=== {title} ===")


# -------------------------------------------------------------------------------------------------
# 1. is there a shared head, and is there exactly one?
# -------------------------------------------------------------------------------------------------
def check_head() -> "object | None":
    import ray

    section("1. Ray head")

    # Count raylets on this box BEFORE connecting. More than one means several private per-run
    # clusters are up, each believing it owns all the cores -> the box is oversubscribed by
    # (n_clusters)x and Ray cannot arbitrate, because arbitration only happens within one cluster.
    try:
        out = subprocess.run(["pgrep", "-fa", "ray/core/src/ray/raylet/raylet"],
                             capture_output=True, text=True).stdout.strip()
        n_raylets = len([l for l in out.splitlines() if l.strip()])
    except Exception:
        n_raylets = -1

    if n_raylets == 0:
        info("No raylet running yet — nothing to connect to. That is fine before a sweep starts;")
        info("`ray start --head` first if you plan to run more than one experiment at a time.")
    elif n_raylets == 1:
        ok("exactly 1 raylet on this host")
    elif n_raylets > 1:
        fail(f"{n_raylets} raylets running on this host — several independent Ray clusters. "
             "Each one thinks it owns every core, so total grading processes = n_clusters x cores "
             "and everything thrashes. Fix: stop all runs, `ray stop --force`, `ray start --head`, "
             "relaunch.")

    try:
        ray.init(address="auto", namespace=RAY_NAMESPACE, include_dashboard=False,
                 log_to_driver=False, logging_level=logging.WARNING)
        ok("connected to a SHARED head via address='auto' — this is what init_ray() will do, so "
           "concurrent runs will share one CPU pool")
    except ConnectionError:
        fail("address='auto' failed: no shared head. init_ray() falls back to a PRIVATE cluster "
             "per run. One run is fine; two or more concurrent runs will each boot their own head "
             "and oversubscribe the box (this is the documented hang). Fix: `ray start --head`.")
        return None
    except Exception as e:  # a genuinely broken install lands here
        fail(f"ray.init(address='auto') raised {type(e).__name__}: {e}")
        return None

    return ray


# -------------------------------------------------------------------------------------------------
# 2. does Ray see the cores the machine actually has?
# -------------------------------------------------------------------------------------------------
def check_resources(ray) -> int:
    section("2. Cluster resources")

    total = ray.cluster_resources()
    avail = ray.available_resources()
    ray_cpus = int(total.get("CPU", 0))

    # os.sched_getaffinity, not os.cpu_count(): under a cpuset/cgroup (Slurm, containers — likely on
    # a shared cluster box) the process may be confined to a subset, and the cpu_scheduler partitions
    # exactly THAT set, while Ray's CPU resource is its own independent number.
    affinity = len(os.sched_getaffinity(0))
    nproc = os.cpu_count() or 0

    info(f"Ray CPU={ray_cpus}  available now={avail.get('CPU', 0):g}   "
         f"GPU={total.get('GPU', 0):g}   nodes={len([n for n in ray.nodes() if n['Alive']])}")
    if ray_cpus and avail.get("CPU", 0) < 0.05 * ray_cpus:
        info("cluster is FULLY COMMITTED right now — every section below will measure contention as "
             "well as health. Re-run when idle before concluding anything is misconfigured.")
    info(f"this process: sched_getaffinity={affinity}  os.cpu_count()={nproc}")

    if ray_cpus == 0:
        fail("Ray reports 0 CPUs")
    elif abs(ray_cpus - affinity) > 1:
        warn(f"Ray thinks it has {ray_cpus} CPUs but this process is confined to {affinity}. "
             "Ray will admit ~{0} concurrent evals while the cpu_scheduler only has "
             "{1}//group_size groups to hand out, so tasks will sit in get_cpu_group() polling "
             "instead of running. Watch queue_p50. Fix: `ray start --head --num-cpus={1}`."
             .format(ray_cpus, affinity))
    else:
        ok(f"Ray CPU count ({ray_cpus}) matches this process's affinity ({affinity})")

    # A cgroup CPU QUOTA is invisible to everything above: sched_getaffinity still reports every
    # core, so the cpu_scheduler happily hands out cores//group_size groups and queue_seconds stays
    # ~0, while the kernel throttles the cgroup's total CPU time. The only symptom is that each eval
    # runs several times slower than it should — exactly the shape of "Ray feels broken" that isn't
    # Ray at all. Check it explicitly.
    checked_quota = False
    for path in ("/sys/fs/cgroup/cpu.max",                       # cgroup v2
                 "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):        # cgroup v1
        try:
            with open(path) as fh:
                raw = fh.read().split()
        except OSError:
            continue
        checked_quota = True
        try:
            if path.endswith("cpu.max"):
                if raw[0] == "max":
                    ok("no cgroup CPU quota (cpu.max = max)")
                    break
                allowed = float(raw[0]) / float(raw[1])
            else:
                quota = float(raw[0])
                if quota < 0:
                    ok("no cgroup CPU quota (cfs_quota_us = -1)")
                    break
                with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
                    allowed = quota / float(fh.read().strip())
        except (ValueError, IndexError, OSError):
            break
        if allowed < affinity * 0.9:
            fail(f"cgroup CPU quota allows only {allowed:.1f} cores of CPU TIME, but affinity "
                 f"exposes {affinity}. The cpu_scheduler will hand out a group per exposed core and "
                 f"queue_seconds will read ~0, while every eval runs up to ~{affinity / allowed:.1f}x "
                 "slow because the kernel throttles the whole cgroup. This is the failure that looks "
                 "like a Ray problem and is not one. Fix: get a real core allocation, or "
                 f"`ray start --head --num-cpus={max(1, int(allowed))}` so concurrency matches the "
                 "CPU time you actually have.")
        else:
            ok(f"cgroup CPU quota ({allowed:.1f} cores) is consistent with affinity ({affinity})")
        break
    if not checked_quota:
        info("no readable cgroup cpu.max / cfs_quota_us — could not rule out CPU-time throttling "
             "this way. If evals are slow with queue_seconds ~0, check section 5 and `cat "
             "/proc/pressure/cpu`.")

    if len([n for n in ray.nodes() if n["Alive"]]) > 1:
        warn("more than one live node — the cpu_scheduler keys groups by node IP and partitions "
             "each host separately, which is correct, but nothing pins a task to the node whose "
             "cores it was given. Multi-node was never tested in this project.")

    return ray_cpus


# -------------------------------------------------------------------------------------------------
# 3. the cpu_scheduler actor: exists, right group size, no leaked groups
# -------------------------------------------------------------------------------------------------
def check_scheduler(ray, ray_cpus: int, expect_group_size: int | None) -> "tuple[object, int] | None":
    section("3. cpu_scheduler actor")

    try:
        actor = ray.get_actor("cpu_scheduler")
    except ValueError:
        info("actor does not exist yet — normal on an idle box; the first run creates it.")
        info("Re-run this while a sweep is up to check its group size, or pass --exec to create "
             "one now with --num-cpus-per-task.")
        return None

    ok("actor found (detached, namespace='icl')")

    try:
        stats = ray.get(actor.stats.remote(), timeout=30)
    except Exception as e:
        fail(f"actor is unresponsive: {type(e).__name__}: {e}. A hung actor blocks EVERY eval, "
             "because get_cpu_group() polls it once a second forever. Fix: `ray stop` and restart.")
        return None

    if not stats:
        info("actor exists but has never handed out a group (no host initialised yet)")
        return actor, -1

    group_size = -1
    for host, s in stats.items():
        group_size = s["group_size"]
        # Groups are created lazily per host on first request, and the trailing partial group is
        # dropped, so this is the real ceiling on concurrent evals.
        expected_total = ray_cpus // group_size
        info(f"host {host}: group_size={group_size}  free groups={s['available_groups']} "
             f"(ceiling on concurrent evals ~{expected_total})")

        if s["available_groups"] > expected_total:
            warn(f"more free groups ({s['available_groups']}) than the core count implies "
                 f"({expected_total}) — release_workers_atomic appends unconditionally, so a "
                 "double release inflates the queue and lets evals oversubscribe cores.")

        # The opposite leak, and the more common one: a worker SIGKILLed mid-eval never runs its
        # release, so its group is lost from the deque for the head's lifetime. Ray's own accounting
        # recovers (it frees the reservation when the task dies) while the actor's does not, so the
        # signature is "Ray says the CPUs are free, the actor says the groups are gone" — and every
        # future run silently grades at reduced parallelism. Only meaningful on an idle cluster.
        idle = ray.available_resources().get("CPU", 0) >= 0.95 * ray_cpus
        if idle and 0 <= s["available_groups"] < expected_total:
            leaked = expected_total - s["available_groups"]
            fail(f"{leaked} of {expected_total} CPU groups are LEAKED: the cluster is idle "
                 f"(Ray reports all {ray_cpus} CPUs free) but only {s['available_groups']} groups are "
                 "available. Killed evals never returned them, and a detached actor keeps that state "
                 f"for the head's lifetime — so the next run will grade {s['available_groups']}-way "
                 f"instead of {expected_total}-way. Fix: `ray stop && ray start --head`.")

    # THE stale-actor trap: num_cpus_per_task is frozen at creation, and init_ray uses
    # get_if_exists=True, so a sweep asking for a different value is silently given the old one.
    # Ray then RESERVES what the sweep asked for while the actor PINS the old group size — the exact
    # mismatch that wasted half the first campaign.
    if expect_group_size is not None and group_size > 0:
        if group_size == expect_group_size:
            ok(f"group_size == --num-cpus-per-task ({expect_group_size})")
        else:
            fail(f"group_size={group_size} but your sweep sets num-cpus-per-task="
                 f"{expect_group_size}. The actor is STALE (created by an earlier run) and "
                 "get_if_exists=True reuses it silently. Ray will reserve "
                 f"{expect_group_size} cores per eval while the child is pinned to {group_size}. "
                 "Fix: stop the runs, `ray stop && ray start --head`, relaunch.")

    return actor, group_size


# -------------------------------------------------------------------------------------------------
# 4. end to end: acquire a group, pin a child to it, release
# -------------------------------------------------------------------------------------------------
def _probe(group_size: int) -> dict:
    """Body of the exec test: the same acquire -> pin child -> release path run_program uses."""
    import ray as _ray
    from sandbox.cpu_scheduler import get_cpu_group, release_cpu_group

    # Wall clock (not perf_counter): the driver compares this against its own submit time to separate
    # "Ray took a long time to START me" (cluster full) from "Ray is slow" (broken). Same host, so
    # time.time() is comparable across the two processes; perf_counter would not be.
    t_task_start = time.time()

    t0 = time.perf_counter()
    actor = _ray.get_actor("cpu_scheduler")
    group = get_cpu_group(actor, timeout_s=120)
    t_queue = time.perf_counter() - t0

    try:
        # Pin THIS worker, then read the affinity back from a fresh child. The evaluator relies on
        # inheritance (it sets affinity in the generated program's preamble), so verifying it in the
        # child is what actually matters — verifying it in the parent proves nothing about the eval.
        if hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, set(group))
        child = subprocess.run(
            [sys.executable, "-c", "import os;print(','.join(map(str,sorted(os.sched_getaffinity(0)))))"],
            capture_output=True, text=True, timeout=120)
        child_cpus = sorted(int(c) for c in child.stdout.strip().split(",") if c)
        # ~0.3 s of real work so overlapping probes actually contend for cores
        spin_t0 = time.perf_counter()
        x = 0
        while time.perf_counter() - spin_t0 < 0.3:
            x += 1
        return {"group": group, "child_cpus": child_cpus, "queue_s": t_queue,
                "pid": os.getpid(), "spins": x, "t_task_start": t_task_start}
    finally:
        # ALWAYS release. A group that is not returned is lost from the deque for the lifetime of
        # the head, permanently shrinking every future run's grading parallelism.
        release_cpu_group(_ray.get_actor("cpu_scheduler"), group)


def check_exec(ray, n: int, ray_cpus: int, num_cpus_per_task: int) -> None:
    section(f"4. End-to-end sandbox path ({n} tasks)")

    from sandbox.cpu_scheduler import CpuScheduler, current_host

    # Create the actor only if missing, exactly as init_ray does.
    CpuScheduler.options(name="cpu_scheduler", lifetime="detached",
                         get_if_exists=True).remote(num_cpus_per_task=num_cpus_per_task,
                                                    num_persistent_workers=0)
    actor = ray.get_actor("cpu_scheduler")
    group_size = ray.get(actor.stats.remote()).get(current_host(), {}).get(
        "group_size", num_cpus_per_task)

    fn = ray.remote(num_cpus=group_size, max_calls=0)(_probe)

    t0 = time.perf_counter()
    submit_wall = time.time()
    try:
        results = ray.get([fn.remote(group_size) for _ in range(n)], timeout=3600)
    except Exception as e:
        fail(f"tasks did not complete: {type(e).__name__}: {e}")
        return
    wall = time.perf_counter() - t0

    ok(f"{len(results)}/{n} tasks completed in {wall:.1f}s")

    # Admission wait is the single most misread number here. A task that sits for minutes before it
    # even STARTS means the cluster is fully committed to other work (a live sweep), NOT that Ray is
    # broken. Crucially, this wait is invisible to queue_seconds -- that clock only starts once the
    # task is already running -- so a saturated cluster shows queue_seconds ~0 while candidates wait
    # ages. It IS included in grade_seconds, so grade_seconds - eval_seconds is where it surfaces.
    adm = sorted(r["t_task_start"] - submit_wall for r in results)
    info(f"Ray admission wait (submit -> task starts): p50 {adm[len(adm)//2]:.1f}s  max {adm[-1]:.1f}s")
    if adm[-1] > 30:
        warn(f"tasks waited up to {adm[-1]:.0f}s just to be ADMITTED. Ray's CPU pool is fully "
             "committed (check 'available now' in section 2). Expected while a sweep is running; it "
             "is contention, not a misconfiguration — and note queue_seconds cannot see it.")

    bad_pin = [r for r in results if r["child_cpus"] != sorted(r["group"])]
    if bad_pin:
        fail(f"{len(bad_pin)}/{n} children were NOT pinned to their assigned group "
             f"(e.g. assigned {bad_pin[0]['group']}, child saw {bad_pin[0]['child_cpus'][:8]}...). "
             "Without pinning, every eval is free to roam all cores and they fight for cache; this "
             "is also how a box ends up with 96 processes on 48 hot cores.")
    else:
        ok("every child process inherited exactly its assigned cores (affinity pinning works)")

    # Distinct groups across CONCURRENT tasks is the mutual-exclusion property. If two overlapping
    # tasks are handed the same cores, the deque is corrupted (double release) and evals collide.
    groups = [tuple(r["group"]) for r in results]
    if n <= ray_cpus // max(1, group_size) and len(set(groups)) != len(groups):
        dupes = len(groups) - len(set(groups))
        fail(f"{dupes} core group(s) handed to more than one task while all {n} ran concurrently "
             "— groups are not mutually exclusive.")
    else:
        ok("no core group was handed to two concurrent tasks")

    qs = sorted(r["queue_s"] for r in results)
    info(f"get_cpu_group() wait: p50 {qs[len(qs)//2]:.2f}s  max {qs[-1]:.2f}s")
    if qs[-1] > 5 and n <= ray_cpus // max(1, group_size):
        warn("tasks waited for a core group even though there should be one free per task — "
             "either a live sweep is holding them (fine) or groups have leaked (not fine). "
             "Re-check on an idle box.")

    # Serial baseline for Ray's own dispatch cost. This is a LOWER BOUND on the harness overhead a
    # real candidate pays: it excludes writing the program file, spawning a fresh interpreter,
    # importing numpy, and pickling the result back — which on an NFS run dir is most of the cost.
    # Measured on the INESC box: ~0.06 s here versus a grade_seconds - eval_seconds gap of ~2 s.
    # Useful as a red flag: if THIS is slow, Ray itself is unhealthy.
    t0 = time.perf_counter()
    ray.get(fn.remote(group_size))
    rt = time.perf_counter() - t0
    info(f"single task round trip: {rt:.2f}s (0.30s of it is work). On an IDLE cluster the remainder "
         "is Ray's dispatch cost — INESC reference 0.06s. On a BUSY cluster it is mostly admission "
         "wait, so read it together with the admission line above, never as an absolute.")

    # THE HARNESS FLOOR. Every eval pays a fresh interpreter + `import numpy` before running a single
    # line of the candidate's algorithm, and it pays it from wherever the venv lives. This is the
    # number that explains a slow box when Ray itself is healthy: on the INESC box the CHEAPEST
    # measured eval of 3,198 candidates was 0.34s end to end, so the floor there is ~0.3s. A floor of
    # seconds means an NFS-mounted venv, and it is charged to eval_seconds — i.e. it eats the
    # eval-timeout budget and inflates every percentile you would use to tune it.
    section("5. Harness floor (interpreter + numpy import, serial)")
    t0 = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import numpy"], capture_output=True, timeout=300)
    warm = time.perf_counter() - t0
    info(f"`{sys.executable} -c 'import numpy'` = {warm:.2f}s serial and warm")
    info(f"venv lives on: {sys.prefix}")
    # Reference: 0.50s on the INESC box, whose venv is ITSELF on an NFS home. So anything much above
    # that is worse than "NFS", not merely "not local disk".
    if warm > 2.0:
        fail(f"{warm:.2f}s to start Python and import numpy, before ANY candidate code runs. "
             "INESC reference: 0.50s — and that venv is on NFS too, so this is worse than NFS. "
             "Every eval pays this, it counts against --eval-timeout, and it multiplies under "
             "concurrency. Fix: put the venv AND --sweep-dir on local disk.")
    elif warm > 1.0:
        warn(f"{warm:.2f}s to start Python and import numpy — 2x+ the 0.50s INESC reference, and it "
             "multiplies under concurrency. Local disk would remove it.")
    else:
        ok(f"{warm:.2f}s interpreter+numpy startup — negligible against real evals")


# Reference timings, INESC box (Xeon Gold 6330 @ 2.00GHz, numpy 2.5.1 / scipy 1.18.0 /
# scipy-openblas), pinned to ONE core exactly as the sandbox pins an eval child.
CORE_REF = {"pyloop": 0.40, "blas": 0.49, "slsqp": 2.01}


def check_core_speed() -> None:
    """Per-core compute speed. This is what explains a slow box once Ray and the floor are clean.

    Three fixed workloads, because they separate the causes that a single number cannot:
      pyloop -> raw integer/branch speed, no libraries involved  => the CPU itself
      blas   -> BLAS build and its thread behaviour              => numeric stack
      slsqp  -> the actual circle-packing workload               => what evals really pay
    Measured on Bosch vs INESC: identical programs ran 5.5x slower (paired, n=19, median ratio 0.18),
    with Ray healthy and the harness floor equal — i.e. neither Ray nor the venv explained it.
    """
    section("6. Per-core compute speed (pinned to 1 core, like an eval child)")

    # The trap: sandbox_reward_evaluator uses env.setdefault() for the thread caps, so a value already
    # exported by the shell, a module system, or venv activation WINS. >1 on a 1-core-pinned eval means
    # BLAS oversubscribes that single core and thrashes — a 5x-class slowdown with no Ray or NFS
    # signature, which is exactly the shape of an unexplained slow box.
    names = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")
    inherited = {k: os.environ[k] for k in names if k in os.environ}
    bad = {k: v for k, v in inherited.items() if v.strip() not in ("1", "")}
    if bad:
        fail("these thread caps are ALREADY set in the environment: "
             + ", ".join(f"{k}={v}" for k, v in bad.items())
             + ". sandbox_reward_evaluator.py uses env.setdefault(), so these PRE-SET values win over "
               "its own cap of --num-cpus-per-task. Every eval pinned to one core will spawn that many "
               "BLAS threads onto it and thrash. Unset them (or export them as 1) before launching.")
    elif inherited:
        ok("inherited thread caps are all 1 — no BLAS oversubscription")
    else:
        info("no thread caps inherited — the sandbox will set them to --num-cpus-per-task itself")

    pinned = sorted(os.sched_getaffinity(0))[:1] if hasattr(os, "sched_setaffinity") else []
    code = r'''
import os, sys, time
if %(pin)r and hasattr(os, "sched_setaffinity"):
    os.sched_setaffinity(0, set(%(pin)r))          # pin BEFORE importing numpy, like the preamble
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
N = 26
def pyloop():
    t = time.perf_counter(); s = 0
    for i in range(3_000_000): s += i %% 7
    return time.perf_counter() - t
def blas():
    a = np.random.default_rng(0).random((900, 900)); b = np.random.default_rng(1).random((900, 900))
    t = time.perf_counter()
    for _ in range(12): a @ b
    return time.perf_counter() - t
def _obj(p): return -np.sum(p[2::3])
def _cons(p):
    c = np.column_stack((p[0::3], p[1::3])); r = p[2::3]
    out = [c[:, 0] - r, 1 - c[:, 0] - r, c[:, 1] - r, 1 - c[:, 1] - r]
    i, j = np.triu_indices(N, 1)
    out.append(np.sqrt(np.sum((c[i] - c[j]) ** 2, axis=1)) - r[i] - r[j])
    return np.concatenate(out)
def slsqp():
    rng = np.random.default_rng(0); t = time.perf_counter()
    for _ in range(3):
        p0 = np.empty(3 * N)
        p0[0::3] = rng.random(N) * .6 + .2; p0[1::3] = rng.random(N) * .6 + .2; p0[2::3] = .08
        minimize(_obj, p0, method="SLSQP", bounds=[(0, 1), (0, 1), (0, .5)] * N,
                 constraints=[NonlinearConstraint(_cons, 0, np.inf)],
                 options={"ftol": 1e-10, "maxiter": 300})
    return time.perf_counter() - t
import json
print(json.dumps({"pyloop": pyloop(), "blas": blas(), "slsqp": slsqp(),
                  "numpy": np.__version__}))
''' % {"pin": pinned}
    try:
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=900)
        got = json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:                       # scipy missing, timeout, crash — all non-fatal here
        warn(f"could not run the compute benchmark ({type(e).__name__}: {e}); skipping section 6")
        return

    info(f"numpy {got['numpy']}   pinned to core {pinned}")
    worst = 1.0
    for k in ("pyloop", "blas", "slsqp"):
        ratio = got[k] / CORE_REF[k]
        worst = max(worst, ratio)
        info(f"  {k:<7}{got[k]:7.2f}s   INESC {CORE_REF[k]:.2f}s   {ratio:5.2f}x")

    if got["pyloop"] / CORE_REF["pyloop"] > 2.0:
        fail(f"raw CPython speed is {got['pyloop'] / CORE_REF['pyloop']:.1f}x slower than the INESC "
             "reference, with no libraries involved — this is the CPU (or a shared/throttled core), "
             "not a misconfiguration. Nothing to fix: re-budget the grid, because every eval second "
             "scales by this factor and eval-timeout means something different on this box.")
    elif worst > 2.0:
        fail(f"CPython speed is fine but numeric work is up to {worst:.1f}x slower than INESC — that "
             "is the numeric stack, not the CPU: wrong/slow BLAS, or thread oversubscription (see the "
             "thread-cap check above). This one IS fixable and is worth the whole factor.")
    elif worst > 1.3:
        warn(f"up to {worst:.1f}x slower than the INESC reference — modest, but eval percentiles and "
             "any --eval-timeout tuned on the other box do not transfer.")
    else:
        ok("per-core speed comparable to the INESC reference — eval timings are portable")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exec", type=int, default=None, metavar="N",
                    help="run N sandboxed probe tasks end to end (0 = one per CPU group). "
                         "Creates the cpu_scheduler actor if absent.")
    ap.add_argument("--num-cpus-per-task", type=int, default=1,
                    help="the value your sweep sets; the actor's group_size is checked against it "
                         "(default: 1, what every current sweep file uses)")
    ap.add_argument("--no-bench", action="store_true",
                    help="skip section 6 (the ~3s single-core compute benchmark)")
    args = ap.parse_args()

    ray = check_head()
    if ray is None:
        print("\nCannot continue without a cluster.\n")
        return 1

    ray_cpus = check_resources(ray)
    sched = check_scheduler(ray, ray_cpus, args.num_cpus_per_task)

    if args.exec is not None:
        n = args.exec if args.exec > 0 else max(1, ray_cpus // max(1, args.num_cpus_per_task))
        gs = sched[1] if sched and sched[1] > 0 else args.num_cpus_per_task
        check_exec(ray, n, ray_cpus, gs)

    if not args.no_bench:
        check_core_speed()

    print(f"\n{'=' * 60}\n{_fails} FAIL, {_warns} WARN\n")
    if _fails:
        print("Ray is NOT set up the way the sandbox expects — see the FAIL lines above.\n")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
