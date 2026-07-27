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
                "pid": os.getpid(), "spins": x}
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
    try:
        results = ray.get([fn.remote(group_size) for _ in range(n)], timeout=600)
    except Exception as e:
        fail(f"tasks did not complete: {type(e).__name__}: {e}")
        return
    wall = time.perf_counter() - t0

    ok(f"{len(results)}/{n} tasks completed in {wall:.1f}s")

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
    info(f"single warm task round trip: {rt:.2f}s = 0.30s of work + {rt - 0.3:.2f}s of Ray "
         "dispatch/pickle/actor RPCs (INESC reference: ~0.06s. Seconds here = unhealthy Ray)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exec", type=int, default=None, metavar="N",
                    help="run N sandboxed probe tasks end to end (0 = one per CPU group). "
                         "Creates the cpu_scheduler actor if absent.")
    ap.add_argument("--num-cpus-per-task", type=int, default=1,
                    help="the value your sweep sets; the actor's group_size is checked against it "
                         "(default: 1, what every current sweep file uses)")
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

    print(f"\n{'=' * 60}\n{_fails} FAIL, {_warns} WARN\n")
    if _fails:
        print("Ray is NOT set up the way the sandbox expects — see the FAIL lines above.\n")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
