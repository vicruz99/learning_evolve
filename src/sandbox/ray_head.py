"""Own the Ray head for one sweep: size it from what the box actually grants, isolate it from every
other machine, and hand each run an explicit address.

Why the sweep starts the head instead of you
--------------------------------------------
Three things have to agree, and only the launcher knows all three at once:

  * ``OMP_NUM_THREADS`` (and its four siblings) must be set on the *``ray start`` process*, because
    eval tasks run in workers forked by the raylet and inherit ITS environment — not the driver's.
    Exporting them in the shell you launch ``run_sweep.py`` from does nothing for grading unless the
    head was started from that same shell. Getting this wrong is silent: every eval quietly spawns
    ``nproc`` BLAS threads and the box thrashes.
  * The right value for those variables is ``num-cpus-per-task``, which lives in the sweep file.
  * ``--num-cpus`` has to match what the cgroup really allows, and be a whole multiple of
    ``num-cpus-per-task``, or Ray admits more concurrent evals than ``cpu_scheduler`` has CPU groups
    to hand out and the excess tasks spin in ``get_cpu_group()`` instead of running.

The shared-/tmp trap
--------------------
``ray.init(address="auto")`` resolves in this order (ray/_private/services.py):

  1. ``RAY_ADDRESS`` if set and non-empty
  2. ``<temp_dir>/ray_current_cluster`` — the file the last ``ray start`` wrote
  3. grepping ``ps`` for a local GCS process

Step 2 beats step 3. So if ``/tmp`` is shared between machines, the head you start on machine B
overwrites ``/tmp/ray/ray_current_cluster``, and a run on machine A then reads B's IP and submits
its eval tasks **to B's cluster, over the network** — while ``ray status`` on both boxes looks fine.
This module removes the ambiguity from both ends: the head gets a per-host ``--temp-dir``, and every
run is handed an explicit ``RAY_ADDRESS`` so step 1 short-circuits discovery entirely. That is
correct whether or not ``/tmp`` turns out to be shared, so it costs nothing to always do it.

Teardown is surgical on purpose: ``ray stop`` matches Ray processes machine-wide by command line and
would kill a co-tenant's cluster (or your own head on another job). :func:`stop_head` only kills
processes whose command line contains the temp dir this module created.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field

from sandbox.ray_doctor import _cgroup_cpu_limits   # same package; the walk it does is non-trivial

GiB = 1024 ** 3

# Ray builds "<temp_dir>/session_<28 chars>/sockets/plasma_store" and a Unix socket path cannot
# exceed 107 bytes, so the temp dir itself has to stay short.
MAX_TEMP_DIR_LEN = 45

# Below this, an LSF MEMLIMIT is the site's boilerplate default rather than a budget anyone intends
# to enforce. One eval reserves 1 GiB, so honouring anything smaller would admit no tasks at all.
LSF_MEMLIMIT_FLOOR = 4 * GiB

# At or below this, a per-slot MEMLIMIT is the site's boilerplate rusage[mem=1024] rather than
# something the submitter chose, and LSF multiplies it by the slot count to get the job ceiling.
LSF_DEFAULT_PER_SLOT = 2 * GiB

# Must match SandboxRewardEvaluator.TASK_MEMORY: what each eval task reserves from Ray's `memory`
# resource. Ray admits min(cpu, memory) worth of tasks, so the heap has to cover the core count.
TASK_MEMORY = GiB


def thread_env(threads: int) -> dict[str, str]:
    """Thread-limit variables the head — and therefore every eval worker — must be started with.

    Ray sets ``OMP_NUM_THREADS`` itself when unset, but only OMP, and only to the task's num_cpus.
    MKL / OpenBLAS / NumExpr / vecLib are left at "one thread per visible core", which is what
    oversubscribes the box. Setting all five explicitly, derived from num-cpus-per-task, keeps them
    in lockstep with the sweep file instead of with whatever was last typed in a shell.
    """
    n = str(max(1, int(threads)))
    return {"OMP_NUM_THREADS": n, "MKL_NUM_THREADS": n, "OPENBLAS_NUM_THREADS": n,
            "NUMEXPR_NUM_THREADS": n, "VECLIB_MAXIMUM_THREADS": n}


def ray_bin() -> str:
    """The ``ray`` CLI belonging to the interpreter we are running under.

    Sweeps are launched as ``/path/to/.venv/bin/python run_sweep.py`` without activating the venv, so
    a bare ``ray`` is either absent from PATH or — worse — a different environment's Ray talking to
    this one's cluster. The console script sits next to the interpreter; prefer it.
    """
    candidate = os.path.join(os.path.dirname(sys.executable), "ray")
    return candidate if os.path.exists(candidate) else "ray"


# --------------------------------------------------------------------------------------------------
# what the box actually grants
# --------------------------------------------------------------------------------------------------
def _cgroup_memory_limit() -> tuple[int | None, str]:
    """Bytes this process's cgroup allows, walking leaf -> ancestors like the CPU probe does.

    Under LSF the limit is on the job's cgroup, not the root, so reading /sys/fs/cgroup/memory.max
    finds nothing. Returns (bytes_or_None, where_we_looked).
    """
    try:
        with open("/proc/self/cgroup") as fh:
            entries = [line.strip() for line in fh if line.strip()]
    except OSError:
        return None, "/proc/self/cgroup unreadable"

    rel = None
    for entry in entries:
        parts = entry.split(":", 2)
        if len(parts) == 3 and (parts[0] == "0" or "memory" in parts[1].split(",")):
            rel = parts[2]
            break
    if rel is None:
        return None, "no memory controller in /proc/self/cgroup"

    segments, cur = [], rel.rstrip("/")
    while True:
        segments.append(cur or "/")
        if not cur:
            break
        cur = cur.rsplit("/", 1)[0]

    best, best_at = None, ""
    for root in ("/sys/fs/cgroup", "/sys/fs/cgroup/memory"):
        if not os.path.isdir(root):
            continue
        for seg in segments:
            base = os.path.join(root, seg.lstrip("/"))
            for fname in ("memory.max", "memory.limit_in_bytes"):
                path = os.path.join(base, fname)
                try:
                    with open(path) as fh:
                        raw = fh.read().strip()
                except OSError:
                    continue
                if raw in ("max", ""):
                    continue
                try:
                    val = int(raw)
                except ValueError:
                    continue
                # cgroup v1 writes a sentinel near 2^63 to mean "unlimited".
                if val <= 0 or val >= 2 ** 62:
                    continue
                if best is None or val < best:
                    best, best_at = val, path
    return best, (best_at or "no memory limit found in this cgroup's ancestry")


def _lsf_slots() -> tuple[int | None, str]:
    """How many slots this job holds, which is the multiplier on every per-slot LSF limit.

    ``LSB_DJOB_NUMPROC`` is the obvious source but is not always exported into the process that ends
    up running the sweep (a tmux session started outside the job, a re-attached shell). Getting this
    wrong is not a small error: a per-slot MEMLIMIT read without its multiplier understates the real
    ceiling by the slot count, which is how a `-M 4096MB` allocation on ~120 slots was mistaken for
    a 4 GiB budget and throttled a whole sweep to two concurrent evals.
    """
    raw = os.environ.get("LSB_DJOB_NUMPROC")
    if raw and raw.strip().isdigit() and int(raw) > 0:
        return int(raw), "LSB_DJOB_NUMPROC"

    # "host1 8 host2 8" -> 16. Present in more contexts than LSB_DJOB_NUMPROC.
    mcpu = os.environ.get("LSB_MCPU_HOSTS")
    if mcpu:
        parts = mcpu.split()
        counts = [int(p) for p in parts[1::2] if p.isdigit()]
        if counts:
            return sum(counts), "LSB_MCPU_HOSTS"

    jobid = os.environ.get("LSB_JOBID")
    if jobid:
        try:
            proc = subprocess.run(["bjobs", "-noheader", "-o", "slots", jobid],
                                  capture_output=True, text=True, timeout=30)
            text = proc.stdout.strip()
            if text.isdigit() and int(text) > 0:
                return int(text), "bjobs slots"
        except Exception:
            pass
    return None, "slot count undeterminable"


def _lsf_memlimit() -> tuple[int | None, str]:
    """The memory ceiling LSF will enforce on this job, in bytes.

    LSF does not always enforce ``-M`` through cgroups — at some sites (Bosch among them) it polls
    and kills instead, so ``memory.max`` reads ``max`` while a real limit is very much in force.
    Sizing Ray from MemTotal in that situation makes Ray admit memory-hungry evals right up to the
    node's RAM and hands LSF a reason to kill the job hours into a sweep. Fails open: an
    undetectable limit returns None rather than guessing.
    """
    per_slot: int | None = None
    src = ""

    # LSB_CG_MEMLIMIT carries the -M value verbatim, in bytes. It reads like a job-wide cgroup
    # number and is NOT one: `-M 4096MB` on 122 slots sets it to 4 GiB, while the ceiling LSF
    # actually enforces is 4 GiB x 122. Scaling it identically to the bjobs value is what keeps the
    # two sources agreeing; reading it raw is how a 488 G allocation was mistaken for 4 G.
    raw = os.environ.get("LSB_CG_MEMLIMIT")
    if raw:
        try:
            per_slot, src = int(raw, 0), "LSB_CG_MEMLIMIT"
        except ValueError:
            pass

    if per_slot is None:
        jobid = os.environ.get("LSB_JOBID")
        if not jobid:
            return None, "not an LSF job"
        try:
            proc = subprocess.run(["bjobs", "-noheader", "-o", "memlimit", jobid],
                                  capture_output=True, text=True, timeout=30)
        except Exception:
            return None, "bjobs unavailable"
        text = proc.stdout.strip()
        match = re.match(r"([\d.]+)\s*([KMGT]?)B?", text, re.IGNORECASE)
        if not match:
            return None, f"bjobs reported memlimit={text!r}"
        scale = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3, "T": 1024 ** 4}
        per_slot, src = int(float(match.group(1)) * scale[match.group(2).upper()]), "bjobs memlimit"

    # MEMLIMIT is PER SLOT and LSF enforces per_slot x nslots against the job's total RSS. Both ends
    # of this are confirmed empirically at this site:
    #   * the injected rusage[mem=1024] default killed a -n 128 job at exactly 1G x 128 = 128 GB;
    #   * `-M 262144MB` on -n 126 asks for 33 TB, which no host can satisfy, so the job never
    #     leaves PEND -- which is only explicable if -M is per-slot as well.
    # So an explicit -M is scaled identically; there is no "job-wide" spelling of this field.
    slots, slot_src = _lsf_slots()
    if slots is None:
        # Using the per-slot figure unscaled is not a conservative fallback, it is a known-wrong
        # answer that under-reports the ceiling by the slot count and silently throttles grading.
        # Better to fall through to node-based sizing than to trust a number we cannot complete.
        return None, f"{src} is per-slot but the {slot_src} — ignoring it"

    # Sanity: a scaled ceiling above the node's own RAM cannot be what LSF enforces, which means the
    # value was job-wide after all (or the request was nonsense). Fall back to the raw figure.
    total = _meminfo_total()
    if total and per_slot * slots > total:
        return per_slot, f"{src}={per_slot / GiB:.0f}G (x{slots} slots exceeds node RAM; job-wide)"
    return (per_slot * slots,
            f"{src}={per_slot / GiB:.0f}G/slot x {slots} slots ({slot_src})")


def _meminfo_total() -> int:
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def _shm_free() -> int:
    """Free bytes in /dev/shm. Ray carves the object store out of it and silently falls back to disk
    (much slower) when it does not fit."""
    try:
        st = os.statvfs("/dev/shm")
        return st.f_bsize * st.f_bavail
    except OSError:
        return 0


def _core_of(cpu: int) -> frozenset[int] | None:
    """The set of logical CPUs sharing a physical core with ``cpu`` — its identity as a core."""
    try:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as fh:
            raw = fh.read().strip()
    except OSError:
        return None
    cpus: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return frozenset(cpus) or None


def machine_cores() -> int:
    """Physical cores on the whole box, regardless of what this process may use.

    The denominator for "what share of this node is ours". It must be in the same unit as
    :func:`granted_cores`, or the memory fair-share comes out wrong by the SMT factor.
    """
    cores, present = set(), []
    try:
        present = [int(d[3:]) for d in os.listdir("/sys/devices/system/cpu")
                   if d.startswith("cpu") and d[3:].isdigit()]
    except OSError:
        pass
    for cpu in present:
        sib = _core_of(cpu)
        if sib is not None:
            cores.add(sib)
    return len(cores) or (os.cpu_count() or 1)


def granted_cores() -> tuple[int, int]:
    """``(physical cores, logical cpus)`` this process may actually run on.

    These differ by the SMT factor, and conflating them is the easy way to size Ray wrong by 2x.
    ``--num-cpus`` has to be counted in PHYSICAL cores: each Ray CPU becomes one logical id pinned by
    ``cpu_scheduler``, so admitting more concurrent logical ids than there are cores puts two evals'
    threads on one core — they then run at roughly half speed while Ray reports full occupancy.

    Falls back to the logical count when the kernel exposes no topology (VMs, containers).
    """
    cpus = sorted(os.sched_getaffinity(0))
    cores = {_core_of(c) for c in cpus}
    if None in cores or not cores:
        return len(cpus), len(cpus)
    return len(cores), len(cpus)


def check_smt_grouping(threads_per_task: int) -> str | None:
    """Warn when an eval's CPU group would land on a single physical core.

    Ray does no pinning at all — ``--num-cpus`` is only an admission counter. The pinning is
    ``cpu_scheduler``, which chops ``sorted(sched_getaffinity)`` into consecutive chunks of
    ``num_cpus_per_task``. Whether chunk ``[0, 1]`` is two physical cores or the two hyperthreads of
    one depends entirely on how the kernel enumerated CPUs, which varies by machine:

      * siblings listed as ``0,128`` (primaries first) -> ``[0, 1]`` is two distinct cores: fine.
      * siblings listed as ``0,1``   (interleaved)     -> ``[0, 1]`` is ONE core: each eval gets
        roughly half the throughput it looks like it has, which silently shifts runtimes and timeout
        rates relative to a machine where the enumeration happened to be the other way.

    Read-only: this reports the situation, it does not change the grouping.
    """
    if threads_per_task < 2:
        return None
    cpus = sorted(os.sched_getaffinity(0))
    cores = {c: _core_of(c) for c in cpus}
    if any(v is None for v in cores.values()):
        return None                                    # no topology exposed (VM/container): stay quiet
    if all(len(v) == 1 for v in cores.values()):
        return None                                    # SMT off: every logical CPU is a core

    groups = [cpus[i:i + threads_per_task] for i in range(0, len(cpus), threads_per_task)]
    groups = [g for g in groups if len(g) == threads_per_task]
    collided = [g for g in groups if len({cores[c] for c in g}) < len(g)]
    if not collided:
        return None
    return (f"SMT is on and {len(collided)}/{len(groups)} of the CPU groups cpu_scheduler will hand "
            f"out sit on a SINGLE physical core (e.g. {collided[0]} share core "
            f"{sorted(cores[collided[0][0]])}). Each eval then gets ~one core of throughput, not "
            f"{threads_per_task} — runtimes and timeout rates will not be comparable to a machine "
            "that enumerates primaries first. Ray itself pins nothing; this is cpu_scheduler "
            "chopping consecutive logical CPU ids.")


def _local_ips() -> set[str]:
    ips = {"127.0.0.1", "localhost"}
    try:
        ips.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True, timeout=10)
        ips.update(out.stdout.split())
    except Exception:
        pass
    return {ip for ip in ips if ip}


# --------------------------------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------------------------------
@dataclass
class HeadPlan:
    """Everything ``ray start`` will be told, plus how each number was arrived at."""
    num_cpus: int
    threads_per_task: int
    memory_bytes: int
    object_store_bytes: int
    temp_dir: str
    port: str                        # "auto" -> --port=0, else a literal port
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def env(self) -> dict[str, str]:
        return thread_env(self.threads_per_task)

    def argv(self) -> list[str]:
        return [
            ray_bin(), "start", "--head",
            f"--num-cpus={self.num_cpus}",
            f"--memory={self.memory_bytes}",
            f"--object-store-memory={self.object_store_bytes}",
            f"--temp-dir={self.temp_dir}",
            f"--port={0 if self.port == 'auto' else self.port}",
            "--include-dashboard=false",       # the dashboard costs a core and a port for nothing here
            "--disable-usage-stats",
        ]

    def describe(self) -> str:
        return (f"cpus={self.num_cpus} (threads/task={self.threads_per_task})  "
                f"memory={self.memory_bytes / GiB:.0f}G  "
                f"object_store={self.object_store_bytes / GiB:.1f}G  temp_dir={self.temp_dir}")


def plan_head(threads_per_task: int, max_parallel: int, cfg: dict | None = None) -> HeadPlan:
    """Size the head from the cgroup, not from ``nproc`` and not from a hardcoded constant.

    ``cfg`` is the sweep file's optional ``sweep.ray`` block; every key is an override for something
    detected here.
    """
    cfg = dict(cfg or {})
    notes: list[str] = []
    warnings: list[str] = []
    threads_per_task = max(1, int(threads_per_task))

    # --- CPUs ------------------------------------------------------------------------------------
    cores, affinity = granted_cores()
    quota, quota_where, throttled = _cgroup_cpu_limits()
    lsf = int(os.environ.get("LSB_DJOB_NUMPROC") or 0)

    # Count in PHYSICAL cores, never in logical CPUs: on an SMT node those differ by 2x, and the
    # batch system reports slots in cores while sched_getaffinity reports threads. Taking the
    # logical number here would double-book every core.
    budget, source = cores, f"{cores} physical cores in this process's affinity"
    if quota is not None and int(quota) < budget:
        budget, source = int(quota), f"cgroup cpu quota={quota:.1f} ({quota_where})"
    if lsf and lsf < budget:
        budget, source = lsf, f"LSB_DJOB_NUMPROC={lsf}"
    smt = affinity / cores if cores else 1.0
    notes.append(f"cpu budget {budget} from {source} "
                 f"(cores={cores}, logical={affinity}{f', SMT x{smt:.0f}' if smt > 1 else ''}, "
                 f"quota={'-' if quota is None else f'{quota:.1f}'}, "
                 f"LSB_DJOB_NUMPROC={lsf or '-'})")

    if lsf and lsf > cores:
        warnings.append(
            f"LSB_DJOB_NUMPROC={lsf} exceeds the {cores} physical cores this process can run on. "
            f"If those slots are logical CPUs, sizing Ray from {lsf} would pin two evals' threads "
            "to the same core and halve their speed while Ray reported full occupancy. Using the "
            "core count instead.")
    smt = check_smt_grouping(threads_per_task)
    if smt:
        warnings.append(smt)
    if throttled:
        path, nr, usec = throttled
        warnings.append(
            f"this cgroup has ALREADY been CPU-throttled (nr_throttled={nr}, {usec / 1e6:.1f}s, "
            f"{path}) — the kernel is capping CPU time regardless of core count, so evals will run "
            "slow no matter how the head is sized. Get a real allocation, not a login node.")

    # An explicit count is the operator answering "how much of this box may Ray use", which is not
    # always "all of it": sharing a node with another job, leaving headroom for a server on the same
    # host, or reproducing a run made on a smaller machine. It is the FINAL --num-cpus, not another
    # budget to subtract a reserve from -- "let Ray see 16" has to mean 16 or it does not answer the
    # question. Detection still runs, because its numbers are what the warnings below compare against.
    want = cfg.get("num_cpus")
    reserve = cfg.get("reserve_cpus")
    reserve = (1 + max_parallel) if reserve is None else int(reserve)
    if want is not None:
        want = int(want)
        # Whole eval slots only, same as the detected path: cpu_scheduler drops the trailing partial
        # group, so a num_cpus that is not a multiple of threads_per_task promises Ray capacity the
        # scheduler cannot back.
        num_cpus = (want // threads_per_task) * threads_per_task
        if num_cpus < threads_per_task:
            raise ValueError(
                f"sweep.ray.num_cpus={want} is less than one {threads_per_task}-cpu eval slot "
                f"(num-cpus-per-task={threads_per_task}). Ask for at least {threads_per_task}.")
        notes.append(
            f"cpu count OVERRIDDEN to {num_cpus} by sweep.ray.num_cpus={want}"
            + (f" (rounded down to a multiple of {threads_per_task})" if num_cpus != want else "")
            + f"; the detected budget ({budget}) and reserve_cpus are NOT applied"
            + (f", leaving {budget - num_cpus} granted core(s) deliberately idle"
               if num_cpus < budget else ""))
        # cpu_scheduler builds its groups from the affinity mask, NOT from --num-cpus, so these two
        # ceilings fail differently and both are worth naming.
        if num_cpus > affinity:
            warnings.append(
                f"sweep.ray.num_cpus={num_cpus} exceeds the {affinity} logical CPU(s) this process "
                f"may run on, and cpu_scheduler chops THAT mask into groups — it can only ever hand "
                f"out {affinity // threads_per_task}. Ray will admit "
                f"{num_cpus // threads_per_task} concurrent evals, and the surplus will spin in "
                "get_cpu_group() until they fail as cpu_starvation. Lower it.")
        elif num_cpus > cores:
            warnings.append(
                f"sweep.ray.num_cpus={num_cpus} exceeds the {cores} physical core(s) granted "
                f"(SMT gives {affinity} logical CPUs). Evals will share cores and run at roughly "
                "half speed while Ray reports full occupancy — runtimes and timeout rates will not "
                "be comparable to a run sized to the cores.")
    else:
        # Leave the supervisor and each concurrent run_icl driver a core: the drivers do the HTTP and
        # tokenization work that actually paces a generation, and starving them slows the whole sweep
        # even while every eval core stays busy.
        usable = budget - reserve
        num_cpus = (usable // threads_per_task) * threads_per_task
        notes.append(f"reserved {reserve} cpu(s) for the supervisor + {max_parallel} driver(s); "
                     f"rounded down to a multiple of {threads_per_task}")
        if num_cpus < threads_per_task:
            raise ValueError(
                f"cpu budget {budget} minus {reserve} reserved leaves no room for even one "
                f"{threads_per_task}-cpu eval. Lower sweep.ray.reserve_cpus, set "
                "sweep.ray.num_cpus explicitly, or get more cores.")

    # --- memory ----------------------------------------------------------------------------------
    cg_mem, mem_where = _cgroup_memory_limit()
    lsf_mem, lsf_where = _lsf_memlimit()
    total = _meminfo_total()

    # A tiny LSF memlimit is the site's unenforced default, NOT a real budget. Bosch hands out 1G by
    # default and does not kill on it (jobs there routinely peak two orders of magnitude above it),
    # so clamping Ray to it would leave less heap than a single eval reserves and admit zero tasks —
    # the sweep would hang instead of run. Warn and size from the node; a real `-M` is above the
    # floor and does get honoured.
    if lsf_mem is not None and lsf_mem < LSF_MEMLIMIT_FLOOR:
        warnings.append(
            f"LSF reports MEMLIMIT={lsf_mem / GiB:.1f}G ({lsf_where}) — the site default, which is "
            "below what one eval reserves. Treating it as advisory and sizing from the node "
            "instead. If your site ever starts ENFORCING it, this job dies immediately: submit "
            'with `-M 262144MB -R "rusage[mem=262144]"` to make the budget real.')
        lsf_mem = None

    if cg_mem or lsf_mem:
        limit = min([v for v in (cg_mem, lsf_mem, total) if v])
        which = mem_where if cg_mem and limit == cg_mem else lsf_where
        notes.append(f"memory limit {limit / GiB:.1f}G from {which}")

        if lsf_mem and "/slot" in lsf_where and lsf and lsf_mem <= LSF_DEFAULT_PER_SLOT * lsf:
            warnings.append(
                f"this {lsf_mem / GiB:.0f}G ceiling is the site's injected rusage[mem=1024] scaled "
                f"by -n, not a choice. Exceed it and LSF kills the job with TERM_MEMLIMIT (SIGINT, "
                "SIGTERM, SIGKILL, 10s apart) — an interactive shell dies with it. Raise it "
                'PER SLOT, e.g. `-M 4096MB -R "rusage[mem=4096]"`; a job-total figure like '
                "-M 262144MB is multiplied by the slot count and the job never leaves PEND.")
        # Ray's own OOM guard measures NODE memory, so on a big shared box it never fires before
        # LSF's per-job ceiling does. Worth saying out loud: Ray will not save this job.
        if total and limit < 0.5 * total:
            warnings.append(
                f"Ray's memory monitor watches NODE memory ({total / GiB:.0f}G), not your "
                f"{limit / GiB:.0f}G LSF ceiling, so it will not fire before LSF kills the job. "
                "Task admission below is the only in-process guard; size the request with headroom.")
    else:
        # No cgroup memory limit means the batch system handed out CPU slots but is NOT confining
        # our memory — so nothing stops us from taking the whole node's RAM. Sizing Ray from
        # MemTotal would do exactly that, and because each eval task requests memory (TASK_MEMORY),
        # Ray's admission control would happily run hundreds of them. The kernel OOM killer is
        # machine-wide here, so overshooting kills a co-tenant's job, not just ours. Claim only the
        # share of RAM that matches the share of CPUs we were allocated.
        # Both sides of this ratio must be physical cores. Dividing our core budget by the LOGICAL
        # cpu count would understate our share by the SMT factor and starve the sweep of memory.
        # When the operator pinned num_cpus, the fair share is of what they ASKED FOR, not of what
        # the box granted: taking 16 of 128 cores and still claiming the whole node's RAM is exactly
        # the co-tenant-hostile case this branch exists to avoid.
        claim = num_cpus if want is not None else budget
        node = machine_cores()
        share = min(1.0, claim / node) if node else 1.0
        limit = int(total * share)
        notes.append(f"no cgroup memory limit ({mem_where}); claiming {share:.0%} of "
                     f"MemTotal={total / GiB:.0f}G = {limit / GiB:.0f}G, the fair share for "
                     f"{claim}/{node} physical cores"
                     + (" (from sweep.ray.num_cpus)" if want is not None else ""))

    # Object store: this workload ships program text and scores — kilobytes. It exists mostly for
    # Ray's own bookkeeping, so a couple of GB is plenty, and every byte of it is taken out of
    # /dev/shm and counted against the job's memory limit.
    shm = _shm_free()
    want_obj = int(float(cfg.get("object_store_gb", 2)) * GiB)
    obj = want_obj
    if shm and obj > 0.5 * shm:
        obj = int(0.5 * shm)
        warnings.append(f"object store trimmed to {obj / GiB:.1f}G: /dev/shm has only "
                        f"{shm / GiB:.1f}G free and Ray falls back to disk when it does not fit")
    if limit:
        obj = min(obj, int(0.2 * limit))

    frac = float(cfg.get("memory_fraction", 0.85))
    memory = int(limit * frac) - obj if limit else 8 * GiB
    if memory <= 0:
        raise ValueError(f"memory limit {limit / GiB:.1f}G is too small for a {obj / GiB:.1f}G "
                         "object store; lower sweep.ray.object_store_gb")
    concurrent = max(1, num_cpus // threads_per_task)

    # An explicit override wins over every heuristic above. Provided because the detected limit can
    # be a cgroup that has nothing to do with the batch allocation, and a wrong one here does not
    # fail loudly — it silently throttles the whole sweep to a couple of evals.
    if cfg.get("memory_gb"):
        memory = int(float(cfg["memory_gb"]) * GiB)
        notes.append(f"heap memory overridden to {memory / GiB:.0f}G by sweep.ray.memory_gb")

    # Ray admits eval tasks against their declared TASK_MEMORY (1 GiB each), so the heap has to
    # cover the CPU-bound concurrency or memory quietly becomes the binding constraint: `ray status`
    # then shows a handful of busy cores, a full memory resource, and hundreds of pending tasks.
    # Where the ceiling allows it, buy the headroom back rather than leaving cores idle.
    needed = concurrent * TASK_MEMORY
    if memory < needed and not cfg.get("memory_gb"):
        headroom = int(0.95 * limit) - obj if limit else needed
        if headroom >= needed:
            notes.append(f"heap raised {memory / GiB:.0f}G -> {needed / GiB:.0f}G so all "
                         f"{concurrent} eval slots are admissible (was below 1G/eval)")
            memory = needed
        else:
            admits = max(1, int(headroom / TASK_MEMORY))
            memory = max(memory, min(headroom, needed))
            warnings.append(
                f"MEMORY, NOT CPU, WILL LIMIT THIS SWEEP: the {limit / GiB:.1f}G ceiling admits "
                f"~{admits} concurrent evals but the cores allow {concurrent}, so most of the box "
                f"sits idle with tasks pending. Either raise the allocation (LSF memory is PER "
                f"SLOT: `-M 4096MB -R \"rusage[mem=4096]\"`), or if that {limit / GiB:.1f}G figure "
                "looks wrong for this machine, override it with sweep.ray.memory_gb.")

    per_eval = memory / concurrent
    notes.append(f"heap {memory / GiB:.0f}G for {concurrent} concurrent evals => "
                 f"{per_eval / GiB:.1f}G each" +
                 (f", {(limit - memory) / GiB:.0f}G left for the {max_parallel} driver(s)"
                  if limit else ""))

    # --- temp dir --------------------------------------------------------------------------------
    # Per-host and per-job. Even if /tmp turns out to be shared between machines, two heads can then
    # never collide on the address file, the session dir, or the sockets.
    base = cfg.get("temp_dir_base") or os.environ.get("RAY_TEMP_BASE") or "/tmp"
    host = socket.gethostname().split(".")[0]
    tag = os.environ.get("LSB_JOBID") or str(os.getpid())
    temp_dir = cfg.get("temp_dir") or os.path.join(base, f"ray-{os.getuid()}-{host}-{tag}")
    if len(temp_dir) > MAX_TEMP_DIR_LEN:
        warnings.append(f"temp dir is {len(temp_dir)} chars ({temp_dir}); Ray appends a session dir "
                        f"and a socket name to it and Unix sockets cap at 107 bytes. Set "
                        f"sweep.ray.temp_dir_base to something shorter if the head fails to start.")

    return HeadPlan(num_cpus=num_cpus, threads_per_task=threads_per_task, memory_bytes=memory,
                    object_store_bytes=obj, temp_dir=temp_dir,
                    port=str(cfg.get("port", "auto")), notes=notes, warnings=warnings)


# --------------------------------------------------------------------------------------------------
# diagnostics, start, stop
# --------------------------------------------------------------------------------------------------
def diagnose_default_address_file() -> str | None:
    """Report a ``/tmp/ray/ray_current_cluster`` that names some *other* machine.

    That file is the shared-/tmp symptom: it outranks ps-grep discovery, so anything on this box that
    calls ``ray.init(address="auto")`` without RAY_ADDRESS would connect to the machine it names.
    """
    path = "/tmp/ray/ray_current_cluster"
    try:
        with open(path) as fh:
            addr = fh.read().strip()
    except OSError:
        return None
    host = addr.rsplit(":", 1)[0]
    if host and host not in _local_ips():
        return (f"{path} points at {addr}, which is not this machine. /tmp really is shared across "
                "machines, and any ray.init(address='auto') here without RAY_ADDRESS would connect "
                "to that host. The sweep sets RAY_ADDRESS explicitly, so its runs are unaffected.")
    return None


def head_is_running() -> tuple[str | None, str | None]:
    """Find a head on THIS machine that the sweep may attach to.

    Returns ``(address, refusal_reason)``. Locality is the whole point: on a shared /tmp,
    ``ray status`` happily succeeds against a head on *another* machine, because the address file it
    reads was written there. Attaching would ship every eval task across the network to a box busy
    with its own sweep, so a non-local head is reported as a refusal rather than as a head, and the
    caller starts a properly sized local one instead.
    """
    local = _local_ips()
    candidates = []
    env_addr = os.environ.get("RAY_ADDRESS")
    if env_addr:
        candidates.append((env_addr, "RAY_ADDRESS"))
    try:
        with open("/tmp/ray/ray_current_cluster") as fh:
            addr = fh.read().strip()
        if addr:
            candidates.append((addr, "/tmp/ray/ray_current_cluster"))
    except OSError:
        pass

    for addr, where in candidates:
        if addr.rsplit(":", 1)[0] not in local:
            return None, (f"{where} names {addr}, which is NOT this machine — refusing to attach to "
                          "it. Starting a local head instead.")
        if _gcs_alive(addr):
            return addr, None

    # No usable address file: ask Ray, which falls back to grepping ps for a local GCS process.
    try:
        if subprocess.run([ray_bin(), "status"], capture_output=True, timeout=30).returncode == 0:
            return "auto", None
    except Exception:
        pass
    return None, None


def _gcs_alive(addr: str) -> bool:
    host, _, port = addr.rpartition(":")
    if not port.isdigit():
        return False
    try:
        with socket.create_connection((host, int(port)), timeout=5):
            return True
    except OSError:
        return False


def start_head(plan: HeadPlan, timeout: float = 300.0) -> str:
    """Start the head and return its address. Raises RuntimeError with ray's stderr on failure."""
    os.makedirs(plan.temp_dir, exist_ok=True)
    env = {**os.environ, **plan.env(), "RAY_TMPDIR": plan.temp_dir}
    proc = subprocess.run(plan.argv(), capture_output=True, text=True, timeout=timeout, env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"`{' '.join(plan.argv())}` failed:\n{proc.stderr.strip()[-1500:]}")

    # Read the address ray just wrote into OUR temp dir rather than scraping stdout: it is the same
    # file ray.init() would consult, so if it is wrong we want to fail here and not at grading time.
    path = os.path.join(plan.temp_dir, "ray_current_cluster")
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with open(path) as fh:
                addr = fh.read().strip()
            if addr:
                return addr
        except OSError:
            pass
        time.sleep(0.5)

    # Fall back to stdout, which prints RAY_ADDRESS='<ip:port>' on success.
    match = re.search(r"RAY_ADDRESS='([^']+)'", proc.stdout)
    if match:
        return match.group(1)
    raise RuntimeError(f"head started but wrote no address to {path}; stdout:\n"
                       f"{proc.stdout.strip()[-1500:]}")


# Path fragments that only ever appear in a real Ray component's command line. Matching the temp dir
# alone is NOT enough: the shell that launched the sweep, an editor, or a grep can all mention it,
# and killing those was an actual bug caught in testing.
_RAY_PROC_MARKERS = (
    "/ray/core/src/ray/",                  # raylet, gcs_server (compiled binaries)
    "/ray/_private/",                      # eval workers, log_monitor, runtime_env agent
    "/ray/dashboard/",
    "/ray/autoscaler/",
)


def _ancestors(pid: int) -> set[int]:
    """This process and every parent, so teardown can never kill the thing calling it."""
    out, cur = set(), pid
    while cur > 1 and cur not in out:
        out.add(cur)
        try:
            with open(f"/proc/{cur}/stat") as fh:
                cur = int(fh.read().rsplit(")", 1)[1].split()[1])
        except (OSError, IndexError, ValueError):
            break
    return out


# Ray's component logs, in rough order of post-mortem value. raylet.* is the one that records worker
# deaths, OOM-killer decisions and object-store spilling; the rest give the cluster's own view of what
# it thought it had. Per-worker logs are handled separately because there can be tens of thousands.
_LOG_KEEP_PREFIXES = ("raylet", "gcs_server", "monitor", "log_monitor", "dashboard", "runtime_env",
                      "ray_client", "plasma_store", "debug_state")

# Worker logs are one pair per worker, and with MAX_CALLS=1 that is one pair PER EVAL: a multi-day
# sweep produces tens of thousands. Keep the newest few hundred (the ones near whatever went wrong)
# rather than all of them or none.
_MAX_WORKER_LOGS = 400


def archive_logs(temp_dir: str, dest_dir: str, compress: bool = True,
                 tail_bytes: int | None = None) -> tuple[str | None, str]:
    """Copy this head's session logs out of ``temp_dir`` into ``dest_dir`` as a .tar.gz.

    Ray writes everything under ``<temp_dir>/session_*/logs``, which lives in ``/tmp`` on the compute
    node. That is the worst possible place for it: ``/tmp`` is node-local, the node is reclaimed when
    the job ends, and :func:`stop_head` (or a job script's ``rm -rf $RAY_TMPDIR``) removes it on the
    way out. So the one artifact that could explain a kill is always gone by the time anyone looks.

    ``compress=False`` writes a plain .tar. Use it on the signal path: LSF's TERM_MEMLIMIT sequence
    is SIGINT, SIGTERM, SIGKILL ten seconds apart, so there is one ~10 s window to get this to disk
    and gzipping a few hundred MB of worker logs will not finish inside it.

    ``tail_bytes`` keeps only the last N bytes of any file bigger than that. Also for the signal path,
    and it is the difference between a dump that completes and one that does not: on a real session
    here ``raylet.out`` was 296 MB and ``gcs_server.out`` 263 MB, so an untruncated emergency archive
    was 762 MB — too slow to finish in the window, especially to a network filesystem. The tail is
    also the part worth having: whatever the cluster said as it died is at the end, not the start.

    Returns ``(archive_path_or_None, human_readable_note)``. Never raises: losing the logs is bad,
    but failing a sweep's teardown over it is worse.
    """
    import glob
    import tarfile

    try:
        sessions = sorted(glob.glob(os.path.join(temp_dir, "session_*")))
        # session_latest is a symlink to one of the above; ignore it so nothing is archived twice.
        sessions = [s for s in sessions if not os.path.islink(s)]
        if not sessions:
            return None, f"no session dir under {temp_dir} — nothing to archive"
        logs_dir = os.path.join(sessions[-1], "logs")
        if not os.path.isdir(logs_dir):
            return None, f"{logs_dir} does not exist — nothing to archive"

        component, workers = [], []
        for name in os.listdir(logs_dir):
            path = os.path.join(logs_dir, name)
            if not os.path.isfile(path):
                continue
            (component if name.startswith(_LOG_KEEP_PREFIXES) else workers).append(path)

        # Newest last-modified first: after a kill, the interesting workers are the ones that were
        # still running when it landed.
        workers.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        dropped = max(0, len(workers) - _MAX_WORKER_LOGS)
        chosen = component + workers[:_MAX_WORKER_LOGS]

        os.makedirs(dest_dir, exist_ok=True)
        suffix = "tar.gz" if compress else "tar"
        archive = os.path.join(dest_dir, f"ray_logs_{os.path.basename(sessions[-1])}.{suffix}")
        truncated = 0
        with tarfile.open(archive, "w:gz" if compress else "w") as tar:
            for path in chosen:
                try:
                    full = os.path.getsize(path)
                    if tail_bytes is None or full <= tail_bytes:
                        tar.add(path, arcname=os.path.basename(path))
                        continue
                    # Seek to the tail and declare the shorter size, so the member is the last
                    # tail_bytes of the file. Renamed so nobody mistakes it for the whole log.
                    with open(path, "rb") as fh:
                        fh.seek(full - tail_bytes)
                        info = tar.gettarinfo(path, arcname=os.path.basename(path) + ".tail")
                        info.size = tail_bytes
                        tar.addfile(info, fh)
                    truncated += 1
                except OSError:
                    continue

        size = os.path.getsize(archive)
        note = (f"archived {len(chosen)} Ray log file(s) ({size / (1024 ** 2):.1f} MB) to {archive}")
        if truncated:
            note += (f"; {truncated} oversized log(s) kept as the last "
                     f"{tail_bytes / (1024 ** 2):.0f} MB only (*.tail)")
        if dropped:
            note += (f"; dropped {dropped} older per-worker log(s) over the {_MAX_WORKER_LOGS} cap "
                     f"(raise ray_head._MAX_WORKER_LOGS if you need the full history)")
        return archive, note
    except Exception as e:
        return None, f"could not archive Ray logs from {temp_dir}: {e!r}"


def remove_temp_dir(temp_dir: str) -> tuple[int, str]:
    """Delete a temp dir this module created, reporting how much it freed.

    Only ever call this on a dir :func:`plan_head` named (``<base>/ray-<uid>-<host>-<tag>``) and only
    after :func:`archive_logs`. Refuses anything else -- notably the shared default ``/tmp/ray``,
    which belongs to whoever ran ``ray start`` by hand and may be in use by a co-tenant.

    Returns ``(bytes_freed, note)``. Never raises.
    """
    import shutil

    base = os.path.basename(temp_dir.rstrip("/"))
    if not base.startswith(f"ray-{os.getuid()}-"):
        return 0, (f"not removing {temp_dir}: only dirs this module named (ray-<uid>-<host>-<tag>) "
                   "are ours to delete")
    try:
        freed = 0
        for root, _dirs, files in os.walk(temp_dir):
            for name in files:
                try:
                    freed += os.path.getsize(os.path.join(root, name))
                except OSError:
                    continue
        shutil.rmtree(temp_dir, ignore_errors=True)
        return freed, f"removed {temp_dir} ({freed / GiB:.2f} G of session logs)"
    except Exception as e:
        return 0, f"could not remove {temp_dir}: {e!r}"


def stop_head(temp_dir: str, grace: float = 10.0) -> int:
    """Kill only the Ray processes belonging to ``temp_dir``.

    ``ray stop`` matches every Ray process on the machine by command line, so on a shared box it
    would take down a co-tenant's cluster — or your own head from another job. Every process in a
    session carries its temp dir on the command line, which makes an exact filter available; pair
    that with a marker for "is actually a Ray component" and the blast radius is exactly one session.
    """
    import signal

    protected = _ancestors(os.getpid())
    victims = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) in protected:
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if temp_dir in cmd and any(marker in cmd for marker in _RAY_PROC_MARKERS):
            victims.append(int(pid))

    for pid in victims:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if not any(os.path.exists(f"/proc/{pid}") for pid in victims):
            break
        time.sleep(0.5)
    for pid in victims:
        try:
            os.kill(pid, signal.SIGKILL)
        except OSError:
            pass
    return len(victims)
