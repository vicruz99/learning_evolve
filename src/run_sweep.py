#!/usr/bin/env python
"""Coordinated launcher for several ICL runs: shared settings + per-run overrides, staggered starts,
a bounded queue, and one place to see what is running.

Why this exists
---------------
Launching ``run_icl.py`` several times by hand has bitten us three ways: simultaneous starts wedge
Ray (each run boots its own head), a sweep's shared settings drift apart between shells, and there is
no single view of which runs are alive. This script owns all three.

Usage
-----
    python run_sweep.py sweeps/ctx_strategies.yaml                 # launch (stays in the foreground)
    python run_sweep.py sweeps/ctx_strategies.yaml --print-cmds    # expand + print, launch nothing
    python run_sweep.py --status runs/ctx_strategies               # table of a sweep's state
    python run_sweep.py --resume runs/ctx_strategies               # restart whatever is not complete,
                                                                   #   each from generation 0 (verified
                                                                   #   against the run dir; add
                                                                   #   --print-cmds to just look)
    python run_sweep.py --continue-run runs/ctx_strategies/cp26_cs-best        # continue ONE run where
    python run_sweep.py --continue-run runs/.../cp26_cs-best --from-generation 7   # it stopped, or at 7
    python run_sweep.py --resume runs/ctx_strategies --continue-run cp26_cs-best:7   # ...and keep the
                                                                   #   rest of the sweep going too
    python run_sweep.py --stop   runs/ctx_strategies               # halt: drop the queue, SIGTERM
                                                                   #   every live run (a supervisor
                                                                   #   watching it stops launching)

Run it under ``tmux`` for anything long: the supervisor must stay alive to enforce ``max_parallel``,
but the runs are started in their own process session, so killing the supervisor does NOT kill them
(and ``--status`` keeps working afterwards).

Sweep file
----------
Keys are ``run_icl.py`` long flags with the leading ``--`` removed, so there is no second vocabulary
to learn and nothing to keep in sync; unknown keys are rejected before anything launches.

    sweep:
      name: ctx_strategies          # -> runs/ctx_strategies/<run>/ (also the sweep's index.csv dir)
      max_parallel: 2
      stagger: 120                  # seconds between launches
      server_max_num_seqs: 256      # optional; only used to warn about oversubscription
      ray:                          # optional; every key overrides something auto-detected
        num_cpus: 16                # cores Ray may use, chosen by hand (--ray-num-cpus overrides).
                                    #   This IS the final --num-cpus: reserve_cpus is not applied
                                    #   on top, and the rest of the allocation is left idle.
        reserve_cpus: 3             # cores kept off Ray for the supervisor + drivers
                                    #   (default 1 + max_parallel; ignored when num_cpus is set)
        object_store_gb: 2          # carved out of /dev/shm; this workload ships kilobytes
        memory_fraction: 0.85       # of the cgroup's memory limit, minus the object store
        temp_dir_base: /tmp         # set to node-local scratch if /tmp is a network mount
        port: auto                  # auto = let the kernel pick, so co-tenants cannot collide

Ray
---
The launcher starts and sizes the head itself (``--ray-head=auto``, the default) and stops it when
the queue drains. You do NOT need to ``ray start`` or export ``OMP_NUM_THREADS`` by hand — the
thread-limit variables are derived from ``num-cpus-per-task`` and set on the head process, which is
the only place that affects eval workers. Runs receive ``RAY_ADDRESS`` explicitly, so a shared
``/tmp`` cannot make them attach to another machine's cluster. See ``sandbox/ray_head.py``.

    common:
      problem: circle_packing_26
      groups-per-batch: 6
      group-size: 15
      num-generations: 30
      reasoning-effort: medium
      vllm-base-url: http://localhost:8001/v1

    grid:                           # cross-product; run names derive from the varying keys
      context-strategy: [best, random]
      n-context: [10, 20]

    runs:                           # explicit entries, each overriding common
      - name: cp26_best_n30
        context-strategy: best
        n-context: 30

Precedence is ``common`` < ``grid``/``runs`` entry. ``grid`` and ``runs`` may be used together; every
resulting run gets its own ``--log-path`` under the sweep directory unless it sets one explicitly.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any

import yaml

from results.resume import inspect_run, rewind, tail_exists
from run_icl import build_parser
from sandbox import ray_head

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = "sweep.json"
# Written by --stop, read by a live supervisor once per pass. A separate file rather than a field in
# sweep.json on purpose: the supervisor rewrites that manifest on every launch and every reap, so a
# flag inside it races with its own writer and gets clobbered within the second.
HALT = ".halted"

# Flags the sweep file must not set: the launcher owns them.
RESERVED = {"log-path", "resume-step"}

# Every key ray_head.plan_head() honours in the `sweep.ray` block. Kept here so a typo is rejected
# before anything launches rather than silently ignored by cfg.get().
RAY_SETTINGS = {"num_cpus", "reserve_cpus", "memory_gb", "memory_fraction", "object_store_gb",
                "temp_dir", "temp_dir_base", "port"}


# --------------------------------------------------------------------------------------------------
# sweep file -> run specs
# --------------------------------------------------------------------------------------------------
class SweepError(Exception):
    """A problem with the sweep file. Always raised before anything is launched."""


def _flag_tables() -> tuple[dict[str, Any], dict[str, dict[bool, str]]]:
    """Introspect ``run_icl.py``'s parser into:

      * ``by_flag``  — flag name (no ``--``) -> argparse action, for validation and value parsing
      * ``bool_pair`` — dest -> {True: affirmative flag, False: negative flag} for store_true/false
                        pairs, so ``include-code: false`` becomes ``--no-include-code``
    """
    by_flag: dict[str, Any] = {}
    bool_pair: dict[str, dict[bool, str]] = {}
    for act in build_parser()._actions:
        for opt in act.option_strings:
            if opt.startswith("--"):
                by_flag[opt[2:]] = act
        if act.nargs == 0 and act.option_strings:           # store_true / store_false
            polarity = bool(getattr(act, "const", True))
            pair = bool_pair.setdefault(act.dest, {})
            pair[polarity] = act.option_strings[0]
    return by_flag, bool_pair


def _abbrev(flag: str) -> str:
    """`context-strategy` -> `cs`; used to auto-name grid runs compactly and predictably."""
    return "".join(word[0] for word in flag.split("-") if word)


def _expand(spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand ``common`` + ``grid`` + ``runs`` into a flat list of {name, flags} run specs."""
    common = dict(spec.get("common") or {})
    grid = dict(spec.get("grid") or {})
    explicit = list(spec.get("runs") or [])
    if not grid and not explicit:
        raise SweepError("sweep file defines neither `grid` nor `runs` — nothing to launch")

    out: list[dict[str, Any]] = []
    if grid:
        for key, values in grid.items():
            if not isinstance(values, list) or not values:
                raise SweepError(f"grid key {key!r} must be a non-empty list, got {values!r}")
        keys = list(grid)
        # Only the keys that actually vary go into the auto name, so a 1-value axis stays silent.
        varying = [k for k in keys if len(grid[k]) > 1] or keys
        for combo in itertools.product(*(grid[k] for k in keys)):
            flags = {**common, **dict(zip(keys, combo))}
            # `problem` is already the name's prefix, so keep it out of the suffix: a grid over
            # problems would otherwise produce `erdos_p-erdos_s-1`.
            suffix = "_".join(f"{_abbrev(k)}-{flags[k]}" for k in varying if k != "problem")
            name = f"{flags.get('problem', 'run')}" + (f"_{suffix}" if suffix else "")
            out.append({"name": name, "flags": flags})

    for i, entry in enumerate(explicit):
        entry = dict(entry)
        name = entry.pop("name", None) or f"run{i:02d}"
        out.append({"name": name, "flags": {**common, **entry}})

    names = [r["name"] for r in out]
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise SweepError(f"duplicate run names: {sorted(dupes)} — give them explicit `name:` keys")
    return out


def _to_argv(flags: dict[str, Any], by_flag: dict, bool_pair: dict) -> list[str]:
    """Turn a run's flag dict into a ``run_icl.py`` argv, validating every key and bool polarity."""
    argv: list[str] = []
    for key, value in flags.items():
        if key in RESERVED:
            raise SweepError(f"`{key}` is owned by the launcher — remove it from the sweep file")
        act = by_flag.get(key)
        if act is None:
            close = sorted(k for k in by_flag if _abbrev(k) == _abbrev(key) or key in k)
            hint = f" (did you mean: {', '.join(close)}?)" if close else ""
            raise SweepError(f"unknown option `{key}` — not a run_icl.py flag{hint}")
        if act.nargs == 0:                                   # boolean flag
            if not isinstance(value, bool):
                raise SweepError(f"`{key}` is a boolean flag; give it true or false, not {value!r}")
            flag = bool_pair.get(act.dest, {}).get(value)
            if flag is None:
                raise SweepError(
                    f"`{key}: {str(value).lower()}` cannot be expressed: run_icl.py has no "
                    f"{'affirmative' if value else 'negative'} flag for it")
            argv.append(flag)
        else:
            argv += [f"--{key}", str(value)]
    return argv


# --------------------------------------------------------------------------------------------------
# preflight checks
# --------------------------------------------------------------------------------------------------
def threads_per_task(entries: list[dict]) -> int:
    """The sweep's ``num-cpus-per-task``, which must be one value for the whole sweep.

    ``cpu_scheduler`` is a *detached* actor created with the first run's ``num_cpus_per_task`` and
    ``get_if_exists=True`` thereafter, so the second run's value is silently ignored and its evals
    get the first run's group size. Rejecting the mix here beats debugging it later.
    """
    values = set()
    for entry in entries:
        flags = _flags_from_cmd(entry["cmd"])
        values.add(int(flags.get("num-cpus-per-task", 1)))
    if len(values) > 1:
        raise SweepError(
            f"runs disagree on num-cpus-per-task ({sorted(values)}). One shared Ray head means one "
            "detached cpu_scheduler actor, and it is created with whichever value starts first — "
            "the others are silently ignored. Split this into one sweep file per value.")
    return values.pop() if values else 1


def _ray_cfg(settings: dict, num_cpus: int | None) -> dict:
    """The sweep file's ``ray`` block with ``--ray-num-cpus`` layered on top.

    One helper rather than a dict literal at each call site, because the head that ``--print-cmds``
    describes and the head that actually starts have to be the same head; that is exactly what drifts
    when two places each build the config themselves.
    """
    cfg = dict(settings.get("ray") or {})
    if num_cpus is not None:
        cfg["num_cpus"] = num_cpus
    return cfg


class RayHead:
    """The sweep's Ray cluster: started here, sized from the cgroup, torn down here.

    Each run_icl.py would otherwise start its own head that prestarts a worker per core, all assuming
    they own the whole box: simultaneous bring-up deadlocks, and even staggered ones over-book the
    CPUs and turn merely-queued candidates into `cpu_starvation` failures.

    Runs get the address through ``RAY_ADDRESS`` in their environment rather than through
    ``address="auto"``, which would otherwise consult ``/tmp/ray/ray_current_cluster`` — a file
    another machine may have written if /tmp is shared. See sandbox/ray_head.py.
    """

    def __init__(self) -> None:
        self.address: str | None = None
        self.temp_dir: str | None = None
        self.owned = False                       # only tear down a head we started ourselves
        self.env: dict[str, str] = {}

    def ensure(self, mode: str, tpt: int, max_parallel: int, cfg: dict) -> None:
        # Sizing only happens on the path that STARTS the head. Saying so beats letting someone size
        # a sweep carefully and never find out the number was dropped on the floor.
        if cfg.get("num_cpus") is not None and mode != "auto":
            print(f"[sweep] ray: WARNING — num_cpus={cfg['num_cpus']} was requested but "
                  f"--ray-head={mode} does not start a head, so nothing is resized. The cluster's "
                  "own --num-cpus decides how many evals run. Use --ray-head auto to apply it, or "
                  f"start the head yourself with `ray start --head --num-cpus={cfg['num_cpus']}`.")
        if mode == "skip":
            print("[sweep] ray: --ray-head=skip — each run will boot its own cluster")
            return

        stale = ray_head.diagnose_default_address_file()
        if stale:
            print(f"[sweep] ray: NOTE — {stale}")

        existing, refusal = ray_head.head_is_running()
        if refusal:
            print(f"[sweep] ray: WARNING — {refusal}")
        if existing:
            self.address = existing
            # Match the head's thread limits anyway: a head someone started by hand may have been
            # started without them, and the drivers at least should stay consistent.
            self.env = ray_head.thread_env(tpt)
            if existing != "auto":
                self.env["RAY_ADDRESS"] = existing
            # Learn where an ATTACHED head keeps its logs, so they can still be archived. `owned`
            # stays False -- we must never stop or delete a head we did not start -- but archiving is
            # read-only, and this is the case that matters most: jobs/icl_sweep.bsub starts the head
            # itself and passes --ray-head require, so without this the one artifact that could
            # explain a kill is never captured on the path where a kill actually happens.
            attached_tmp = os.environ.get("RAY_TMPDIR")
            if attached_tmp and os.path.isdir(attached_tmp):
                self.temp_dir = attached_tmp
                print(f"[sweep] ray: existing head found ({existing}) — attaching, not resizing it. "
                      f"Its logs ({attached_tmp}) will still be archived on exit. "
                      "Run `python -m sandbox.ray_doctor` if grading looks slow.")
            else:
                print(f"[sweep] ray: existing head found ({existing}) — attaching, not resizing it. "
                      "RAY_TMPDIR is unset, so this sweep CANNOT archive that head's logs: export it "
                      "in the shell that ran `ray start`, or use --ray-head auto and let the sweep "
                      "own the head. Run `python -m sandbox.ray_doctor` if grading looks slow.")
            return

        if mode == "require":
            raise SweepError("no Ray head reachable and --ray-head=require; "
                             "run `ray start --head` first, or use --ray-head=auto")

        plan = ray_head.plan_head(tpt, max_parallel, cfg)
        for note in plan.notes:
            print(f"[sweep] ray: {note}")
        for warning in plan.warnings:
            print(f"[sweep] ray: WARNING — {warning}")
        print(f"[sweep] ray: starting head — {plan.describe()}")
        try:
            self.address = ray_head.start_head(plan)
        except Exception as e:
            raise SweepError(f"could not start the Ray head, refusing to launch runs that would "
                             f"each boot their own cluster:\n{e}") from e
        self.temp_dir = plan.temp_dir
        self.owned = True
        self.env = {**plan.env(), "RAY_ADDRESS": self.address, "RAY_TMPDIR": plan.temp_dir}
        print(f"[sweep] ray: head up at {self.address}  "
              f"(inspect it with: RAY_ADDRESS={self.address} ray status)")

    def child_env(self) -> dict[str, str]:
        return {**os.environ, **self.env}

    # Enough of a big log's tail to cover the kill window, small enough that the whole emergency
    # archive lands inside LSF's ~10 s SIGINT->SIGKILL gap even on a network filesystem.
    SIGNAL_TAIL_BYTES = 8 * 1024 ** 2

    def save_logs(self, dest: str, compress: bool = True,
                  tail_bytes: int | None = None) -> None:
        """Copy the head's session logs somewhere that outlives the compute node's /tmp.

        Gated on ``temp_dir`` rather than on ``owned``: reading a head's logs is safe whether or not
        this sweep started it, and an attached head is exactly the case where they are most likely to
        be lost. Only teardown and deletion require ownership.
        """
        if not self.temp_dir:
            return
        _, note = ray_head.archive_logs(self.temp_dir, dest, compress=compress,
                                        tail_bytes=tail_bytes)
        print(f"[sweep] ray: {note}")

    def teardown(self, log_dest: str | None = None) -> None:
        # Archive BEFORE the ownership gate. An attached head's logs are the ones most likely to be
        # lost (a job script's `rm -rf $RAY_TMPDIR`, or the compute node being reclaimed), and reading
        # them is safe either way — only stopping and deleting need ownership.
        if log_dest:
            self.save_logs(log_dest)
        if not (self.owned and self.temp_dir):
            return
        n = ray_head.stop_head(self.temp_dir)
        print(f"[sweep] ray: stopped the head this sweep started ({n} process(es))")
        # Nothing else reaps these: plan_head gives every sweep its own per-host/per-job temp dir, so
        # without this they accumulate one session per sweep forever. Ray writes one worker log per
        # eval worker (~75 KB), which on this project's own boxes had already reached 5.8 GB / 78k
        # files in /tmp. Safe now: the logs are archived above and stop_head has killed the holders.
        _, note = ray_head.remove_temp_dir(self.temp_dir)
        print(f"[sweep] ray: {note}")


def _flags_from_cmd(cmd: list[str]) -> dict[str, str]:
    """Recover ``{flag: value}`` from a built argv. Read from the manifest rather than re-expanding
    the sweep file, so ``--resume`` checks exactly the commands it is about to run."""
    flags = {}
    for i, tok in enumerate(cmd):
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            flags[tok[2:]] = cmd[i + 1]
    return flags


# Tokens to leave for the parts of the prompt `--max-context-tokens` does NOT bound: the problem
# intro, the rules/current-solution tail (which renders the parent's whole program), and the chat
# template. Deliberately generous — the point is to catch a budget with no headroom at all, and the
# cost of the warning being early is one line of output.
PROMPT_OVERHEAD_TOKENS = 8000


def _check_token_budget(flagsets: list[dict], context_len: dict[str, int | None]) -> None:
    """Warn when a run's token budget cannot fit the server's context window.

    vLLM rejects a request whose prompt plus ``max_tokens`` exceeds ``--max-model-len``, with a 400 —
    which ``icl.loop`` treats as permanent (retrying it cannot help) and stops the run on. Two things
    make this easy to walk into and hard to see coming:

      * ``--max-context-tokens`` bounds only the CONTEXT BLOCK. The intro and the rules/current-
        solution tail are added on top and are not counted against it.
      * ``build_context_block`` trims by a chars/4 ESTIMATE, and code tokenises denser than that, so
        the block can be bigger than the budget it was packed against.

    So a config summing to exactly the window — which is what the cp26/ac1 sweeps do, 94000 + 34000
    against a 128000 window — has no headroom for either effect.
    """
    for f in flagsets:
        model = f.get("model", "openai/gpt-oss-120b")
        window = context_len.get(model)
        ctx, decode = f.get("max-context-tokens"), f.get("max-tokens")
        if not window or ctx is None or decode is None:
            continue
        headroom = int(window) - int(ctx) - int(decode)
        if headroom >= PROMPT_OVERHEAD_TOKENS:
            continue
        print(f"[sweep] WARNING: {model} serves {int(window):,} tokens, but max-context-tokens "
              f"({int(ctx):,}) + max-tokens ({int(decode):,}) leaves only {headroom:,} for the "
              f"problem intro, the rules + current-solution tail and the chat template — and the "
              f"context block is trimmed by a chars/4 estimate, so it can overshoot its own budget. "
              f"A prompt that crosses the window is a 400, which stops the run. Lower "
              f"max-context-tokens to <= {int(window) - int(decode) - PROMPT_OVERHEAD_TOKENS:,}.")


def check_server(entries: list[dict], max_parallel: int, server_max_num_seqs: int | None) -> None:
    """Warn (never block) about the two server-side footguns a sweep can walk into: a model name the
    server does not serve, and more in-flight sequences than the server can co-batch."""
    import urllib.request

    flagsets = [_flags_from_cmd(e["cmd"]) for e in entries]
    urls = {f["vllm-base-url"] for f in flagsets if f.get("vllm-base-url")}
    for url in sorted(urls):
        served: list[str] = []
        try:
            with urllib.request.urlopen(url.rstrip("/") + "/models", timeout=10) as r:
                cards = json.load(r).get("data", [])
            served = [m["id"] for m in cards]
            # vLLM puts the served context length on the model card. Not every build does, so this
            # stays optional — an absent field just skips the budget check below.
            context_len = {m["id"]: m.get("max_model_len") for m in cards}
        except Exception as e:
            print(f"[sweep] WARNING: could not reach {url} ({e}) — launching anyway")
            continue
        here = [f for f in flagsets if f.get("vllm-base-url") == url]
        wanted = {f.get("model", "openai/gpt-oss-120b") for f in here}
        for model in sorted(wanted - set(served)):
            print(f"[sweep] WARNING: {url} does not serve {model!r} (serves: {', '.join(served)}) "
                  f"— every request of those runs will fail")
        _check_token_budget(here, context_len)

    # --max-num-seqs counts SEQUENCES, not requests, and each request asks for n=group-size
    # completions. Comparing request counts against it under-reports the load by that factor.
    loads = []
    for f in flagsets:
        group = int(f.get("group-size", 1))
        chunk = int(f.get("grade-chunk-size", 0) or 0) or group
        n_chunks = -(-group // chunk)                       # ceil: requests each group splits into
        reqs = min(int(f.get("max-gen-concurrency", 8)), int(f.get("groups-per-batch", 1)) * n_chunks)
        loads.append((reqs * chunk, reqs))
    loads.sort(reverse=True)
    peak_seqs = sum(s for s, _ in loads[:max_parallel])
    peak_reqs = sum(r for _, r in loads[:max_parallel])
    top = min(max_parallel, len(loads))
    print(f"[sweep] peak in-flight: {peak_seqs} sequences across {peak_reqs} requests "
          f"(top {top} run(s); a request carries n=group-size completions)")
    if server_max_num_seqs and peak_seqs > server_max_num_seqs:
        print(f"[sweep] WARNING: peak {peak_seqs} sequences exceeds the server's --max-num-seqs "
              f"({server_max_num_seqs}); the excess queues server-side instead of co-batching. "
              "If another sweep shares this server, double it.")


# --------------------------------------------------------------------------------------------------
# manifest + status
# --------------------------------------------------------------------------------------------------
def _read_manifest(sweep_dir: str) -> dict:
    path = os.path.join(sweep_dir, MANIFEST)
    if not os.path.exists(path):
        raise SweepError(f"no {MANIFEST} in {sweep_dir} — is that a sweep directory?")
    with open(path) as f:
        return json.load(f)


def _write_manifest(sweep_dir: str, manifest: dict) -> None:
    os.makedirs(sweep_dir, exist_ok=True)
    tmp = os.path.join(sweep_dir, MANIFEST + ".tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(sweep_dir, MANIFEST))       # atomic: --status may read concurrently


def _proc_start(pid: int) -> int | None:
    """The kernel's start-time stamp for ``pid`` (field 22 of /proc/<pid>/stat), or None.

    PIDs are recycled, and a sweep's manifest outlives the job that wrote it — on a busy compute node
    the pid of a long-dead run is very likely to belong to something else by the time anyone reads the
    manifest. This stamp plus the pid identifies a process uniquely for as long as it lives, which is
    what ``--stop`` needs before it signals anything and what the live-run guard needs before it
    believes a run is still going.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return int(fh.read().rsplit(")", 1)[1].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _halt_path(sweep_dir: str) -> str:
    return os.path.join(sweep_dir, HALT)


def _set_halt(sweep_dir: str) -> None:
    os.makedirs(sweep_dir, exist_ok=True)
    with open(_halt_path(sweep_dir), "w") as f:
        json.dump({"at": datetime.now().isoformat(timespec="seconds"), "by_pid": os.getpid()}, f)


def _clear_halt(sweep_dir: str) -> bool:
    """Drop a previous --stop's marker. Returns whether there was one (so a deliberate relaunch can
    say it is overriding it, rather than silently un-stopping a sweep somebody stopped on purpose)."""
    try:
        os.remove(_halt_path(sweep_dir))
        return True
    except OSError:
        return False


def _alive(pid: int | None, start: int | None = None) -> bool:
    """Is ``pid`` still the process we started? ``start`` is its recorded ``_proc_start`` stamp.

    Without ``start`` this can only answer "some process holds that pid", which is what it used to
    do — and what made a recycled pid read as a running run.
    """
    if not pid:
        return False
    now = _proc_start(pid)
    if now is not None:
        return now == start if start is not None else True
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError) as e:
        return isinstance(e, PermissionError)                # exists but not ours


def _entry_alive(entry: dict) -> bool:
    return _alive(entry.get("pid"), entry.get("pid_start"))


def _live_entries(manifest: dict, names: set[str] | None = None) -> list[dict]:
    """Entries of ``manifest`` whose recorded process is verifiably still running."""
    return [e for e in manifest["entries"]
            if (names is None or e["name"] in names) and _entry_alive(e)]


def _run_progress(log_path: str, num_generations: int | None = None) -> dict:
    """One run's state, VERIFIED against its artifacts (see results.resume).

    ``gens`` is what summary.json claims and ``good`` is what the generation directories, the PUCT
    snapshot and the context pool can actually back. ``complete`` is the launch decision — never
    summary.json's ``status``, which survives the deletion of everything it describes.
    """
    prog = inspect_run(log_path, num_generations)
    return {
        "gens": prog.summary_generations, "status": prog.summary_status,
        "best": prog.best, "wall": prog.wall, "tok_s": prog.tok_s, "updated": prog.updated_at,
        "good": prog.good_generations, "resume_step": prog.resume_step,
        "complete": prog.complete, "damage": prog.damage,
    }


def _state(entry: dict, prog: dict) -> str:
    """Reconcile 'what the manifest says' with 'what is actually true on the box'."""
    if prog.get("complete"):
        return "complete"
    if _entry_alive(entry):
        return "running"
    if entry.get("returncode") not in (None, 0):
        return f"exit {entry['returncode']}"
    if prog.get("status") == "complete":
        # summary.json claims the run finished but its generations do not back that up: data was
        # deleted under it, or an older resume rewrote the summary with only part of the run.
        return "DAMAGED"
    if prog["status"] == "failed":
        return "failed"
    if entry.get("pid") is None:
        return "pending"
    return "DIED"                                            # manifest says running, pid is gone


def _age(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        delta = (datetime.now() - datetime.fromisoformat(iso)).total_seconds()
    except Exception:
        return "-"
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return f"{int(delta)}s ago"


def _entry_generations(entry: dict, manifest: dict) -> int | None:
    return entry.get("num_generations") or manifest.get("num_generations")


def _check_log_path(entry: dict) -> None:
    """The dir we verify and rewind must be the dir the run writes to.

    ``build_specs`` sets ``log_path`` and the command's ``--log-path`` from the same value, so these
    agree unless a manifest was hand-edited — in which case a resume would inspect one run and relaunch
    into another, which is precisely the silent mismatch this module exists to stop."""
    cmd = entry["cmd"]
    if "--log-path" not in cmd:
        raise SweepError(f"{entry['name']}: its recorded command has no --log-path")
    in_cmd = cmd[cmd.index("--log-path") + 1]
    if os.path.normpath(in_cmd) != os.path.normpath(entry["log_path"]):
        raise SweepError(f"{entry['name']}: the manifest's log_path ({entry['log_path']}) and its "
                         f"command's --log-path ({in_cmd}) disagree — fix the manifest before "
                         f"resuming, or the run would be verified in one dir and written in another")


def print_status(sweep_dir: str) -> None:
    manifest = _read_manifest(sweep_dir)
    rows, notes = [], []
    for entry in manifest["entries"]:
        want = _entry_generations(entry, manifest)
        prog = _run_progress(entry["log_path"], want)
        # `gens` is what the artifacts back; where summary.json claims more, the difference is damage
        # and is spelled out under the table rather than hidden behind one number.
        gens = f"{prog['good']}/{want or '?'}"
        rows.append([
            entry["name"], str(entry.get("pid") or "-"), _state(entry, prog), gens,
            f"{prog['best']:.4f}" if isinstance(prog["best"], (int, float)) else "-",
            f"{prog['wall']:.0f}s" if prog["wall"] else "-",
            str(prog["tok_s"] or "-"), _age(prog["updated"]),
        ])
        for line in prog["damage"]:
            notes.append(f"  {entry['name']}: {line}")
    header = ["run", "pid", "state", "gens", "best", "gen wall", "tok/s", "updated"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(f"\n[sweep] {manifest['name']}  ({sweep_dir})")
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    if notes:
        print("\n[sweep] gens counts only VERIFIED generations; --resume restarts any run short of "
              "its target from generation 0:")
        for n in notes:
            print(n)
    print()


def _continue_step(name: str, entry: dict, prog, want: int | None, asked: str) -> int:
    """The generation one named run may continue at, or a SweepError saying why it may not.

    ``asked`` is ``auto`` (the last generation the run's files can back) or a generation number. A
    number is honoured but checked: continuing needs that step's buffer snapshot, and the ways it went
    wrong before were silent or cryptic (a FileNotFoundError from inside the sampler, or a context pool
    that never reached N so every later prompt lost its context block).
    """
    if asked in ("auto", "last"):
        if prog.complete:
            raise SweepError(f"{name} is already complete ({prog.good_generations}/{want}) — nothing "
                             f"to continue. Name a generation ({name}:N) to redo it from there.")
        return prog.resume_step
    try:
        step = int(asked)
    except ValueError:
        raise SweepError(f"{name}: expected a generation number or 'auto', got {asked!r}")
    if step < 0:
        raise SweepError(f"{name}: the generation to continue at must be >= 0")
    if want is not None and step >= want:
        raise SweepError(f"{name}: the run only has {want} generations (0..{want - 1}), so continuing "
                         f"at {step} would leave nothing to run")
    if step and step not in prog.snapshots:
        have = ", ".join(str(s) for s in prog.snapshots) or "none"
        raise SweepError(
            f"{name}: no loadable PUCT snapshot for generation {step} in {entry['log_path']} (have: "
            f"{have}); a run can only continue from a generation whose buffer survived. Use 'auto' for "
            f"{prog.resume_step}, or 0 to start that run over.")
    if step > prog.good_generations:
        print(f"[sweep] continue: {name}: WARNING only {prog.good_generations} generation(s) are "
              f"verifiable, but you asked to continue from {step} — generations "
              f"{prog.good_generations}..{step - 1} are damaged or missing")
    return step


def guard_live_runs(sweep_dir: str, names: set[str] | None = None, *, force: bool = False,
                    action: str = "relaunch") -> None:
    """Refuse to touch runs that are still going, unless a human says otherwise.

    Everything downstream of here rewinds a run directory and starts a second process writing into
    it: two run_icl.py appending to one events.jsonl, one rewinding the generations the other is
    still producing. The manifest already knows which pids were launched, so this is only a question
    of asking — which nothing did.

    On a terminal, ask. Off one (a batch job, a nohup, a cron) there is nobody to ask, so refuse and
    say what to do about it: proceeding silently is the failure this exists to prevent, and a job that
    exits non-zero is cheap next to a corrupted run directory. ``--force`` skips the check entirely.
    """
    if force:
        return
    try:
        manifest = _read_manifest(sweep_dir)
    except SweepError:
        return                                          # no manifest yet: nothing can be running
    live = _live_entries(manifest, names)
    if not live:
        return
    listing = ", ".join(f"{e['name']} (pid {e['pid']}, started {e.get('started_at') or '?'})"
                        for e in live)
    print(f"\n[sweep] WARNING: {len(live)} run(s) of this sweep are STILL RUNNING: {listing}")
    print(f"[sweep] A {action} would rewind their run directories under the live processes and start "
          f"a second writer in each — the two accounts interleave in one events.jsonl and neither is "
          f"recoverable.")
    if not sys.stdin.isatty():
        raise SweepError(
            f"refusing to {action} {len(live)} live run(s) with no terminal to confirm on. Stop them "
            f"first (python run_sweep.py --stop {sweep_dir}), wait for them to finish, or pass "
            f"--force if you are certain those pids are stale.")
    answer = input(f"[sweep] Go ahead with the {action} anyway? Type 'yes' to continue: ").strip().lower()
    if answer != "yes":
        raise SweepError(f"cancelled — the {len(live)} live run(s) were left alone")


def plan_runs(manifest: dict, *, continue_at: dict[str, str] | None = None,
              restart_others: bool = True, dry_run: bool = False) -> set[str]:
    """Decide what each run of a sweep does next, make its dir match, and return the names to queue.

    Two dials, because the two costs pull opposite ways:

    ``restart_others=True`` (a ``--resume``) requeues every run that is not verifiably complete FROM
    ITS FIRST GENERATION. Continuing mid-run is cheaper, but it makes a run's generations a mixture of
    two processes — with the interruption's cause (a dead server, a throttled box, an evicted job)
    sitting somewhere inside the search it produced, and nothing in the results saying where. One
    process, one context pool, one clean lineage is what the arms are compared on.

    ``continue_at={name: "auto"|"N"}`` exempts named runs from that: they continue where they stopped
    instead, because redoing 12 generations of a long run can cost more than the mixed-lineage caveat
    is worth. Both dials compose — continue the one expensive run, restart the rest of the sweep,
    supervise the lot as one queue.

    Either way the run dir is made to match the command that will run: everything from the resume point
    on goes to ``stale_<timestamp>/`` (nothing deleted), so a relaunch cannot append onto the attempt it
    replaces. Verifiably complete runs are left untouched and are not queued.
    """
    continue_at = {k: (v or "auto").strip().lower() for k, v in (continue_at or {}).items()}
    names = [e["name"] for e in manifest["entries"]]
    unknown = sorted(set(continue_at) - set(names))
    if unknown:
        raise SweepError(f"not run(s) of this sweep: {', '.join(unknown)} "
                         f"(runs: {', '.join(names)})")

    queued: set[str] = set()
    for entry in manifest["entries"]:
        name = entry["name"]
        asked = continue_at.get(name)
        if asked is None and not restart_others:
            continue                                   # this call is only about the named runs
        _check_log_path(entry)
        want = _entry_generations(entry, manifest)
        prog = inspect_run(entry["log_path"], want)
        tag = "continue" if asked is not None else "resume"
        for line in prog.damage:
            print(f"[sweep] {tag}: {name}: {line}")

        if asked is not None:
            step = _continue_step(name, entry, prog, want, asked)
            verified = " (the last generation its files can back)" if asked in ("auto", "last") else ""
            where = "generation 0, from scratch" if step == 0 else f"generation {step}{verified}"
            print(f"[sweep] continue: {name}: {prog.good_generations}/{want or '?'} verified -> "
                  f"continuing at {where}"
                  + (" — nothing moved: --print-cmds" if dry_run else ""))
        elif prog.complete:
            print(f"[sweep] resume: {name}: {prog.describe()} — skipping")
            continue
        else:
            step = 0
            # Report what is being MOVED, not what verifies. A run damaged at generation 2 of 15 has
            # 2 verified generations and 15 on disk, and it was the smaller number that got printed —
            # so "discarding 2 generation(s)" was the caption on throwing away thirteen more.
            on_disk = prog.generations_on_disk
            had = ""
            if on_disk:
                had = f" (discarding {on_disk} generation(s) of work"
                had += (f", only {prog.good_generations} of them verified)"
                        if prog.good_generations != on_disk else ")")
            print(f"[sweep] resume: {name}: restarting from generation 0{had}"
                  + (" — nothing moved: --print-cmds" if dry_run and prog.has_tail else ""))

        if not dry_run and tail_exists(entry["log_path"], step):
            for line in rewind(entry["log_path"], step):
                print(f"[sweep] {tag}: {name}: moved {line}")
        # Rebuild the flag from the decision: a stale --resume-step from an earlier relaunch must not
        # survive into a run that now starts over, and a restarted run must carry none.
        cmd = [tok for tok in entry["cmd"]]
        if "--resume-step" in cmd:
            i = cmd.index("--resume-step")
            del cmd[i:i + 2]
        if step:
            cmd += ["--resume-step", str(step)]
        entry["cmd"] = cmd
        entry["pid"], entry["returncode"] = None, None
        queued.add(name)
    return queued


def plan_resume(manifest: dict, *, continue_at: dict[str, str] | None = None,
                dry_run: bool = False) -> set[str]:
    """A whole-sweep resume: restart every incomplete run, except those ``continue_at`` names."""
    return plan_runs(manifest, continue_at=continue_at, restart_others=True, dry_run=dry_run)


def plan_continue_run(run_dir: str, from_generation: str | None = None, *,
                      dry_run: bool = False) -> tuple[str, dict, dict, str]:
    """Continue ONE run of a sweep and touch nothing else — not even the sweep's other unfinished runs.

    Returns (sweep_dir, settings, manifest, run_name); the manifest is the sweep's FULL manifest, so
    ``--status`` keeps working, and only ``run_name`` is queued.
    """
    run_dir = os.path.normpath(run_dir)
    sweep_dir = os.path.dirname(run_dir) or "."
    name = os.path.basename(run_dir)
    manifest = _read_manifest(sweep_dir)
    plan_runs(manifest, continue_at={name: from_generation or "auto"}, restart_others=False,
              dry_run=dry_run)
    return sweep_dir, (manifest.get("settings") or {}), manifest, name


def parse_continue_specs(values: list[str], from_generation: str | None,
                         sweep_dir: str | None) -> tuple[str, dict[str, str]]:
    """Turn ``--continue-run`` values into (sweep_dir, {run name: generation}).

    A value is a run name, a name with the generation appended (``n10_s3_random:7``), or the run's
    directory in either form — the path is what shell completion gives you, and it is what says which
    sweep is meant when there is no ``--resume``.
    """
    resolved: dict[str, str] = {}
    for raw in values:
        value, gen = raw, None
        head, sep, tail = raw.rpartition(":")
        if sep and (tail.isdigit() or tail.strip().lower() in ("auto", "last")):
            value, gen = head, tail.strip().lower()
        value = os.path.normpath(value.rstrip(os.sep))
        if os.sep in value:
            parent, name = os.path.dirname(value), os.path.basename(value)
            if sweep_dir is not None and os.path.normpath(sweep_dir) != parent:
                raise SweepError(f"{raw}: that run is in {parent}, but the sweep being resumed is "
                                 f"{sweep_dir} — pass just the run name, or resume the other sweep")
            sweep_dir = parent
        else:
            name = value
            if sweep_dir is None:
                raise SweepError(f"--continue-run {raw}: give the run's directory "
                                 f"(runs/<sweep>/{name}) so the sweep is unambiguous, or add "
                                 f"--resume runs/<sweep> to keep the rest of the sweep going")
        if name in resolved:
            raise SweepError(f"--continue-run {name}: named twice")
        resolved[name] = gen or "auto"

    if from_generation is not None:
        if len(resolved) != 1:
            raise SweepError("--from-generation applies to a single --continue-run; with several, "
                             "append the generation to each (--continue-run NAME:7)")
        name, gen = next(iter(resolved.items()))
        if gen != "auto":
            raise SweepError(f"--from-generation and {name}:{gen} both say where to continue — keep one")
        resolved[name] = from_generation.strip().lower()
    return sweep_dir, resolved


# --------------------------------------------------------------------------------------------------
# launching
# --------------------------------------------------------------------------------------------------
def _clear_for_launch(entry: dict) -> None:
    """Make the run dir match the command about to run.

    A relaunch writes into the dir the previous attempt left behind: without ``--resume-step`` it
    starts at generation 0 while events.jsonl / progress.csv / solutions/ still hold the old attempt,
    and the two accounts interleave in one file (we have a run dir on disk with 21 progress rows for
    15 generations, and generation dirs written before generation 0's). Whatever the command does not
    claim goes to stale_<timestamp>/ first.
    """
    cmd = entry["cmd"]
    step = 0
    if "--resume-step" in cmd:
        try:
            step = int(cmd[cmd.index("--resume-step") + 1])
        except (IndexError, ValueError):
            step = 0
    if tail_exists(entry["log_path"], step):
        where = "generation 0" if step == 0 else f"generation {step}"
        print(f"[sweep] {entry['name']}: relaunching at {where}; moving what is already on disk aside")
        for line in rewind(entry["log_path"], step):
            print(f"[sweep] {entry['name']}: moved {line}")


def plan_queue(manifest: dict, only: set[str] | None = None) -> tuple[list[dict], list[str]]:
    """Split a sweep's entries into (to launch, skipped-as-complete-with-reasons).

    Two rules, and no third one:

      * ``only`` — the runs this invocation planned. ``None`` means every entry is a candidate.
      * the ARTIFACTS say whether a run is finished (``results.resume``), not summary.json's status and
        NOT the manifest's ``returncode``. That return code is one number remembered from a past
        process, and "exited 0" is not "did all its generations": a run killed at a generation boundary
        by LSF, or one whose relaunch exited 0 early, carried a 0 that skipped it from then on — the
        sweep would print a status table and exit having launched nothing.
    """
    pending, skipped = [], []
    for entry in manifest["entries"]:
        if only is not None and entry["name"] not in only:
            continue
        want = _entry_generations(entry, manifest)
        prog = _run_progress(entry["log_path"], want)
        if prog["complete"]:
            skipped.append(f"{entry['name']} (complete: {prog['good']}/{want or '?'} verified)")
        else:
            pending.append(entry)
    return pending, skipped


def _launch(entry: dict, env: dict[str, str] | None = None) -> subprocess.Popen:
    _clear_for_launch(entry)
    os.makedirs(entry["log_path"], exist_ok=True)
    out = open(os.path.join(entry["log_path"], "launch.out"), "a")
    # start_new_session: the run gets its own process group, so Ctrl-C / death of this supervisor
    # does not take the runs down with it.
    proc = subprocess.Popen(entry["cmd"], cwd=HERE, stdout=out,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    entry["pid"] = proc.pid
    # Recorded alongside the pid so a later invocation can tell "this run is still going" from "some
    # unrelated process inherited that pid after the node recycled it". See _alive.
    entry["pid_start"] = _proc_start(proc.pid)
    entry["started_at"] = datetime.now().isoformat(timespec="seconds")
    entry["returncode"] = None
    return proc


def supervise(sweep_dir: str, manifest: dict, stagger: float, max_parallel: int, refresh: float,
              env: dict[str, str] | None = None, only: set[str] | None = None,
              on_signal: Any = None) -> int:
    """Run the queue to completion: at most ``max_parallel`` runs in flight, ``stagger`` seconds
    between launches, a status table every ``refresh`` seconds. Returns the number that failed.

    ``only`` restricts what may be launched to those run names (``--continue-run`` queues exactly one)
    while the manifest written back stays the sweep's full one.
    """
    pending, skipped = plan_queue(manifest, only)
    print(f"[sweep] queue: {len(pending)} run(s) to launch"
          + (": " + ", ".join(e["name"] for e in pending) if pending else ""))
    if skipped:
        print(f"[sweep] queue: skipping {len(skipped)} verifiably complete run(s): "
              + ", ".join(skipped))
    if not pending:
        print("[sweep] nothing to launch. `--status` shows each run's verified generation count; "
              "`--resume` restarts anything short of its target.")
    launched = 0
    live: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    last_render = 0.0
    failed = 0

    def shutdown(signum, _frame):
        print(f"\n[sweep] signal {signum} — leaving {len(live)} run(s) alive "
              f"(stop them with: python run_sweep.py --stop {sweep_dir})")
        _write_manifest(sweep_dir, manifest)
        # The signal path is the one that matters for a post-mortem: an LSF TERM_MEMLIMIT arrives
        # here, and if the logs are still in the compute node's /tmp when the node is reclaimed there
        # is nothing left to read. Uncompressed, because LSF follows up with SIGKILL 10 s later.
        if on_signal is not None:
            try:
                on_signal()
            except Exception as e:
                print(f"[sweep] could not save Ray logs on the way out: {e!r}")
        sys.exit(130)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    halted = False
    while pending or live:
        # --stop's marker. Without this, SIGTERMing a run only frees a slot: the loop below reaps it
        # and immediately launches the next queued one, so `--stop` on a 12-run sweep at
        # max_parallel 2 killed two runs and started two more.
        if pending and not halted and os.path.exists(_halt_path(sweep_dir)):
            halted = True
            print(f"[sweep] halted by --stop: dropping {len(pending)} queued run(s) "
                  + ", ".join(e["name"] for e in pending)
                  + (f"; waiting for {len(live)} still in flight" if live else ""))
            pending = []
        for name, proc in list(live.items()):
            rc = proc.poll()
            if rc is None:
                continue
            entry = next(e for e in manifest["entries"] if e["name"] == name)
            entry["returncode"] = rc
            entry["finished_at"] = datetime.now().isoformat(timespec="seconds")
            failed += rc != 0
            print(f"[sweep] {name} finished (exit {rc})"
                  + ("" if rc == 0 else f" — see {entry['log_path']}/launch.out"))
            live.pop(name)
            _write_manifest(sweep_dir, manifest)

        now = time.monotonic()
        if pending and len(live) < max_parallel and (not live or now - last_launch >= stagger):
            entry = pending.pop(0)
            live[entry["name"]] = _launch(entry, env)
            last_launch = now
            launched += 1
            done = launched - len(live)
            print(f"[sweep] launched {entry['name']} (pid {entry['pid']}) — "
                  f"{len(live)} running, {len(pending)} queued, {done} finished")
            _write_manifest(sweep_dir, manifest)

        if time.monotonic() - last_render >= refresh:
            print_status(sweep_dir)
            last_render = time.monotonic()
        time.sleep(1.0)

    _write_manifest(sweep_dir, manifest)
    print_status(sweep_dir)
    # Counts of what THIS invocation did. It used to report len(entries) - failed as "ok", which called
    # a sweep that launched nothing "12 ok".
    print(f"[sweep] done: {launched} launched, {launched - failed} ok, {failed} failed")
    return failed


def stop_sweep(sweep_dir: str) -> None:
    """Halt a sweep: stop what is queued, then signal what is live.

    The marker goes down FIRST. A supervisor reaps an exiting run within a second and fills the slot
    it freed, so signalling before halting is a race this would usually lose.
    """
    manifest = _read_manifest(sweep_dir)
    _set_halt(sweep_dir)
    print(f"[sweep] halt marker written to {_halt_path(sweep_dir)} — a running supervisor will drop "
          f"its queue (relaunching this sweep clears it)")
    stopped = 0
    for entry in manifest["entries"]:
        if not _entry_alive(entry):
            continue
        pid = entry["pid"]
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)        # the whole run_icl process group
        except Exception as e:
            # Fall back to the bare pid, but keep going: one unkillable entry must not leave the rest
            # of the sweep running, which an unguarded os.kill here used to do.
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception as e2:
                print(f"[sweep] could NOT stop {entry['name']} (pid {pid}): {e!r} / {e2!r}")
                continue
        print(f"[sweep] SIGTERM -> {entry['name']} (pid {pid})")
        stopped += 1
    print(f"[sweep] stopped {stopped} run(s)")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def build_specs(sweep_file: str, sweep_dir_override: str | None) -> tuple[str, dict, dict]:
    """Parse + validate the sweep file into (sweep_dir, settings, manifest). Launches nothing."""
    with open(sweep_file) as f:
        raw = yaml.safe_load(f) or {}
    unknown = set(raw) - {"sweep", "common", "grid", "runs"}
    if unknown:
        raise SweepError(f"unknown top-level key(s) {sorted(unknown)}; expected sweep/common/grid/runs")

    settings = dict(raw.get("sweep") or {})
    unknown = set(settings) - {"name", "max_parallel", "stagger", "server_max_num_seqs", "ray"}
    if unknown:
        raise SweepError(f"unknown key(s) in `sweep`: {sorted(unknown)}")
    # plan_head reads this block with cfg.get(), so a typo here would otherwise be silently ignored
    # and the sweep would run at a size nobody chose -- the same failure mode as not setting it at all.
    unknown = set(settings.get("ray") or {}) - RAY_SETTINGS
    if unknown:
        raise SweepError(f"unknown key(s) in `sweep.ray`: {sorted(unknown)}; "
                         f"expected {sorted(RAY_SETTINGS)}")
    name = settings.get("name") or os.path.splitext(os.path.basename(sweep_file))[0]
    sweep_dir = sweep_dir_override or os.path.join("runs", name)

    by_flag, bool_pair = _flag_tables()
    specs = _expand(raw)
    entries = []
    for spec in specs:
        log_path = os.path.join(sweep_dir, spec["name"])
        argv = _to_argv(spec["flags"], by_flag, bool_pair)
        entries.append({
            "name": spec["name"],
            "log_path": log_path,
            "num_generations": spec["flags"].get("num-generations"),
            "cmd": [sys.executable, "run_icl.py", *argv, "--log-path", log_path],
            "pid": None, "started_at": None, "finished_at": None, "returncode": None,
        })
    manifest = {
        "name": name,
        "sweep_file": os.path.abspath(sweep_file),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": settings,
        "entries": entries,
    }
    return sweep_dir, settings, manifest


def main() -> int:
    p = argparse.ArgumentParser(
        description="Launch and track several ICL runs from one sweep file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run under tmux; the supervisor must stay alive to enforce --max-parallel.")
    p.add_argument("sweep_file", nargs="?", help="YAML sweep file (see this module's docstring).")
    p.add_argument("--sweep-dir", default=None,
                   help="Override the output dir (default: runs/<sweep name>).")
    p.add_argument("--status", metavar="SWEEP_DIR", help="Print a sweep's status table and exit.")
    p.add_argument("--stop", metavar="SWEEP_DIR",
                   help="Halt a sweep: drop whatever is still queued and SIGTERM every live run. A "
                        "supervisor watching this sweep sees the halt marker and stops launching "
                        "(without it, killing a run only freed a slot and the next queued run "
                        "started immediately). Relaunching the sweep clears the marker.")
    p.add_argument("--resume", metavar="SWEEP_DIR",
                   help="Restart every run of a sweep that is not complete, from its FIRST generation "
                        "(a partial run is never continued mid-run: one run = one process = one "
                        "lineage). 'Complete' is verified against the run dir, never read off "
                        "summary.json; a restarted run's old dir is moved to <run>/stale_<timestamp>/ "
                        "rather than deleted. Combine with --print-cmds to see the decisions, and how "
                        "much each restart discards, without touching anything.")
    p.add_argument("--continue-run", metavar="RUN[:N]", action="append",
                   help="Continue a run mid-run instead of restarting it, reusing the command line the "
                        "sweep recorded for it. Takes a run directory (runs/<sweep>/<run>) or, "
                        "alongside --resume, just the run name; append :N to name the generation. "
                        "Repeatable. On its own it touches only the runs named; WITH --resume the rest "
                        "of the sweep is planned as usual and everything runs in one queue.")
    p.add_argument("--from-generation", metavar="N|auto", default=None,
                   help="The generation a single --continue-run continues at (same as RUN:N). 'auto' "
                        "(default) uses the last verifiable one; N is checked against the run's buffer "
                        "snapshots; 0 restarts that run. Generations from N on move to "
                        "<run>/stale_<timestamp>/.")
    p.add_argument("--print-cmds", action="store_true",
                   help="Expand the sweep and print the exact commands; launch nothing.")
    p.add_argument("--force", action="store_true",
                   help="Skip the live-run check. By default a launch/resume that would rewind a run "
                        "whose process is still alive asks first (and refuses outright when there is "
                        "no terminal to ask on).")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="Max runs in flight (default: sweep.max_parallel, else 1).")
    p.add_argument("--stagger", type=float, default=None,
                   help="Seconds between launches (default: sweep.stagger, else 120).")
    p.add_argument("--refresh", type=float, default=60.0, help="Status-table interval, seconds.")
    p.add_argument("--ray-head", choices=["auto", "require", "skip"], default="auto",
                   help="auto: start a shared Ray head sized from this box's cgroup if none is up, "
                        "and stop it when the sweep drains; require: attach to an existing head, "
                        "fail if none; skip: leave Ray alone (each run boots its own cluster).")
    p.add_argument("--ray-num-cpus", type=int, default=None, metavar="N",
                   help="Cores the Ray head may use, instead of detecting them. This is the final "
                        "--num-cpus: reserve_cpus is not applied on top and the rest of the "
                        "allocation is left idle. Overrides sweep.ray.num_cpus; only meaningful "
                        "with --ray-head auto, since an existing head cannot be resized.")
    a = p.parse_args()

    if a.status:
        print_status(a.status)
        return 0
    if a.stop:
        stop_sweep(a.stop)
        return 0

    if a.from_generation is not None and not a.continue_run:
        p.error("--from-generation only applies to --continue-run (--resume always starts at 0)")

    only: set[str] | None = None
    if a.continue_run or a.resume:
        # --continue-run alone touches only the runs it names; with --resume the rest of the sweep is
        # planned too (incomplete restarted, complete skipped) and the lot is supervised as one queue.
        sweep_dir, continue_at = parse_continue_specs(
            a.continue_run or [], a.from_generation, a.resume)
        manifest = _read_manifest(sweep_dir)
        settings = manifest.get("settings") or {}
        if not a.print_cmds:
            # Before plan_runs, which is what rewinds the run dirs. --print-cmds moves nothing, so it
            # is free to describe a sweep that is currently running.
            named = set(continue_at) if not a.resume else None
            guard_live_runs(sweep_dir, named, force=a.force,
                            action="continue" if not a.resume else "resume")
        only = plan_runs(manifest, continue_at=continue_at, restart_others=bool(a.resume),
                         dry_run=a.print_cmds)
        if not only:
            print("[sweep] nothing to launch — every run of this sweep verifies as complete")
            return 0
        if not a.print_cmds:
            _write_manifest(sweep_dir, manifest)
    else:
        if not a.sweep_file:
            p.error("give a sweep file, or one of --status/--stop/--resume/--continue-run")
        sweep_dir, settings, manifest = build_specs(a.sweep_file, a.sweep_dir)
        # build_specs writes a manifest with no pids, so the check has to read the one already on
        # disk — relaunching the same sweep file over a live sweep is the easiest way into this mess
        # (it is what a resubmitted batch job does) and the one the new manifest cannot see.
        if not a.print_cmds:
            guard_live_runs(sweep_dir, force=a.force, action="relaunch")

    launching = [e for e in manifest["entries"] if only is None or e["name"] in only]
    single = only is not None and len(only) == 1
    max_parallel = a.max_parallel or (1 if single else int(settings.get("max_parallel", 1)))
    stagger = a.stagger if a.stagger is not None else float(settings.get("stagger", 120))

    if a.print_cmds:
        print(f"[sweep] {manifest['name']}: {len(launching)} run(s) -> {sweep_dir}")
        print(f"[sweep] max_parallel={max_parallel} stagger={stagger:g}s")
        # Show exactly how the head would be sized on THIS box without starting it: the numbers come
        # from the cgroup, so this is the way to check a machine before committing a sweep to it.
        tpt = threads_per_task(launching)
        stale = ray_head.diagnose_default_address_file()
        if stale:
            print(f"\n[sweep] ray: NOTE — {stale}")
        try:
            plan = ray_head.plan_head(tpt, max_parallel, _ray_cfg(settings, a.ray_num_cpus))
            print("\n# ray head that would be started")
            for note in plan.notes:
                print(f"#   {note}")
            for warning in plan.warnings:
                print(f"#   WARNING: {warning}")
            print(f"#   env: " + " ".join(f"{k}={v}" for k, v in plan.env().items()))
            print(" ".join(plan.argv()))
        except ValueError as e:
            print(f"\n# ray head CANNOT be sized on this box: {e}")
        for entry in launching:
            print(f"\n# {entry['name']}\n" + " ".join(entry["cmd"]))
        return 0

    check_server(launching, max_parallel, settings.get("server_max_num_seqs"))

    tpt = threads_per_task(launching)
    head = RayHead()
    head.ensure(a.ray_head, tpt, max_parallel, _ray_cfg(settings, a.ray_num_cpus))

    _write_manifest(sweep_dir, manifest)
    # Launching IS the deliberate act that overrides a previous --stop; leaving the marker would halt
    # this sweep on its first pass.
    if _clear_halt(sweep_dir):
        print("[sweep] cleared the halt marker left by a previous --stop")
    print(f"[sweep] {manifest['name']}: {len(launching)} run(s) -> {sweep_dir} "
          f"(max_parallel={max_parallel}, stagger={stagger:g}s)")
    # supervise() exits the process on SIGINT/SIGTERM and deliberately leaves the runs alive, so the
    # head must survive that path — only tear it down when the queue actually drained.
    failed = supervise(sweep_dir, manifest, stagger, max_parallel, a.refresh, head.child_env(),
                       only=only,
                       on_signal=lambda: head.save_logs(
                           sweep_dir, compress=False,
                           tail_bytes=RayHead.SIGNAL_TAIL_BYTES))
    head.teardown(log_dest=sweep_dir)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SweepError as e:
        print(f"[sweep] ERROR: {e}", file=sys.stderr)
        sys.exit(2)
