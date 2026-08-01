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
    python run_sweep.py --resume runs/ctx_strategies               # relaunch whatever is not complete
    python run_sweep.py --stop   runs/ctx_strategies               # SIGTERM every live run

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
        reserve_cpus: 3             # cores kept off Ray for the supervisor + drivers
                                    #   (default 1 + max_parallel)
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

from run_icl import build_parser
from sandbox import ray_head

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = "sweep.json"

# Flags the sweep file must not set: the launcher owns them.
RESERVED = {"log-path", "resume-step"}


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
            print(f"[sweep] ray: existing head found ({existing}) — attaching, not resizing it. "
                  "Run `python -m sandbox.ray_doctor` if grading looks slow.")
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

    def teardown(self) -> None:
        if not (self.owned and self.temp_dir):
            return
        n = ray_head.stop_head(self.temp_dir)
        print(f"[sweep] ray: stopped the head this sweep started ({n} process(es))")


def _flags_from_cmd(cmd: list[str]) -> dict[str, str]:
    """Recover ``{flag: value}`` from a built argv. Read from the manifest rather than re-expanding
    the sweep file, so ``--resume`` checks exactly the commands it is about to run."""
    flags = {}
    for i, tok in enumerate(cmd):
        if tok.startswith("--") and i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
            flags[tok[2:]] = cmd[i + 1]
    return flags


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
                served = [m["id"] for m in json.load(r).get("data", [])]
        except Exception as e:
            print(f"[sweep] WARNING: could not reach {url} ({e}) — launching anyway")
            continue
        wanted = {f.get("model", "openai/gpt-oss-120b") for f in flagsets
                  if f.get("vllm-base-url") == url}
        for model in sorted(wanted - set(served)):
            print(f"[sweep] WARNING: {url} does not serve {model!r} (serves: {', '.join(served)}) "
                  f"— every request of those runs will fail")

    concurrency = sorted((int(f.get("max-gen-concurrency", 8)) for f in flagsets), reverse=True)
    peak = sum(concurrency[:max_parallel])
    print(f"[sweep] peak in-flight requests: {peak} "
          f"(top {min(max_parallel, len(concurrency))} runs' --max-gen-concurrency)")
    if server_max_num_seqs and peak > server_max_num_seqs:
        print(f"[sweep] WARNING: peak {peak} exceeds the server's --max-num-seqs "
              f"({server_max_num_seqs}); the excess queues server-side instead of co-batching")


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


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError) as e:
        return isinstance(e, PermissionError)                # exists but not ours


def _run_progress(log_path: str) -> dict:
    """Latest per-generation numbers for one run, read from the files the tracker already writes."""
    out = {"gens": 0, "best": None, "status": None, "wall": None, "tok_s": None, "updated": None}
    summary = os.path.join(log_path, "summary.json")
    if os.path.exists(summary):
        try:
            with open(summary) as f:
                d = json.load(f)
            out["status"] = d.get("status")
            out["gens"] = len(d.get("per_generation") or [])
            out["best"] = (d.get("best") or {}).get("score")
            out["updated"] = d.get("updated_at")
            per_gen = d.get("per_generation") or []
            if per_gen:
                last = per_gen[-1]
                wall = last.get("wall_seconds")
                ct = (last.get("usage") or {}).get("completion_tokens") or 0
                out["wall"] = wall
                out["tok_s"] = round(ct / wall) if wall else None
        except Exception:
            pass                                             # a half-written summary is not an error
    return out


def _state(entry: dict, prog: dict) -> str:
    """Reconcile 'what the manifest says' with 'what is actually true on the box'."""
    if prog["status"] == "complete":
        return "complete"
    if _alive(entry.get("pid")):
        return "running"
    if entry.get("returncode") not in (None, 0):
        return f"exit {entry['returncode']}"
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


def print_status(sweep_dir: str) -> None:
    manifest = _read_manifest(sweep_dir)
    total = manifest.get("num_generations")
    rows = []
    for entry in manifest["entries"]:
        prog = _run_progress(entry["log_path"])
        gens = f"{prog['gens']}/{entry.get('num_generations') or total or '?'}"
        rows.append([
            entry["name"], str(entry.get("pid") or "-"), _state(entry, prog), gens,
            f"{prog['best']:.4f}" if isinstance(prog["best"], (int, float)) else "-",
            f"{prog['wall']:.0f}s" if prog["wall"] else "-",
            str(prog["tok_s"] or "-"), _age(prog["updated"]),
        ])
    header = ["run", "pid", "state", "gens", "best", "gen wall", "tok/s", "updated"]
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h)
              for i, h in enumerate(header)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths))
    print(f"\n[sweep] {manifest['name']}  ({sweep_dir})")
    print(line)
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    print()


# --------------------------------------------------------------------------------------------------
# launching
# --------------------------------------------------------------------------------------------------
def _launch(entry: dict, env: dict[str, str] | None = None) -> subprocess.Popen:
    os.makedirs(entry["log_path"], exist_ok=True)
    out = open(os.path.join(entry["log_path"], "launch.out"), "a")
    # start_new_session: the run gets its own process group, so Ctrl-C / death of this supervisor
    # does not take the runs down with it.
    proc = subprocess.Popen(entry["cmd"], cwd=HERE, stdout=out,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    entry["pid"] = proc.pid
    entry["started_at"] = datetime.now().isoformat(timespec="seconds")
    entry["returncode"] = None
    return proc


def supervise(sweep_dir: str, manifest: dict, stagger: float,
              max_parallel: int, refresh: float, env: dict[str, str] | None = None) -> int:
    """Run the queue to completion: at most ``max_parallel`` runs in flight, ``stagger`` seconds
    between launches, a status table every ``refresh`` seconds. Returns the number that failed."""
    pending = [e for e in manifest["entries"] if e.get("returncode") != 0
               and _run_progress(e["log_path"])["status"] != "complete"]
    live: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    last_render = 0.0
    failed = 0

    def shutdown(signum, _frame):
        print(f"\n[sweep] signal {signum} — leaving {len(live)} run(s) alive "
              f"(stop them with: python run_sweep.py --stop {sweep_dir})")
        _write_manifest(sweep_dir, manifest)
        sys.exit(130)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    while pending or live:
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
            done = len(manifest["entries"]) - len(pending) - len(live)
            print(f"[sweep] launched {entry['name']} (pid {entry['pid']}) — "
                  f"{len(live)} running, {len(pending)} queued, {done} finished")
            _write_manifest(sweep_dir, manifest)

        if time.monotonic() - last_render >= refresh:
            print_status(sweep_dir)
            last_render = time.monotonic()
        time.sleep(1.0)

    _write_manifest(sweep_dir, manifest)
    print_status(sweep_dir)
    print(f"[sweep] done: {len(manifest['entries']) - failed} ok, {failed} failed")
    return failed


def stop_sweep(sweep_dir: str) -> None:
    manifest = _read_manifest(sweep_dir)
    stopped = 0
    for entry in manifest["entries"]:
        if _alive(entry.get("pid")):
            try:
                os.killpg(os.getpgid(entry["pid"]), signal.SIGTERM)   # the whole run_icl process group
            except Exception:
                os.kill(entry["pid"], signal.SIGTERM)
            print(f"[sweep] SIGTERM -> {entry['name']} (pid {entry['pid']})")
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
    p.add_argument("--stop", metavar="SWEEP_DIR", help="SIGTERM every live run of a sweep and exit.")
    p.add_argument("--resume", metavar="SWEEP_DIR",
                   help="Relaunch every run of a sweep that is not complete (keeps its log dir).")
    p.add_argument("--print-cmds", action="store_true",
                   help="Expand the sweep and print the exact commands; launch nothing.")
    p.add_argument("--max-parallel", type=int, default=None,
                   help="Max runs in flight (default: sweep.max_parallel, else 1).")
    p.add_argument("--stagger", type=float, default=None,
                   help="Seconds between launches (default: sweep.stagger, else 120).")
    p.add_argument("--refresh", type=float, default=60.0, help="Status-table interval, seconds.")
    p.add_argument("--ray-head", choices=["auto", "require", "skip"], default="auto",
                   help="auto: start a shared Ray head sized from this box's cgroup if none is up, "
                        "and stop it when the sweep drains; require: attach to an existing head, "
                        "fail if none; skip: leave Ray alone (each run boots its own cluster).")
    a = p.parse_args()

    if a.status:
        print_status(a.status)
        return 0
    if a.stop:
        stop_sweep(a.stop)
        return 0

    if a.resume:
        sweep_dir = a.resume
        manifest = _read_manifest(sweep_dir)
        settings = manifest.get("settings") or {}
        for entry in manifest["entries"]:
            prog = _run_progress(entry["log_path"])
            if prog["status"] != "complete" and prog["gens"]:
                # Continue where the run stopped rather than redoing finished generations.
                if "--resume-step" not in entry["cmd"]:
                    entry["cmd"] += ["--resume-step", str(prog["gens"])]
                else:
                    entry["cmd"][entry["cmd"].index("--resume-step") + 1] = str(prog["gens"])
            entry["pid"], entry["returncode"] = None, None
    else:
        if not a.sweep_file:
            p.error("give a sweep file, or one of --status/--stop/--resume")
        sweep_dir, settings, manifest = build_specs(a.sweep_file, a.sweep_dir)

    max_parallel = a.max_parallel or int(settings.get("max_parallel", 1))
    stagger = a.stagger if a.stagger is not None else float(settings.get("stagger", 120))

    if a.print_cmds:
        print(f"[sweep] {manifest['name']}: {len(manifest['entries'])} run(s) -> {sweep_dir}")
        print(f"[sweep] max_parallel={max_parallel} stagger={stagger:g}s")
        # Show exactly how the head would be sized on THIS box without starting it: the numbers come
        # from the cgroup, so this is the way to check a machine before committing a sweep to it.
        tpt = threads_per_task(manifest["entries"])
        stale = ray_head.diagnose_default_address_file()
        if stale:
            print(f"\n[sweep] ray: NOTE — {stale}")
        try:
            plan = ray_head.plan_head(tpt, max_parallel, dict(settings.get("ray") or {}))
            print("\n# ray head that would be started")
            for note in plan.notes:
                print(f"#   {note}")
            for warning in plan.warnings:
                print(f"#   WARNING: {warning}")
            print(f"#   env: " + " ".join(f"{k}={v}" for k, v in plan.env().items()))
            print(" ".join(plan.argv()))
        except ValueError as e:
            print(f"\n# ray head CANNOT be sized on this box: {e}")
        for entry in manifest["entries"]:
            print(f"\n# {entry['name']}\n" + " ".join(entry["cmd"]))
        return 0

    check_server(manifest["entries"], max_parallel, settings.get("server_max_num_seqs"))

    tpt = threads_per_task(manifest["entries"])
    head = RayHead()
    head.ensure(a.ray_head, tpt, max_parallel, dict(settings.get("ray") or {}))

    _write_manifest(sweep_dir, manifest)
    print(f"[sweep] {manifest['name']}: {len(manifest['entries'])} run(s) -> {sweep_dir} "
          f"(max_parallel={max_parallel}, stagger={stagger:g}s)")
    # supervise() exits the process on SIGINT/SIGTERM and deliberately leaves the runs alive, so the
    # head must survive that path — only tear it down when the queue actually drained.
    failed = supervise(sweep_dir, manifest, stagger, max_parallel, a.refresh, head.child_env())
    head.teardown()
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SweepError as e:
        print(f"[sweep] ERROR: {e}", file=sys.stderr)
        sys.exit(2)
