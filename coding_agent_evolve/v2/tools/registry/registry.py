#!/home/crv1pi/venvs/agent-eval/bin/python
"""registry -- per-run candidate-solution registry for coding-agent optimisation runs.

Two levels:
  * a PROBLEM is registered once (evaluation script, auxiliary files, direction, timeout,
    baseline) into $REGISTRY_HOME/problems/<name>/  (default ~/.local/share/registry)
  * a RUN FOLDER is bound to a problem with `registry init --problem NAME`, which copies the
    problem's files in and creates run/registry/registry.sqlite.

The agent then `submit`s candidates (its own number is stored as claimed_score; the `score`
column is written only by the problem's evaluator), and can `list`, `show`, `best`, `lineage`.
The registry owns copies of every artifact/program and maintains run/best.<ext> / run/best.py.

Stdlib only. `registry help` or `registry <cmd> -h` for usage.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as _dt
import hashlib
import json
import os
import random
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import time

SCHEMA_VERSION = 1
DEFAULT_TIMEOUT = 1100
DEFAULT_MAX_COPY_MB = 64
DEFAULT_SCORE_REGEX = r"^(?:SCORE|C5)\b[^=:\n]*[=:]\s*(-?(?:\d+\.?\d*(?:[eE][-+]?\d+)?|inf|nan))\b"
POLICIES = ("always", "on-claimed-best", "never")  # plus every-N
STATUSES = ("done", "failed", "timeout", "invalid")
KINDS = ("candidate", "note")
EVAL_STATUSES = ("ok", "invalid", "timeout", "error", "skipped")
REL_DB = os.path.join("run", "registry", "registry.sqlite")


# ----------------------------------------------------------------------------- helpers
def die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt_score(x) -> str:
    if x is None:
        return "-"
    return f"{x:.6f}"


def registry_home() -> str:
    return os.environ.get("REGISTRY_HOME") or os.path.expanduser("~/.local/share/registry")


def problems_dir() -> str:
    return os.path.join(registry_home(), "problems")


def find_root(explicit: str | None) -> str:
    """Walk up from cwd (or --root) to the directory containing run/registry/registry.sqlite."""
    if explicit:
        root = os.path.abspath(explicit)
        if not os.path.exists(os.path.join(root, REL_DB)):
            die(f"no registry at {root} (expected {REL_DB}); run `registry init` there first")
        return root
    d = os.getcwd()
    while True:
        if os.path.exists(os.path.join(d, REL_DB)):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            die("no registry found in this directory or any parent; run `registry init --problem NAME` "
                "in the run folder, or pass --root DIR")
        d = parent


def rel(root: str, path: str) -> str:
    return os.path.relpath(path, root)


# ----------------------------------------------------------------------------- database
def connect(root: str) -> sqlite3.Connection:
    conn = sqlite3.connect(os.path.join(root, REL_DB), timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 60000")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _is_transient(exc: Exception) -> bool:
    s = str(exc).lower()
    return "locked" in s or "busy" in s or "disk i/o" in s


def run_tx(conn: sqlite3.Connection, fn, retries: int = 8):
    """Run fn(conn) inside BEGIN IMMEDIATE; retry the whole thing on lock/busy errors."""
    delay = 0.2
    for attempt in range(retries):
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                out = fn(conn)
                conn.execute("COMMIT")
                return out
            except BaseException:
                with contextlib.suppress(Exception):
                    conn.execute("ROLLBACK")
                raise
        except sqlite3.OperationalError as exc:
            if attempt == retries - 1 or not _is_transient(exc):
                raise
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 3.0)


SCHEMA = """
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('done','failed','timeout','invalid')),
  kind TEXT NOT NULL DEFAULT 'candidate' CHECK (kind IN ('candidate','note')),
  description TEXT NOT NULL,
  tag TEXT,
  claimed_score REAL,
  score REAL,
  eval_status TEXT CHECK (eval_status IN ('ok','invalid','timeout','error','skipped')),
  eval_error TEXT,
  eval_runtime_s REAL,
  runtime_s REAL,
  config TEXT,
  artifact TEXT,
  artifact_sha256 TEXT,
  artifact_bytes INTEGER,
  program TEXT,
  program_sha256 TEXT,
  parents TEXT,
  sources TEXT,
  error TEXT
);
CREATE INDEX IF NOT EXISTS candidates_score ON candidates(score);
CREATE INDEX IF NOT EXISTS candidates_tag ON candidates(tag);
"""


def load_config(conn: sqlite3.Connection) -> dict:
    return {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM config")}


def set_config(conn: sqlite3.Connection, key: str, value) -> None:
    conn.execute("INSERT OR REPLACE INTO config (key, value) VALUES (?, ?)", (key, str(value)))


def is_max(cfg: dict) -> bool:
    return cfg["direction"] == "max"


def better(cfg: dict, a: float, b: float) -> bool:
    """True if score a strictly beats score b under the problem's direction."""
    return a > b if is_max(cfg) else a < b


def current_best(conn: sqlite3.Connection, cfg: dict):
    order = "DESC" if is_max(cfg) else "ASC"
    return conn.execute(
        f"SELECT * FROM candidates WHERE kind='candidate' AND score IS NOT NULL "
        f"ORDER BY score {order}, id ASC LIMIT 1").fetchone()


def get_row(conn: sqlite3.Connection, cid: int):
    row = conn.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
    if row is None:
        die(f"no candidate with id {cid}")
    return row


# ----------------------------------------------------------------------------- problems
def problem_path(name: str) -> str:
    return os.path.join(problems_dir(), name)


def load_problem(name: str) -> dict:
    p = os.path.join(problem_path(name), "problem.json")
    if not os.path.exists(p):
        have = sorted(os.listdir(problems_dir())) if os.path.isdir(problems_dir()) else []
        die(f"unknown problem '{name}'. Registered: {have or 'none'} "
            f"(store: {problems_dir()}; add one with `registry problem add`)")
    with open(p) as fh:
        return json.load(fh)


def cmd_problem_add(a) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.name):
        die("problem name may contain only letters, digits, _ . -")
    if "{artifact}" not in a.eval and "{program}" not in a.eval:
        die("--eval must contain {artifact} and/or {program} placeholders")
    dest = problem_path(a.name)
    if os.path.exists(dest) and not a.force:
        die(f"problem '{a.name}' already exists at {dest}; use --force to replace")
    for f in list(a.files) + ([a.baseline] if a.baseline else []):
        if not os.path.isfile(f):
            die(f"file not found: {f}")
    try:
        re.compile(a.score_regex)
    except re.error as exc:
        die(f"bad --score-regex: {exc}")
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.makedirs(dest)
    files = []
    for f in a.files:
        shutil.copy2(f, os.path.join(dest, os.path.basename(f)))
        files.append(os.path.basename(f))
    baseline = None
    if a.baseline:
        baseline = os.path.basename(a.baseline)
        shutil.copy2(a.baseline, os.path.join(dest, baseline))
    ext = a.artifact_ext or (os.path.splitext(baseline)[1] if baseline else "")
    spec = {
        "name": a.name, "direction": a.direction, "eval": a.eval, "files": files,
        "baseline": baseline, "timeout": a.timeout, "artifact_ext": ext,
        "score_regex": a.score_regex, "description": a.description or "",
        "created_at": now(),
    }
    with open(os.path.join(dest, "problem.json"), "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"registered problem {a.name} at {dest}")
    print(f"direction={a.direction} eval={a.eval!r} files={files} baseline={baseline} timeout={a.timeout}s")


def cmd_problem_list(a) -> None:
    d = problems_dir()
    names = sorted(n for n in os.listdir(d)) if os.path.isdir(d) else []
    if not names:
        print(f"no problems registered (store: {d})")
        return
    print(f"store={d}")
    for n in names:
        try:
            s = load_problem(n)
        except SystemExit:
            continue
        print(f"{n:<16} {s['direction']:<3} timeout={s['timeout']:<5} eval={s['eval']}")


def cmd_problem_show(a) -> None:
    s = load_problem(a.name)
    for k, v in s.items():
        print(f"{k}={json.dumps(v) if not isinstance(v, str) else v}")
    print(f"path={problem_path(a.name)}")


# ----------------------------------------------------------------------------- evaluator
def run_evaluator(root: str, cfg: dict, artifact_abs: str | None, program_abs: str | None, log_path: str) -> dict:
    """Run the problem's evaluator on the registry's copy. Never raises."""
    cmd = cfg["eval_cmd"]
    if "{artifact}" in cmd and not artifact_abs:
        return {"eval_status": "error", "score": None, "eval_error": "evaluator needs an artifact", "eval_runtime_s": 0.0}
    if "{program}" in cmd and not program_abs:
        return {"eval_status": "error", "score": None, "eval_error": "evaluator needs a program", "eval_runtime_s": 0.0}
    cmd = cmd.replace("{artifact}", artifact_abs or "").replace("{program}", program_abs or "")
    timeout = float(cfg["eval_timeout_s"])
    t0 = time.time()
    proc = subprocess.Popen(cmd, shell=True, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout)
        rc = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        with contextlib.suppress(Exception):
            os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        rc = None
        timed_out = True
    dt = time.time() - t0
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a") as fh:
        fh.write(f"=== {now()} cmd={cmd!r} rc={rc} elapsed={dt:.1f}s timeout={timed_out}\n")
        fh.write("--- stdout\n" + (out or "") + "\n--- stderr\n" + (err or "") + "\n")
    if timed_out:
        return {"eval_status": "timeout", "score": None, "eval_error": f"evaluator exceeded {timeout:.0f}s", "eval_runtime_s": dt}
    invalid_line = next((ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip().startswith("INVALID")), None)
    if rc != 0 or invalid_line:
        msg = invalid_line or (err.strip().splitlines()[-1] if err.strip() else f"exit {rc}")
        return {"eval_status": "invalid", "score": None, "eval_error": f"exit {rc}: {msg}"[:500], "eval_runtime_s": dt}
    matches = re.findall(cfg["score_regex"], out, flags=re.MULTILINE)
    if not matches:
        return {"eval_status": "error", "score": None, "eval_error": "no SCORE line in evaluator output", "eval_runtime_s": dt}
    val = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
    try:
        score = float(val)
    except ValueError:
        return {"eval_status": "error", "score": None, "eval_error": f"unparsable score {val!r}", "eval_runtime_s": dt}
    if score != score or score in (float("inf"), float("-inf")):
        return {"eval_status": "invalid", "score": None, "eval_error": f"evaluator returned {val}", "eval_runtime_s": dt}
    return {"eval_status": "ok", "score": score, "eval_error": None, "eval_runtime_s": dt}


def eval_log_path(root: str, cid: int) -> str:
    return os.path.join(root, "run", "registry", "eval_logs", f"{cid:06d}.log")


def store_path(root: str, sub: str, cid: int, ext: str) -> str:
    return os.path.join(root, "run", "registry", sub, f"{cid:06d}{ext}")


def promote(root: str, cfg: dict, row) -> str | None:
    """Copy the row's stored artifact/program to run/best.<ext> / run/best.py. Caller holds the tx."""
    if not row["artifact"]:
        return None
    src = os.path.join(root, row["artifact"])
    ext = os.path.splitext(src)[1]
    dst = os.path.join(root, "run", f"best{ext}")
    tmp = os.path.join(root, "run", ".best.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)
    if row["program"]:
        ptmp = os.path.join(root, "run", ".best.py.tmp")
        shutil.copy2(os.path.join(root, row["program"]), ptmp)
        os.replace(ptmp, os.path.join(root, "run", "best.py"))
    with open(os.path.join(root, "run", "registry", "best.json"), "w") as fh:
        json.dump({"id": row["id"], "score": row["score"], "sha256": row["artifact_sha256"],
                   "artifact": row["artifact"], "promoted_at": now()}, fh, indent=2)
    return rel(root, dst)


def apply_eval_result(conn: sqlite3.Connection, root: str, cfg: dict, cid: int, res: dict) -> dict:
    """Write an evaluator result to the row, recompute best, promote if needed. Runs inside a tx."""
    before = current_best(conn, cfg)
    status_sql = ", status='invalid'" if res["eval_status"] == "invalid" else ""
    conn.execute(
        f"UPDATE candidates SET score=?, eval_status=?, eval_error=?, eval_runtime_s=?{status_sql} WHERE id=?",
        (res["score"], res["eval_status"], res["eval_error"], res["eval_runtime_s"], cid))
    row = get_row(conn, cid)
    out = {"best_before": before, "new_best": False, "promoted": None, "delta": None}
    if row["score"] is not None:
        if before is not None and before["id"] != cid:
            out["delta"] = row["score"] - before["score"]
        if before is None or before["id"] == cid or better(cfg, row["score"], before["score"]):
            out["new_best"] = True
            out["promoted"] = promote(root, cfg, row)
    return out


# ----------------------------------------------------------------------------- init
def parse_policy(s: str) -> tuple[str, int]:
    if s in POLICIES:
        return s, 0
    m = re.fullmatch(r"every-(\d+)", s)
    if m and int(m.group(1)) > 0:
        return "every-N", int(m.group(1))
    die(f"bad --eval-policy {s!r}; choose always | on-claimed-best | every-N | never")


def probe_wal(db_path: str) -> bool:
    """Set WAL and verify a child process sees it and can read a row we wrote."""
    try:
        conn = sqlite3.connect(db_path, timeout=10, isolation_level=None)
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if mode.lower() != "wal":
            conn.close()
            return False
        conn.execute("CREATE TABLE IF NOT EXISTS _probe (v TEXT)")
        conn.execute("INSERT INTO _probe VALUES ('x')")
        code = ("import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); "
                "m=c.execute('PRAGMA journal_mode').fetchone()[0]; n=c.execute('SELECT count(*) FROM _probe').fetchone()[0]; "
                "print(m, n)")
        r = subprocess.run([sys.executable, "-c", code, db_path], capture_output=True, text=True, timeout=30)
        conn.execute("DROP TABLE _probe")
        conn.close()
        return r.returncode == 0 and r.stdout.split()[:2] == ["wal", "1"]
    except Exception:
        return False


def cmd_init(a) -> None:
    root = os.path.abspath(a.root or os.getcwd())
    db_path = os.path.join(root, REL_DB)
    if os.path.exists(db_path):
        die(f"registry already initialised at {root}")
    spec = load_problem(a.problem)
    policy, every_n = parse_policy(a.eval_policy)
    pdir = problem_path(a.problem)
    # copy the problem's files into the run folder (agents import eval.py from here)
    for f in spec["files"] + ([spec["baseline"]] if spec["baseline"] else []):
        src, dst = os.path.join(pdir, f), os.path.join(root, f)
        if os.path.exists(dst):
            if sha256_file(src) != sha256_file(dst) and not a.force:
                die(f"{dst} exists and differs from the problem's copy; move it away or pass --force")
        else:
            shutil.copy2(src, dst)
    os.makedirs(os.path.join(root, "run", "registry", "artifacts"), exist_ok=True)
    os.makedirs(os.path.join(root, "run", "registry", "programs"), exist_ok=True)
    os.makedirs(os.path.join(root, "run", "registry", "eval_logs"), exist_ok=True)
    journal = "delete"
    if a.journal != "delete" and probe_wal(db_path):
        journal = "wal"
    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    conn.execute(f"PRAGMA journal_mode={journal.upper()}")
    conn.close()
    conn = connect(root)
    conn.executescript(SCHEMA)
    cfg = {
        "schema_version": SCHEMA_VERSION, "tool_sha256": sha256_file(os.path.abspath(__file__)),
        "problem": a.problem, "direction": spec["direction"], "eval_cmd": spec["eval"],
        "eval_timeout_s": a.eval_timeout if a.eval_timeout is not None else spec["timeout"],
        "eval_policy": policy, "eval_every_n": every_n, "eval_counter": 0,
        "score_regex": spec["score_regex"], "artifact_ext": spec["artifact_ext"],
        "max_copy_mb": a.max_copy_mb, "journal_mode": journal, "created_at": now(),
        "problem_files": json.dumps(spec["files"]), "baseline_file": spec["baseline"] or "",
    }

    def write_cfg(c):
        for k, v in cfg.items():
            set_config(c, k, v)
    run_tx(conn, write_cfg)
    cfg = load_config(conn)
    print(f"initialised registry at {root}")
    print(f"problem={a.problem} direction={spec['direction']} eval={spec['eval']!r} "
          f"timeout={cfg['eval_timeout_s']}s policy={a.eval_policy} journal={journal}")
    if not spec["baseline"]:
        print("baseline=none (ids start at 1)")
        return
    # baseline row, id 0, always evaluated
    ext = spec["artifact_ext"]
    art_dst = store_path(root, "artifacts", 0, ext)
    shutil.copy2(os.path.join(root, spec["baseline"]), art_dst)
    sha, size = sha256_file(art_dst), os.path.getsize(art_dst)

    def ins(c):
        c.execute(
            "INSERT INTO candidates (id, created_at, status, kind, description, tag, artifact, artifact_sha256, "
            "artifact_bytes, sources, parents) VALUES (0, ?, 'done', 'candidate', ?, 'baseline', ?, ?, ?, 'supplied', '[]')",
            (now(), f"baseline construction ({spec['baseline']})", rel(root, art_dst), sha, size))
    run_tx(conn, ins)
    res = run_evaluator(root, cfg, art_dst, None, eval_log_path(root, 0))
    run_tx(conn, lambda c: apply_eval_result(c, root, cfg, 0, res))
    if res["eval_status"] == "ok":
        print(f"baseline id=0 score={res['score']!r} eval=ok ({res['eval_runtime_s']:.1f}s) -> run/best{ext}")
        if a.baseline_expect is not None:
            exp = a.baseline_expect
            if abs(res["score"] - exp) > a.baseline_tol * max(1.0, abs(exp)):
                die(f"baseline scores {res['score']!r} but --baseline-expect says {exp!r} "
                    f"(tolerance {a.baseline_tol:g} relative); fix the prompt/expectation before running agents", 3)
            print(f"baseline matches expectation {exp!r} (within {a.baseline_tol:g})")
    else:
        print(f"WARNING: baseline unverified: eval={res['eval_status']} ({res['eval_error']}); "
              f"deltas have no reference until `registry eval 0` succeeds", file=sys.stderr)


# ----------------------------------------------------------------------------- submit
def parse_config_arg(s: str | None) -> str | None:
    if s is None:
        return None
    if s.startswith("@"):
        with open(s[1:]) as fh:
            s = fh.read()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError as exc:
        die(f"--config is not valid JSON ({exc}); pass an object like '{{\"n\": 4096, \"seed\": 3}}' or @file.json")
    if not isinstance(obj, dict):
        die("--config must be a JSON object")
    return json.dumps(obj, sort_keys=True)


def parse_parents(conn: sqlite3.Connection, s: str | None) -> str:
    if not s:
        return "[]"
    ids = []
    for tok in re.split(r"[,\s]+", s.strip()):
        if not tok:
            continue
        if not tok.isdigit():
            die(f"--parents expects candidate ids, got {tok!r}")
        cid = int(tok)
        if conn.execute("SELECT 1 FROM candidates WHERE id=?", (cid,)).fetchone() is None:
            die(f"--parents: no candidate with id {cid}")
        ids.append(cid)
    return json.dumps(ids)


def copy_into_store(root: str, cfg: dict, cid: int, sub: str, src: str, ext_override: str | None, warnings: list) -> tuple[str, str, int]:
    """Copy src into run/registry/<sub>/. Returns (rel_path, sha256, bytes). Large artifacts are referenced, not copied."""
    src = os.path.abspath(src)
    sha, size = sha256_file(src), os.path.getsize(src)
    cap = float(cfg["max_copy_mb"]) * 1024 * 1024
    if sub == "artifacts" and size > cap:
        warnings.append(f"artifact not copied ({size / 1e6:.0f} MB > {cfg['max_copy_mb']} MB); do not overwrite {src}")
        return src, sha, size
    ext = ext_override if ext_override is not None else os.path.splitext(src)[1]
    dst = store_path(root, sub, cid, ext)
    shutil.copy2(src, dst)
    return rel(root, dst), sha, size


def emit(kv: list[tuple[str, object]], as_json: bool) -> None:
    if as_json:
        print(json.dumps({k: v for k, v in kv}, default=str))
    else:
        for k, v in kv:
            print(f"{k}={v}")


def cmd_submit(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    cfg = load_config(conn)
    kind = a.kind
    status = a.status
    if kind == "note":
        if a.claimed is not None or a.artifact or a.program:
            die("--kind note is for analyses/bounds/observations: it takes --desc (and --tag/--parents/--sources/--config) "
                "but no --claimed, --artifact or --program. Submit runnable candidates as kind=candidate.")
        status = "done"
    else:
        if status == "done" and not a.artifact:
            die("a done candidate needs --artifact PATH (the solution file the evaluator scores). "
                "Use --status failed|timeout for attempts that produced nothing.")
    for p in (a.artifact, a.program):
        if p and not os.path.isfile(p):
            die(f"file not found: {p}")
    config_json = parse_config_arg(a.config)
    parents_json = parse_parents(conn, a.parents)
    warnings: list[str] = []

    def insert(c):
        cur = c.execute(
            "INSERT INTO candidates (created_at, status, kind, description, tag, claimed_score, runtime_s, config, "
            "parents, sources, error) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (now(), status, kind, a.desc, a.tag, a.claimed, a.runtime, config_json, parents_json, a.sources, a.error))
        return cur.lastrowid
    cid = run_tx(conn, insert)

    art_rel = art_sha = prog_rel = prog_sha = None
    art_bytes = None
    if a.artifact:
        art_rel, art_sha, art_bytes = copy_into_store(root, cfg, cid, "artifacts", a.artifact, cfg["artifact_ext"] or None, warnings)
    if a.program:
        prog_rel, prog_sha, _ = copy_into_store(root, cfg, cid, "programs", a.program, ".py", warnings)

    def attach(c):
        c.execute("UPDATE candidates SET artifact=?, artifact_sha256=?, artifact_bytes=?, program=?, program_sha256=? WHERE id=?",
                  (art_rel, art_sha, art_bytes, prog_rel, prog_sha, cid))
        dup = None
        if art_sha:
            dup = c.execute("SELECT id FROM candidates WHERE artifact_sha256=? AND id<>? ORDER BY id LIMIT 1", (art_sha, cid)).fetchone()
        if dup is None and prog_sha:
            dup = c.execute("SELECT id FROM candidates WHERE program_sha256=? AND id<>? ORDER BY id LIMIT 1", (prog_sha, cid)).fetchone()
        # decide the evaluation policy under the lock
        best = current_best(c, cfg)
        decision = "skip"
        reason = ""
        if kind == "candidate" and status == "done" and not a.no_eval:
            pol = cfg["eval_policy"]
            if pol == "always":
                decision = "eval"
            elif pol == "never":
                reason = "policy=never"
            else:  # on-claimed-best or every-N
                if a.claimed is not None and (best is None or better(cfg, a.claimed, best["score"])):
                    decision = "eval"
                elif a.claimed is None:
                    reason = f"policy={pol}, no --claimed score"
                else:
                    reason = f"policy={pol}, claimed {a.claimed!r} does not beat best {best['score']!r}"
                if pol == "every-N":
                    n = int(cfg.get("eval_counter", 0)) + 1
                    set_config(c, "eval_counter", n)
                    if decision != "eval" and n % int(cfg["eval_every_n"]) == 0:
                        decision, reason = "eval", ""
        elif a.no_eval:
            reason = "--no-eval"
        if decision != "eval" and kind == "candidate" and status == "done":
            c.execute("UPDATE candidates SET eval_status='skipped' WHERE id=?", (cid,))
        return dup["id"] if dup else None, decision, reason, best
    dup_id, decision, reason, best_before = run_tx(conn, attach)

    res = None
    outcome = {"best_before": best_before, "new_best": False, "promoted": None, "delta": None}
    if decision == "eval":
        res = run_evaluator(root, cfg, os.path.join(root, art_rel) if art_rel else None,
                            os.path.join(root, prog_rel) if prog_rel else None, eval_log_path(root, cid))
        outcome = run_tx(conn, lambda c: apply_eval_result(c, root, cfg, cid, res))
    row = get_row(conn, cid)
    best_after = current_best(conn, cfg)

    kv: list[tuple[str, object]] = [("id", cid), ("status", row["status"]), ("kind", kind)]
    if kind == "candidate":
        kv.append(("claimed_score", "null" if a.claimed is None else repr(a.claimed)))
        kv.append(("score", "null" if row["score"] is None else repr(row["score"])))
        if res is None:
            kv.append(("eval", f"skipped ({reason})" if reason else "skipped"))
        elif res["eval_status"] == "ok":
            kv.append(("eval", f"ok ({res['eval_runtime_s']:.1f}s)"))
        else:
            kv.append(("eval", f"{res['eval_status']} ({res['eval_error']})"))
        if best_after is not None:
            kv.append(("best", f"{best_after['score']!r} (id={best_after['id']})"))
        else:
            kv.append(("best", "none yet"))
        if outcome["delta"] is not None:
            kv.append(("delta", f"{outcome['delta']:+.6g}"))
        kv.append(("new_best", "yes" if outcome["new_best"] else "no"))
        if outcome["promoted"]:
            kv.append(("promoted", outcome["promoted"]))
        if a.claimed is not None and row["score"] is not None:
            diff = abs(a.claimed - row["score"])
            if diff > 1e-6 * max(1.0, abs(row["score"])):
                kv.append(("warning", f"claimed differs from verified by {diff:.1e}"))
        if dup_id is not None:
            kv.append(("duplicate_of", dup_id))
        for w in warnings:
            kv.append(("warning", w))
        if art_rel:
            kv.append(("artifact", art_rel))
    emit(kv, a.json)


# ----------------------------------------------------------------------------- read commands
def row_dict(r) -> dict:
    d = dict(r)
    for k in ("config", "parents"):
        if d.get(k):
            with contextlib.suppress(Exception):
                d[k] = json.loads(d[k])
    return d


def header_line(conn: sqlite3.Connection, cfg: dict) -> str:
    best = current_best(conn, cfg)
    n = conn.execute("SELECT count(*) FROM candidates").fetchone()[0]
    unv = conn.execute("SELECT count(*) FROM candidates WHERE kind='candidate' AND status='done' AND score IS NULL").fetchone()[0]
    b = f"{best['score']!r} (id={best['id']})" if best else "none"
    return f"problem={cfg['problem']} direction={cfg['direction']} best={b} rows={n} unverified={unv}"


def cmd_list(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    cfg = load_config(conn)
    where, params = [], []
    if a.tag:
        where.append("tag=?"); params.append(a.tag)
    if a.status:
        where.append("status=?"); params.append(a.status)
    if a.grep:
        like = f"%{a.grep}%"
        where.append("(description LIKE ? OR tag LIKE ? OR config LIKE ? OR sources LIKE ? OR error LIKE ?)")
        params += [like] * 5
    sql = "SELECT * FROM candidates"
    if where:
        sql += " WHERE " + " AND ".join(where)
    if a.sort == "score":
        sql += f" ORDER BY score IS NULL, score {'DESC' if is_max(cfg) else 'ASC'}, id"
    else:
        sql += " ORDER BY id DESC"
    if not a.all:
        sql += f" LIMIT {int(a.last)}"
    rows = conn.execute(sql, params).fetchall()
    if a.sort == "id":
        rows = list(reversed(rows))
    if a.json:
        print(json.dumps([row_dict(r) for r in rows], default=str))
        return
    best = current_best(conn, cfg)
    best_id = best["id"] if best else None
    print(header_line(conn, cfg))
    print(f"{'id':>4} {'status':<8} {'score':<11} {'claimed':<10} {'dbest':<10} {'tag':<10} {'runtime':<8} desc")
    for r in rows:
        if r["kind"] == "note":
            sc, cl, db = "note", "-", "-"
        else:
            if r["score"] is None:
                sc = "?" if r["status"] == "done" else "-"
            else:
                sc = fmt_score(r["score"]) + ("*" if r["id"] == best_id else "")
            cl = fmt_score(r["claimed_score"])
            db = f"{r['score'] - best['score']:+.6f}" if (r["score"] is not None and best) else "-"
            if r["claimed_score"] is not None and r["score"] is not None and \
                    abs(r["claimed_score"] - r["score"]) > 1e-6 * max(1.0, abs(r["score"])):
                cl += "!"
        rt = f"{r['runtime_s']:.0f}s" if r["runtime_s"] is not None else "-"
        desc = (r["description"] or "").replace("\n", " ")
        if len(desc) > 60:
            desc = desc[:57] + "..."
        print(f"{r['id']:>4} {r['status']:<8} {sc:<11} {cl:<10} {db:<10} {(r['tag'] or '-')[:10]:<10} {rt:<8} {desc}")
    if rows and not a.all:
        print(f"(showing {len(rows)}; --all for everything, --grep TEXT to search)")


def cmd_show(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    r = get_row(conn, a.id)
    d = row_dict(r)
    if d.get("artifact"):
        p = os.path.join(root, d["artifact"])
        if os.path.exists(p):
            d["artifact_drift"] = "yes" if sha256_file(p) != d["artifact_sha256"] else "no"
        else:
            d["artifact_drift"] = "missing"
    if a.json:
        print(json.dumps(d, default=str))
        return
    for k, v in d.items():
        if v is None or v == "" or v == []:
            continue
        print(f"{k}={json.dumps(v) if isinstance(v, (dict, list)) else v}")
    lp = eval_log_path(root, a.id)
    if os.path.exists(lp):
        print(f"eval_log={rel(root, lp)}")


def cmd_best(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    cfg = load_config(conn)
    best = current_best(conn, cfg)
    if best is None:
        print("no verified candidate yet")
        return
    d = row_dict(best)
    if a.json:
        print(json.dumps(d, default=str))
        return
    print(f"id={d['id']} score={d['score']!r} claimed={d['claimed_score']!r} tag={d['tag']} "
          f"runtime_s={d['runtime_s']} created={d['created_at']}")
    ext = os.path.splitext(d["artifact"] or "")[1]
    print(f"artifact={d['artifact']} sha256={(d['artifact_sha256'] or '')[:12]}.. bytes={d['artifact_bytes']} -> run/best{ext}")
    if d.get("program"):
        print(f"program={d['program']} -> run/best.py")
    print(f"parents={json.dumps(d['parents'])} sources={d['sources'] or ''}")
    print(f"desc={d['description']}")
    if d.get("config"):
        print(f"config={json.dumps(d['config'])}")


def cmd_lineage(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    seen = set()

    def walk(cid: int, depth: int):
        if depth > 50 or cid in seen:
            return
        seen.add(cid)
        r = conn.execute("SELECT * FROM candidates WHERE id=?", (cid,)).fetchone()
        if r is None:
            print("  " * depth + f"#{cid} (missing)")
            return
        sc = fmt_score(r["score"]) if r["kind"] == "candidate" else "note"
        cl = f" claimed={fmt_score(r['claimed_score'])}" if r["claimed_score"] is not None else ""
        print("  " * depth + f"#{r['id']} {r['status']} score={sc}{cl} tag={r['tag'] or '-'} :: {r['description']}")
        for p in json.loads(r["parents"] or "[]"):
            walk(p, depth + 1)
    get_row(conn, a.id)
    walk(a.id, 0)
    kids = conn.execute("SELECT id, description, score FROM candidates WHERE parents LIKE ? ORDER BY id",
                        (f"%{a.id}%",)).fetchall()
    kids = [k for k in kids if a.id in json.loads(conn.execute("SELECT parents FROM candidates WHERE id=?", (k["id"],)).fetchone()[0] or "[]")]
    if kids:
        print(f"children of #{a.id}: " + ", ".join(f"#{k['id']}({fmt_score(k['score'])})" for k in kids))


def cmd_eval(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    cfg = load_config(conn)
    if a.id is None and not a.pending:
        die("give a candidate ID or --pending")
    if a.pending:
        order = "DESC" if is_max(cfg) else "ASC"
        rows = conn.execute(
            "SELECT id FROM candidates WHERE kind='candidate' AND status='done' AND artifact IS NOT NULL AND "
            "(eval_status IS NULL OR eval_status IN ('skipped','timeout','error')) "
            f"ORDER BY claimed_score IS NULL, claimed_score {order}, id LIMIT ?", (a.limit,)).fetchall()
        ids = [r["id"] for r in rows]
        if not ids:
            print("nothing pending")
            return
    else:
        ids = [a.id]
    for cid in ids:
        r = get_row(conn, cid)
        if r["kind"] != "candidate":
            die(f"#{cid} is a note; nothing to evaluate")
        if not r["artifact"] and "{artifact}" in cfg["eval_cmd"]:
            print(f"id={cid} eval=error (no artifact stored)")
            continue
        old = r["score"]
        if a.timeout:
            cfg = dict(cfg, eval_timeout_s=a.timeout)
        res = run_evaluator(root, cfg, os.path.join(root, r["artifact"]) if r["artifact"] else None,
                            os.path.join(root, r["program"]) if r["program"] else None, eval_log_path(root, cid))
        if res["eval_status"] == "invalid" and r["status"] != "done":
            pass
        out = run_tx(conn, lambda c: apply_eval_result(c, root, cfg, cid, res))
        line = f"id={cid} eval={res['eval_status']}"
        if res["eval_status"] == "ok":
            line += f" score={res['score']!r} ({res['eval_runtime_s']:.1f}s)"
            if old is not None and abs(old - res["score"]) > 1e-9:
                line += f" previous={old!r}"
        else:
            line += f" ({res['eval_error']})"
        if out["new_best"]:
            line += f" new_best=yes promoted={out['promoted']}"
        print(line)


def cmd_export(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    rows = conn.execute("SELECT * FROM candidates ORDER BY id").fetchall()
    out = open(a.path, "w") if a.path else sys.stdout
    try:
        if a.csv:
            w = csv.writer(out)
            if rows:
                w.writerow(rows[0].keys())
                for r in rows:
                    w.writerow([r[k] for k in r.keys()])
        else:
            for r in rows:
                out.write(json.dumps(row_dict(r), default=str) + "\n")
    finally:
        if a.path:
            out.close()
            print(f"wrote {len(rows)} rows to {a.path}")


def cmd_config(a) -> None:
    root = find_root(a.root)
    conn = connect(root)
    print(f"root={root}")
    for k, v in sorted(load_config(conn).items()):
        print(f"{k}={v}")


# ----------------------------------------------------------------------------- argparse
HELP_EPILOG = """
typical use inside a run folder:
  registry best                                   # what is the verified best so far
  registry list --last 20                         # recent attempts (chronological)
  registry list --grep gauss                      # did I already try this idea
  registry submit --desc "coarse-to-fine, n=4096" --artifact run/attempts/c2f.npy \\
      --program run/attempts/c2f.py --claimed 0.8731 --runtime 412 \\
      --config '{"n":4096,"seed":3}' --parents 12 --tag c2f
  registry submit --kind note --desc "flat-top bound: C<=0.9529 for q=8" --tag bounds
  registry show 17 ; registry lineage 17

`score` is written only by the problem's evaluator; your own number is `claimed_score`.
run/best.<ext> and run/best.py are maintained by the registry -- never write them yourself.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="registry", description=__doc__.split("\n\n")[0],
                                epilog=HELP_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    def add_root(sp):
        sp.add_argument("--root", help="run folder (default: search upward from cwd)")

    # problem
    pp = sub.add_parser("problem", help="register / inspect problem definitions (operator)")
    psub = pp.add_subparsers(dest="pcmd", metavar="SUBCOMMAND")
    pa = psub.add_parser("add", help="register a problem: evaluator + files + direction + baseline")
    pa.add_argument("name")
    pa.add_argument("--direction", required=True, choices=("max", "min"))
    pa.add_argument("--eval", required=True, help='evaluator command, e.g. "python eval.py {artifact}"; {program} also allowed')
    pa.add_argument("--files", nargs="+", required=True, help="evaluation script and auxiliary files copied into every run folder")
    pa.add_argument("--baseline", help="starting construction; becomes candidate id 0 at init")
    pa.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help=f"evaluator timeout in seconds (default {DEFAULT_TIMEOUT})")
    pa.add_argument("--artifact-ext", help="extension for stored artifacts (default: the baseline's)")
    pa.add_argument("--score-regex", default=DEFAULT_SCORE_REGEX, help="regex with one group capturing the score in evaluator stdout")
    pa.add_argument("--description", help="one line about the problem")
    pa.add_argument("--force", action="store_true", help="replace an existing definition")
    pa.set_defaults(func=cmd_problem_add)
    pl = psub.add_parser("list", help="list registered problems")
    pl.set_defaults(func=cmd_problem_list)
    ps = psub.add_parser("show", help="show one problem definition")
    ps.add_argument("name")
    ps.set_defaults(func=cmd_problem_show)

    # init
    pi = sub.add_parser("init", help="bind this run folder to a problem and create the table (operator)")
    pi.add_argument("--problem", required=True)
    pi.add_argument("--eval-policy", default="on-claimed-best",
                    help="always | on-claimed-best (default) | every-N | never -- when the evaluator runs at submit")
    pi.add_argument("--eval-timeout", type=int, help="override the problem's evaluator timeout (s)")
    pi.add_argument("--max-copy-mb", type=float, default=DEFAULT_MAX_COPY_MB, help="artifacts above this are referenced, not copied")
    pi.add_argument("--baseline-expect", type=float, help="fail init if the baseline does not score this")
    pi.add_argument("--baseline-tol", type=float, default=1e-3, help="relative tolerance for --baseline-expect (default 1e-3)")
    pi.add_argument("--force", action="store_true", help="tolerate differing copies of the problem files in the run folder")
    pi.add_argument("--journal", choices=("auto", "delete"), default="auto", help=argparse.SUPPRESS)
    add_root(pi)
    pi.set_defaults(func=cmd_init)

    # submit
    psb = sub.add_parser("submit", help="register a candidate (or a note)", formatter_class=argparse.RawDescriptionHelpFormatter)
    psb.add_argument("--desc", required=True, help="one line: what the candidate is / the idea")
    psb.add_argument("--artifact", help="solution file the evaluator scores (.npy etc.); required for done candidates")
    psb.add_argument("--program", help="the script that produced it (copied for reproducibility)")
    psb.add_argument("--claimed", type=float, help="the score you measured yourself (stored as claimed_score)")
    psb.add_argument("--runtime", type=float, help="how long the candidate took to produce, seconds")
    psb.add_argument("--config", help="free-form JSON object with the candidate's configuration, or @file.json")
    psb.add_argument("--parents", help="ids of the candidates this one builds on, e.g. 3,7")
    psb.add_argument("--sources", help="free text: paper, url, idea origin")
    psb.add_argument("--tag", help="approach family / short label for grouping")
    psb.add_argument("--status", choices=STATUSES[:3], default="done", help="done (default) | failed | timeout")
    psb.add_argument("--error", help="for failed/timeout: what happened")
    psb.add_argument("--kind", choices=KINDS, default="candidate", help="candidate (default) | note (analysis/bound, no score)")
    psb.add_argument("--no-eval", action="store_true", help="skip the evaluator for this submission")
    psb.add_argument("--json", action="store_true")
    add_root(psb)
    psb.set_defaults(func=cmd_submit)

    # list
    pls = sub.add_parser("list", help="table of candidates")
    pls.add_argument("--last", type=int, default=20)
    pls.add_argument("--all", action="store_true")
    pls.add_argument("--tag")
    pls.add_argument("--status", choices=STATUSES)
    pls.add_argument("--grep", help="substring search over desc/tag/config/sources/error")
    pls.add_argument("--sort", choices=("id", "score"), default="id")
    pls.add_argument("--json", action="store_true")
    add_root(pls)
    pls.set_defaults(func=cmd_list)

    psh = sub.add_parser("show", help="all fields of one candidate")
    psh.add_argument("id", type=int)
    psh.add_argument("--json", action="store_true")
    add_root(psh)
    psh.set_defaults(func=cmd_show)

    pb = sub.add_parser("best", help="the verified best candidate")
    pb.add_argument("--json", action="store_true")
    add_root(pb)
    pb.set_defaults(func=cmd_best)

    pln = sub.add_parser("lineage", help="ancestors (via --parents) and children of a candidate")
    pln.add_argument("id", type=int)
    add_root(pln)
    pln.set_defaults(func=cmd_lineage)

    pe = sub.add_parser("eval", help="run the evaluator on a candidate (or sweep unverified ones)")
    pe.add_argument("id", type=int, nargs="?")
    pe.add_argument("--pending", action="store_true", help="evaluate all done candidates without a verified score")
    pe.add_argument("--limit", type=int, default=1000)
    pe.add_argument("--timeout", type=int, help="override evaluator timeout (s) for this sweep")
    add_root(pe)
    pe.set_defaults(func=cmd_eval)

    px = sub.add_parser("export", help="dump all rows as JSONL (default) or CSV")
    px.add_argument("path", nargs="?")
    px.add_argument("--csv", action="store_true")
    px.add_argument("--jsonl", action="store_true")
    add_root(px)
    px.set_defaults(func=cmd_export)

    pc = sub.add_parser("config", help="show this run's registry configuration")
    add_root(pc)
    pc.set_defaults(func=cmd_config)

    ph = sub.add_parser("help", help="show this help")
    ph.set_defaults(func=lambda a: p.print_help())
    return p


def main(argv=None) -> None:
    with contextlib.suppress(Exception):
        sys.stdout.reconfigure(line_buffering=True)
    p = build_parser()
    a = p.parse_args(argv)
    if a.cmd is None or (a.cmd == "problem" and getattr(a, "pcmd", None) is None):
        p.print_help()
        sys.exit(0 if a.cmd is None else 2)
    a.func(a)


if __name__ == "__main__":
    main()
