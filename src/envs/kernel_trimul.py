"""TriMul kernel optimisation -- TTT-Discover's ``gpu_mode`` task, graded on a local GPU.

Port of ``discover/examples/gpu_mode/env.py`` (``GpuModeEnv`` / ``GpuModeRewardEvaluator``) with one
substitution: TTT-Discover grades through a Modal RPC, we grade on a local card via
``coding_agent_evolve/gpumode/evaluate.py``. The prompt body, the initial state and the reward formula
are unchanged -- see :mod:`envs.kernel_prompt` for the verbatim prompt text.

WHAT "THE SAME EVALUATION" MEANS HERE
-------------------------------------
The score that guided their search is reproduced step for step, not just in formula:

  ==========================  ==============================================  ====================
  step                        TTT-Discover                                    here
  ==========================  ==============================================  ====================
  reject non-triton code      ``env.py:123``                                  same, before the eval
  reject identity kernels     ``env.py:125`` (trimul only)                    same
  grading mode                ``mode="leaderboard"`` (``env.py:132``)         same (see below)
  correctness gate            18 ``tests:`` shapes, must all pass             same
  timed shapes                7 ``benchmarks:`` shapes, ``recheck=True``      same
  reps per shape              3..100, stop at rel.err < 0.1 % / 30 s / 120 s  same (their eval.py)
  score                       geometric mean of the 7 means, in us            same
  reward                      ``1500 / score_us``                             same
  evals per candidate         exactly ONE, no averaging                       same
  ==========================  ==============================================  ====================

``leaderboard`` is not one run but two, and that is the whole point of it: ``run_eval.run_evaluation``
(libkernelbot, line 807-821) runs the ``test`` phase first and only proceeds to the timed phase if
every shape passed, and the timed phase then re-generates input and re-checks correctness on EVERY
rep. A kernel cannot buy speed with a wrong answer. Bare ``benchmark`` mode -- which this module used
until it was made faithful -- skips the 18-shape gate entirely and checks correctness once per shape,
so it scores kernels their search would have thrown away.

What remains different is the machine, not the procedure: they timed on a Modal H100, we time on a
local card. That is why ``trimul_a100`` exists and why scores never pool across architectures.

TWO PROBLEMS, ONE TASK: ``trimul_a100`` AND ``trimul_h100``
-----------------------------------------------------------
The only difference is the single rules line naming the target card. It matters more than it looks:
the prompt drives the model's block-size choices, and a config that is legal on an H100 (228 KB of
shared memory) dies on an A100 (166 KB) with ``OutOfResources`` -- observed in the very first smoke
run. So ``trimul_h100`` keeps TTT-Discover's line verbatim (the faithful baseline) and
``trimul_a100`` names the A100 and its shared-memory ceiling instead. Everything else -- prompt body,
target, reward scale, grader, task file -- is identical, so scores are comparable within an
architecture and NOT across one.

THREE WAYS THIS DIFFERS FROM THE MATH ENVS
------------------------------------------
1. **Grading is serial, on one GPU.** ``uses_sandbox = False`` keeps Ray out of it (as
   ``envs.toy_ee`` does) and :func:`_gpu_guard` serialises the eval. The children of a group are
   graded CONCURRENTLY through ``base.SAFE_GRADE_EXECUTOR``, so without that guard N evals would land
   on the same card at once and every timing would be meaningless -- ``eval.py``'s convergence rule
   (relative error < 0.1 %) cannot be met on a contended GPU, so each benchmark would also burn its
   whole rep budget.

2. **The guard is cross-process, not just cross-thread.** A sweep runs each of its runs as a separate
   PROCESS, so a ``threading.Lock`` alone would serialise within a run and let two runs trash each
   other's measurements with nothing in the logs to show it. The flock below is what makes
   ``max_parallel > 1`` safe. It is keyed by GPU index, so runs pinned to different cards do not
   block each other.

3. **The grader runs under a different interpreter.** The harness needs torch 2.7.1 / triton 3.3.1
   (``/scratch/vicstorage/learning_evolve/.venv`` on guadiana); this package does not. So we shell
   out and never import triton in-process. ``evaluate.py`` resolves its task dir from its own
   ``__file__``, so this process's working directory is irrelevant.

Measured on guadiana's idle A100 80GB PCIe (2026-08-10) with TTT-Discover's published kernel:
``--mode leaderboard`` takes **36 s** wall (13.2 s test + 14.8 s timed + ~8 s startup/compile) and
scores **2412 us**; the same kernel under the old ``--mode benchmark`` took ~11-23 s and scored
2467 us. So fidelity costs about 2.5x, and the two modes do NOT agree to within their own noise --
another reason not to mix scores from the two. A candidate that fails to compile still fails in ~5 s.
See ``gpumode_local/reference/README.md`` for the reference numbers to expect on each card.

CONFIGURATION -- PREFER THE SWEEP FILE
--------------------------------------
Where and how to grade is a property of the MACHINE, not of the experiment, but it still belongs in
the sweep file: keys there are validated against run_icl.py's parser, appear in ``--print-cmds``, and
land in the run's ``config.json`` -- so a finished run records the interpreter and the card that
produced its timings, which an environment variable does not. Precedence is flag, then env var,
then default.

======================  =========================  ============================  ==================
sweep key / CLI flag    environment fallback       meaning                       default
======================  =========================  ============================  ==================
``trimul-eval-python``  ``TRIMUL_EVAL_PYTHON``     interpreter for the harness    the scratch venv
``trimul-eval-gpu``     ``TRIMUL_EVAL_GPU``        card index, or ``inherit``     per problem (below)
``trimul-evaluate-py``  ``TRIMUL_EVALUATE_PY``     path to ``evaluate.py``        repo copy
``trimul-eval-mode``    ``TRIMUL_EVAL_MODE``       test|benchmark|leaderboard     ``leaderboard``
(env only)              ``TRIMUL_LOCK_DIR``        where the flock files live     ``/tmp``
======================  =========================  ============================  ==================

``trimul-eval-gpu`` accepts a card index OR the word ``inherit`` (leave ``CUDA_VISIBLE_DEVICES``
untouched, grade on whatever the environment grants). ``inherit`` is the ONLY correct setting inside
a scheduler allocation -- an LSF job on Bosch holds one GPU and LSF says which via
``CUDA_VISIBLE_DEVICES``, so naming an index there overrides the assignment and can land the eval on
another job's card. It is the default for ``trimul_h100``. Name an index only on an unscheduled box
like guadiana, where it is how you keep grading off the vLLM server's card.

``TRIMUL_LOCK_DIR`` is env-only on purpose: it must be identical for every process sharing a card, so
pinning it per-run in a sweep file would be a way to defeat the guard rather than configure it. Set it
once, in the job script. **Keep it on node-local storage** (the ``/tmp`` default is right): a GPU
belongs to exactly one host, so every process that can contend for it runs on that host, and ``flock``
over NFS is not dependable. A shared-filesystem lock dir buys nothing here and can silently fail to
exclude.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from envs.base import Environment, objective_only_prompt
from envs.kernel_prompt import TRIMUL_PROMPT
from puct import State
from sandbox.base_reward_evaluator import BaseRewardEvaluator

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATE_PY = REPO_ROOT / "coding_agent_evolve" / "gpumode" / "evaluate.py"
DEFAULT_EVAL_PYTHON = "/scratch/vicstorage/learning_evolve/.venv/bin/python"

# Reward scale and prompt target, straight from TTT-Discover (env.py:106-108, 191). The target is
# aspirational and deliberately left at the upstream value on BOTH cards: the best published kernel
# does 2198 us on an A100 and 1161 us on an H100, so 1000 us is out of reach either way.
SCORE_SCALE = 1500.0
TARGET_US = 1000

# THE GRADING MODE TTT-DISCOVER'S SEARCH ACTUALLY RAN ON. env.py:129-137 submits every candidate with
# ``mode="leaderboard"``, and libkernelbot's run_eval.run_evaluation (line 807-821) expands that into
# TWO runner calls: the 18 ``tests:`` shapes first, then -- only if they all pass -- the 7
# ``benchmarks:`` shapes with ``recheck=True``. Our evaluate.py:325-326 expands it identically, so the
# number that reaches the buffer is produced the same way theirs was. Measured cost of the extra
# fidelity on an idle A100: 36 s vs ~15 s for bare ``benchmark`` mode -- about 2.5x, not the ~100x an
# earlier version of this file and the reference README both guessed. Nothing rides on that guess now.
DEFAULT_EVAL_MODE = "leaderboard"

# Two submission gates, applied before the GPU is touched, from env.py:122-126. They are part of the
# reward function upstream, not a lint: a kernel that never calls into triton, or one that returns its
# input, can post a fast "score" that means nothing. Rejecting them here also saves the ~36 s eval.
#
# Both are blunt substring tests, and that bluntness is INHERITED, not an oversight -- upstream's
# banned-word check rejects any candidate containing "identity" anywhere, comments and variable names
# included. Tightening it (say, to a regex on the return statement) would accept candidates their
# search rejected, which is the one thing this module is not allowed to do. Leave it.
_REQUIRED_TOKEN = "@triton.jit"
_BANNED_TOKEN = "identity"

# "inherit" (or simply not naming a card at all) means: do NOT touch CUDA_VISIBLE_DEVICES, use
# whatever the environment already grants. This is the ONLY safe setting under a scheduler: an LSF
# job on Bosch holds exactly one of the node's GPUs and LSF communicates which one via
# CUDA_VISIBLE_DEVICES -- naming an index here would override that and can re-point grading at a
# card that belongs to someone else's job.
INHERIT_GPU = "inherit"

# Which card each problem grades on unless TRIMUL_EVAL_GPU / the sweep key says otherwise.
# trimul_a100 defaults to guadiana's card 1 (card 0 holds the vLLM server; nothing schedules GPUs
# there, so naming an index is both safe and necessary). trimul_h100 runs on Bosch under LSF, so its
# default is to inherit the scheduler's assignment.
_DEFAULT_GPU = {"trimul_a100": "1", "trimul_h100": INHERIT_GPU}

# Serialises access to the grading GPU WITHIN this process. Module-level on purpose: every
# environment instance here contends for the same card, so a per-instance lock serialises nothing.
_GPU_LOCK = threading.Lock()


def _gpu_key(gpu: str) -> str:
    """The per-card key the lock file and the dead-GPU marker are both named by."""
    if gpu == INHERIT_GPU:
        # No index of our own to key on -- key on what the scheduler granted this process. Processes
        # in the same allocation see the same value and correctly share one lock; under LSF's
        # one-GPU-per-job model there is no cross-job contention for the flock to referee anyway.
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "inherit").replace("/", "_").replace(",", "+")
    return gpu


@contextlib.contextmanager
def _gpu_guard(gpu: str):
    """Hold exclusive use of one GPU across every thread AND every process on this machine.

    Two layers, because there are two ways to lose exclusivity:
      * threads inside one run  -> the module-level :data:`_GPU_LOCK`;
      * separate runs of a sweep -> an ``flock`` on a per-GPU file.

    ``flock`` is released automatically if the holder dies, so a killed run cannot wedge the card --
    which is the property a lock file with a pid in it would not give us.
    """
    lock_dir = Path(os.environ.get("TRIMUL_LOCK_DIR", tempfile.gettempdir()))
    lock_path = lock_dir / f"learning_evolve-trimul-gpu{_gpu_key(gpu)}.lock"
    with _GPU_LOCK:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o666)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# On Exclusive_Process cards (every LSF GPU here) one leaked CUDA context bricks the card for
# everyone after it: each later eval dies at context creation with "device busy or unavailable"
# in ~7 s, which reads exactly like 500 bad kernels in a row. Verified on rng-dl01-w093
# 2026-08-14: a candidate hung at 14:10 (Xid 31), its pool worker outlived evaluate.py with the
# context held, and three sub-runs graded 0/~510 until the wedged kernel MMU-faulted at 16:29.
_INFRA_SIGNATURES = (
    "CUDA-capable device(s) is/are busy or unavailable",
    "all CUDA-capable devices are busy or unavailable",
    "CUDA error: initialization error",
    "CUDA driver initialization failed",
)

# How long after SIGKILL to keep polling for the eval process group to disappear. A kernel wedged
# on the GPU can sit unkillable in D-state until the driver faults it out (2h19m in the incident
# above), so this is a heartbeat interval for logging, not a promise.
_REAP_GRACE_SECONDS = 60.0
_UNKILLABLE_WAIT_SECONDS = 1800.0


def _reap_group(pgid: int, grace: float = _REAP_GRACE_SECONDS) -> bool:
    """SIGKILL an eval's entire process group and wait until it is actually gone.

    ``subprocess``'s own timeout handling kills the DIRECT child only -- but the CUDA context
    lives two forks down (evaluate.py -> eval.py -> multiprocessing spawn worker), so anything
    short of the whole group leaves an orphan holding the card. Killing the group on the SUCCESS
    path too is deliberate: the incident orphan was created by a normal-looking evaluate.py exit,
    not by a timeout.
    """
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        return True                          # nothing left of the group
    except PermissionError:                  # a pid was recycled into someone else's group
        logger.warning("reap: pgid %d no longer ours; leaving it alone", pgid)
        return True
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.25)
    return False                             # unkillable: kernel still resident on the card


# When an eval group survives SIGKILL past _UNKILLABLE_WAIT_SECONDS, the card is not merely busy --
# the driver itself is wedged (observed twice: rng-dl01-w093 GPU7 and w092, both spinning in
# uvm_va_space_destroy with SIGKILL pending for 14h+). Every process that subsequently touches the
# card goes straight to D-state at CUDA init, so continuing costs a full
# timeout+grace+unkillable-wait cycle (~56 min) AND leaks one more unkillable process PER CANDIDATE
# -- w092 accumulated 46 of them over two days. The marker below makes the wedge sticky and
# machine-wide: it sits next to the flock file (node-local /tmp, shared by every run process on the
# host), and once it exists no eval on that card spawns anything at all. The failure is then instant,
# so the run-level empty-generation stop halts the run within one generation instead of never.
def _dead_marker_path(gpu: str) -> Path:
    lock_dir = Path(os.environ.get("TRIMUL_LOCK_DIR", tempfile.gettempdir()))
    return lock_dir / f"learning_evolve-trimul-gpu{_gpu_key(gpu)}.dead"


def _mark_gpu_dead(gpu: str, pgid: int) -> Path:
    marker = _dead_marker_path(gpu)
    with contextlib.suppress(OSError):
        marker.write_text(json.dumps({
            "pgid": pgid,
            "host": os.uname().nodename,
            "marked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "why": f"eval process group survived SIGKILL for {_UNKILLABLE_WAIT_SECONDS:.0f}s; "
                   "GPU driver wedged (see dmesg for nvidia_uvm / Xid). Needs admin reset/reboot. "
                   "This file auto-clears when the wedged group is gone; delete it to retry sooner.",
        }, indent=2))
    return marker


def _gpu_dead_reason(gpu: str) -> str | None:
    """If this card was declared dead, say why; self-heal the marker when the wedge has cleared.

    The wedged process group outliving SIGKILL is what MADE the card dead, so its disappearance
    (driver finally faulted it out, or the node rebooted -- which also empties /tmp) is the one
    user-space observable that the card may be usable again. A recycled pgid shows up as
    ProcessLookupError or PermissionError all the same: either way the group we recorded is gone.
    """
    marker = _dead_marker_path(gpu)
    try:
        info = json.loads(marker.read_text())
    except FileNotFoundError:
        return None
    except Exception:
        info = {}
    pgid = info.get("pgid")
    if isinstance(pgid, int):
        try:
            os.killpg(pgid, 0)
        except (ProcessLookupError, PermissionError):
            with contextlib.suppress(OSError):
                marker.unlink()
            logger.warning("dead-GPU marker %s cleared: wedged group %d is gone", marker, pgid)
            return None
    return (f"GPU marked dead at {info.get('marked_at', '?')} on {info.get('host', '?')} "
            f"(wedged pgid {pgid}, marker {marker}). "
            f"{info.get('why', '')} -- infra failure, not the candidate.")


def _eval_settings(problem_type: str, overrides: dict | None = None) -> dict:
    """Resolve where and how to grade, in precedence order: explicit override, env var, default.

    The overrides come from ``--trimul-eval-*`` via ``EnvConfig.evaluator_options``, so a sweep file
    can pin the interpreter and the card and have both recorded in the run's config.json. The
    environment variables stay supported for one-off shell use and for the standalone commands in
    ``gpumode_local/reference/README.md``.
    """
    o = overrides or {}
    return {
        "gpu": o.get("eval_gpu") or os.environ.get("TRIMUL_EVAL_GPU")
               or _DEFAULT_GPU.get(problem_type, INHERIT_GPU),
        "mode": o.get("eval_mode") or os.environ.get("TRIMUL_EVAL_MODE") or DEFAULT_EVAL_MODE,
        # expanduser: these paths are handed to subprocess argv, which never sees shell tilde
        # expansion -- and the Bosch sweep files say ~/venvs/kernel-eval/bin/python on purpose,
        # since the absolute home prefix differs per user there.
        "python": os.path.expanduser(o.get("eval_python") or os.environ.get("TRIMUL_EVAL_PYTHON")
                                     or DEFAULT_EVAL_PYTHON),
        "evaluate_py": Path(o.get("evaluate_py") or os.environ.get("TRIMUL_EVALUATE_PY")
                            or DEFAULT_EVALUATE_PY).expanduser(),
    }


class TrimulLocalReward(BaseRewardEvaluator):
    """Score a kernel by shelling out to ``evaluate.py`` while holding the grading GPU.

    ``evaluate.py --json`` writes a one-element list; the element carries ``score_us`` (``None`` on
    failure), a ``failed_phase`` key when something went wrong, and per-phase ``stderr``. That is the
    whole contract this class depends on -- verified against real success and failure runs rather
    than read off the source.
    """

    # In practice base.Environment always passes eval_timeout from the problem spec (registry.py)
    # or --eval-timeout, so this default is a fallback for direct construction only. Keep it equal
    # to the registry value; the sizing rationale lives on the registry entry.
    def __init__(self, problem_type, log_dir, eval_timeout: int = 1500, num_cpus_per_task: int = 1,
                 eval_python: str | None = None, eval_gpu: str | None = None,
                 evaluate_py: str | None = None, eval_mode: str | None = None, **kwargs):
        # The first four are what base.Environment._run_verification passes to every evaluator; the
        # rest arrive from EnvConfig.evaluator_options, i.e. from --trimul-eval-* in the sweep file.
        self.problem_type = problem_type
        self.log_dir = log_dir
        self.eval_timeout = eval_timeout
        self._overrides = {"eval_python": eval_python, "eval_gpu": eval_gpu,
                           "evaluate_py": evaluate_py, "eval_mode": eval_mode}
        self._last_timing: dict = {}

    def _fail(self, msg: str, failure_type: str, timing: dict) -> dict:
        self._last_timing = timing
        return {
            "reward": 0.0,
            "msg": msg,
            "correctness": 0.0,
            "raw_score": 0.0,
            "result_construction": [],
            "stdout": "",
            "failure_type": failure_type,
        }

    def get_reward(self, code: str, state: State) -> dict:
        cfg = _eval_settings(self.problem_type, self._overrides)

        # The upstream gates (env.py:122-126), before anything is written or the card is taken.
        # Both problems here are trimul, so both gates apply to both.
        no_timing = {"queue_seconds": 0.0, "eval_seconds": 0.0}
        if _REQUIRED_TOKEN not in (code or ""):
            return self._fail("Code must contain @triton.jit.", "invalid_result", no_timing)
        if _BANNED_TOKEN in (code or ""):
            return self._fail("Identity kernel is not allowed.", "invalid_result", no_timing)

        if not cfg["evaluate_py"].exists():
            # Ours, not the candidate's: surface it as infra so it is not read as a bad kernel.
            return self._fail(
                f"grader missing: {cfg['evaluate_py']}", "harness_error",
                {"queue_seconds": 0.0, "eval_seconds": 0.0},
            )

        # Fail BEFORE spawning anything if an earlier eval declared this card dead: a process that
        # touches a wedged card joins it in unkillable D-state and costs ~56 min to give up on.
        dead = _gpu_dead_reason(cfg["gpu"])
        if dead:
            return self._fail(dead, "harness_error", no_timing)

        with tempfile.TemporaryDirectory(prefix="trimul-cand-") as tmp:
            cand = Path(tmp) / "candidate.py"
            cand.write_text(code or "")
            out_json = Path(tmp) / "result.json"
            cmd = [
                cfg["python"], str(cfg["evaluate_py"]), str(cand),
                "--task", "trimul",          # one task file serves both problems
                "--mode", cfg["mode"],
                "--json", str(out_json),
            ]
            if cfg["gpu"] != INHERIT_GPU:
                # Only name a card when one was actually configured; otherwise evaluate.py leaves
                # CUDA_VISIBLE_DEVICES alone and the subprocess grades on whatever the scheduler
                # granted (see INHERIT_GPU above for why overriding it under LSF is dangerous).
                cmd += ["--gpu", cfg["gpu"]]

            t_queue = time.perf_counter()
            with _gpu_guard(cfg["gpu"]):     # <-- the whole point; see the module docstring
                queue_seconds = time.perf_counter() - t_queue
                t_eval = time.perf_counter()
                # Re-check under the lock: when the wedge is first detected, a generation's worth of
                # sibling threads is already queued on the flock behind the discoverer, and each
                # would otherwise spawn one more process into the dead driver as it gets the lock.
                dead = _gpu_dead_reason(cfg["gpu"])
                if dead:
                    return self._fail(dead, "harness_error",
                                      {"queue_seconds": round(queue_seconds, 3),
                                       "eval_seconds": 0.0})
                # start_new_session=True makes proc.pid the pgid, so _reap_group below can take
                # out evaluate.py, eval.py AND the spawn worker that owns the CUDA context. The
                # flock must not be released until that whole group is provably dead -- otherwise
                # the "free" card is handed to the next candidate with a context still on it.
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    start_new_session=True,
                )
                timed_out = False
                try:
                    proc_stdout, proc_stderr = proc.communicate(timeout=self.eval_timeout)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    proc_stdout, proc_stderr = "", ""
                finally:
                    reaped = _reap_group(proc.pid)
                    if timed_out:
                        # The group is dead (or being killed); collect the direct child so it
                        # does not linger as a zombie. Output is best-effort at this point.
                        with contextlib.suppress(Exception):
                            proc_stdout, proc_stderr = proc.communicate(timeout=10)
                    if not reaped:
                        # SIGKILL did not land: a kernel is wedged on the card and the worker sits
                        # in D-state until the driver faults it out. Hold the flock and wait --
                        # a stalled run is recoverable, a run grinding into a bricked card is not.
                        logger.critical(
                            "eval process group %d survived SIGKILL; kernel likely wedged on the "
                            "GPU. Holding the GPU lock up to %.0fs while polling.",
                            proc.pid, _UNKILLABLE_WAIT_SECONDS)
                        wait_deadline = time.monotonic() + _UNKILLABLE_WAIT_SECONDS
                        while time.monotonic() < wait_deadline:
                            if _reap_group(proc.pid):
                                reaped = True
                                break
                            logger.warning("still waiting on unkillable eval group %d", proc.pid)
                eval_seconds = time.perf_counter() - t_eval
                timing = {"queue_seconds": round(queue_seconds, 3),
                          "eval_seconds": round(eval_seconds, 3)}
                if not reaped:
                    # Give up loudly AND declare the card dead machine-wide. Returning a plain
                    # harness_error here was not enough (rng-dl01-w092, 2026-08-15..17): the run
                    # kept feeding candidates to the wedged driver at one unkillable D-state
                    # process per ~56 min cycle, 46 of them over two days. With the marker, every
                    # later candidate on this card -- this run or any sibling process -- fails in
                    # microseconds without spawning, so the empty-generation stop actually fires.
                    marker = _mark_gpu_dead(cfg["gpu"], proc.pid)
                    logger.critical(
                        "declaring GPU dead: eval group %d survived SIGKILL for %.0fs "
                        "(driver wedge; see dmesg). Marker: %s. Needs admin GPU reset/reboot.",
                        proc.pid, _UNKILLABLE_WAIT_SECONDS, marker)
                    return self._fail(
                        f"GPU still held by unkillable eval process group {proc.pid} after "
                        f"{_UNKILLABLE_WAIT_SECONDS:.0f}s -- driver wedged; card declared dead "
                        f"({marker}). Infra failure, not the candidate.",
                        "harness_error", timing)
                if timed_out:
                    return self._fail(
                        f"Process timed out after {self.eval_timeout}s (process group killed)",
                        "eval_timeout", timing)

            if not out_json.exists():
                # The grader itself did not get far enough to write results.
                tail = (proc_stderr or proc_stdout or "")[-1500:]
                return self._fail(f"grader wrote no results:\n{tail}", "harness_error", timing)
            try:
                runs = json.loads(out_json.read_text())
                run = runs[0]
            except Exception as e:
                return self._fail(f"unreadable grader output: {e}", "harness_error", timing)

        score_us = run.get("score_us")
        phases = run.get("phases", {})
        scoring_phase = phases.get(cfg["mode"], {})

        if score_us is None or not scoring_phase.get("passed", False):
            # A genuine failure of the candidate: wrong output, compile error, OOM, crash.
            # In leaderboard mode the failure is usually in the TEST phase, in which case there is no
            # entry under phases["leaderboard"] at all -- read the detail off whichever phase failed,
            # or this reports an empty error.
            #
            # Two ways a phase fails, and they surface differently: a kernel that RAISES leaves a
            # traceback on stderr, while one that merely returns the wrong numbers exits cleanly and
            # reports per-shape verdicts through the popcorn protocol -- stderr is empty. Handling
            # only the first left "FAILED in the 'test' phase" with nothing after it for the
            # commonest failure of all. (This msg goes to the log, never to the model, so being more
            # informative than upstream's bare "Failed to pass test cases." costs no fidelity.)
            failed_phase = run.get("failed_phase", cfg["mode"])
            entry = phases.get(failed_phase, scoring_phase)
            full_stderr = entry.get("stderr") or ""
            detail = full_stderr[-1500:]
            tests = entry.get("tests") or {}
            if not detail.strip() and tests.get("failed"):
                shapes = "; ".join(f"[{f['idx']}] {f['spec']}: {f['error']}"
                                   for f in tests["failed"][:3])
                detail = (f"{tests['passed']}/{tests['count']} shapes passed. "
                          f"First failures: {shapes}")[:1500]
            self._last_timing = timing
            # A context-creation failure is OURS (a leaked context bricked the Exclusive_Process
            # card -- see _INFRA_SIGNATURES), not the candidate's. Scoring it as process_crash
            # burns the candidate and, at 96/generation, trips the empty-generation stop with a
            # message blaming the kernels. Match against the FULL stderr, not the 1500-char tail:
            # the signature sits mid-traceback and can be truncated out of the tail.
            if any(sig in full_stderr for sig in _INFRA_SIGNATURES):
                return self._fail(
                    f"GPU unavailable (infra, not the candidate) in the '{failed_phase}' "
                    f"phase:\n{detail}", "harness_error", timing)
            return {
                "reward": 0.0,
                "msg": f"FAILED in the '{failed_phase}' phase -- no score.\n{detail}",
                "correctness": 0.0,
                "raw_score": 0.0,
                "result_construction": [],
                "stdout": "",
                "failure_type": "process_crash",
            }

        score_us = float(score_us)
        self._last_timing = timing
        return {
            # TTT-Discover's reward (env.py:155). Lower runtime -> higher reward.
            "reward": SCORE_SCALE / score_us,
            "msg": f"\nOverall leaderboard score (microseconds, geom): {score_us} us",
            "correctness": 1.0,
            "raw_score": score_us,
            # Carry nothing across states: the construction is the code itself.
            "result_construction": [],
            "stdout": "",
        }


# TTT-Discover's line, verbatim (env.py:204). Typo "trition" is theirs; left alone.
_HW_RULE_H100 = "- You must use trition 3.3.1 and these kernels will be run on an H100."
# The A100 rewrite. Naming the shared-memory ceiling is not decoration: the first smoke run lost a
# candidate to `OutOfResources: Required: 393216, Hardware limit: 166912`, which is an H100-legal
# block size meeting an A100.
_HW_RULE_A100 = (
    "- You must use trition 3.3.1 and these kernels will be run on an NVIDIA A100 80GB (sm80).\n"
    "- The A100 allows at most 166912 bytes of shared memory per kernel launch, so block sizes or\n"
    "  num_stages needing more than that will fail to launch. There are no fp8 tensor cores."
)


class _TrimulEnvBase(Environment):
    reward_function = TrimulLocalReward
    state_type = State
    uses_sandbox = False        # grading is a local subprocess; the loop skips init_ray

    hardware_rule: str = _HW_RULE_H100

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        # env.py:178 -- trimul starts from NO seed program (unlike mla_decode). The -1e6 value is
        # TTT-Discover's, and with is_maximize() False it renders as "current runtime 1000000 us",
        # i.e. "you have nothing yet".
        return State(timestep=-1, construction=None, code="", value=-1_000_000)

    def is_maximize(self) -> bool:
        return False            # runtime in microseconds; lower is better

    def _should_keep_code_separators(self) -> bool:
        return False            # env.py:181 -- hand the grader bare python, no ``` fence

    # --- Prompt zone 1: constant across parents, so vLLM can prefix-cache it ---
    def problem_intro(self) -> str:
        return TRIMUL_PROMPT

    # --- Prompt zone 3: rules, then the parent to improve on (rendered LAST) ---
    def improvement_task(self) -> str:
        if self.show_parent_solution:
            current = "--- Current kernel to improve upon ---\n" + self.initial_state.to_prompt(
                TARGET_US, metric_name="runtime (microseconds)", maximize=False, language="python",
            )
        else:
            current = objective_only_prompt(
                TARGET_US, metric_name="runtime (microseconds)", maximize=False)
        # Rules copied from env.py:199-206, unchanged in wording apart from the hardware line. THREE
        # deliberate deviations from upstream, all matching what this project does elsewhere:
        #
        # 1. ORDER. TTT-Discover renders {state_ctx} BEFORE the rules. Here the rules come first and
        #    the current kernel LAST, because Environment.improvement_task's contract is that the only
        #    parent-dependent text is the trailing solution -- that is what keeps intro + context block
        #    + rules a shared prefix across a generation's parents, so vLLM re-prefills only the tail.
        #    Same text, same information, different position. See circle_packing.improvement_task.
        # 2. The trailing <strategy> request. Upstream asks for it only in circle_packing; this project
        #    added it to erdos and ac1/ac2 as well, and without it `state.strategy` is always empty
        #    here, which silently turns --include-strategy and every strategy-level ICL arm into a
        #    no-op (context.selection.render_solution falls back to code when strategy is blank).
        # 3. The hardware line, per subclass -- see _HW_RULE_A100.
        return f"""
Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
{self.hardware_rule}
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.

{current}

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""


class TrimulH100Env(_TrimulEnvBase):
    """TTT-Discover's prompt verbatim. The faithful baseline; run this on an H100."""
    hardware_rule = _HW_RULE_H100


class TrimulA100Env(_TrimulEnvBase):
    """Same task, prompt retargeted at an A100 so candidates are not handicapped by the wrong card."""
    hardware_rule = _HW_RULE_A100
