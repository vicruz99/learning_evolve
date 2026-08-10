"""TriMul kernel optimisation -- TTT-Discover's ``gpu_mode`` task, graded on a local GPU.

Port of ``discover/examples/gpu_mode/env.py`` (``GpuModeEnv`` / ``GpuModeRewardEvaluator``) with one
substitution: TTT-Discover grades through a Modal RPC, we grade on a local card via
``coding_agent_evolve/gpumode/evaluate.py``. The prompt body, the initial state and the reward formula
are unchanged -- see :mod:`envs.kernel_prompt` for the verbatim prompt text.

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
``--mode benchmark`` takes ~11-23 s wall and scores **2467 us** (mean of 3, spread 1.0 %). A candidate
that fails to compile fails in ~5 s. See ``gpumode_local/reference/README.md`` for the reference
numbers to expect on each card.

Configuration is by environment variable, because which card to grade on is a property of the
machine, not of the experiment:

===========================  ==================================================  ==================
variable                     meaning                                             default
===========================  ==================================================  ==================
``TRIMUL_EVAL_GPU``          ``CUDA_VISIBLE_DEVICES`` value for the eval          per problem (below)
``TRIMUL_EVAL_MODE``         ``test`` | ``benchmark`` | ``leaderboard``           ``benchmark``
``TRIMUL_EVAL_PYTHON``       interpreter that runs ``evaluate.py``                the scratch venv
``TRIMUL_EVALUATE_PY``       path to ``evaluate.py``                              repo copy
``TRIMUL_LOCK_DIR``          where the per-GPU flock files live                   ``/tmp``
===========================  ==================================================  ==================
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from envs.base import Environment
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

# Which card each problem grades on unless TRIMUL_EVAL_GPU says otherwise. guadiana's GPU 0 holds a
# vLLM server, so the A100 default is 1; a fresh H100 box is assumed to have the card at 0.
_DEFAULT_GPU = {"trimul_a100": "1", "trimul_h100": "0"}

# Serialises access to the grading GPU WITHIN this process. Module-level on purpose: every
# environment instance here contends for the same card, so a per-instance lock serialises nothing.
_GPU_LOCK = threading.Lock()


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
    lock_path = lock_dir / f"learning_evolve-trimul-gpu{gpu}.lock"
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


def _eval_settings(problem_type: str) -> dict:
    return {
        "gpu": os.environ.get("TRIMUL_EVAL_GPU", _DEFAULT_GPU.get(problem_type, "0")),
        "mode": os.environ.get("TRIMUL_EVAL_MODE", "benchmark"),
        "python": os.environ.get("TRIMUL_EVAL_PYTHON", DEFAULT_EVAL_PYTHON),
        "evaluate_py": Path(os.environ.get("TRIMUL_EVALUATE_PY", str(DEFAULT_EVALUATE_PY))),
    }


class TrimulLocalReward(BaseRewardEvaluator):
    """Score a kernel by shelling out to ``evaluate.py`` while holding the grading GPU.

    ``evaluate.py --json`` writes a one-element list; the element carries ``score_us`` (``None`` on
    failure), a ``failed_phase`` key when something went wrong, and per-phase ``stderr``. That is the
    whole contract this class depends on -- verified against real success and failure runs rather
    than read off the source.
    """

    def __init__(self, problem_type, log_dir, eval_timeout: int = 1200, num_cpus_per_task: int = 1, **kwargs):
        # Same four kwargs base.Environment._run_verification passes to every evaluator.
        self.problem_type = problem_type
        self.log_dir = log_dir
        self.eval_timeout = eval_timeout
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
        cfg = _eval_settings(self.problem_type)
        if not cfg["evaluate_py"].exists():
            # Ours, not the candidate's: surface it as infra so it is not read as a bad kernel.
            return self._fail(
                f"grader missing: {cfg['evaluate_py']}", "harness_error",
                {"queue_seconds": 0.0, "eval_seconds": 0.0},
            )

        with tempfile.TemporaryDirectory(prefix="trimul-cand-") as tmp:
            cand = Path(tmp) / "candidate.py"
            cand.write_text(code or "")
            out_json = Path(tmp) / "result.json"
            cmd = [
                cfg["python"], str(cfg["evaluate_py"]), str(cand),
                "--task", "trimul",          # one task file serves both problems
                "--gpu", cfg["gpu"],
                "--mode", cfg["mode"],
                "--json", str(out_json),
            ]

            t_queue = time.perf_counter()
            with _gpu_guard(cfg["gpu"]):     # <-- the whole point; see the module docstring
                queue_seconds = time.perf_counter() - t_queue
                t_eval = time.perf_counter()
                try:
                    proc = subprocess.run(
                        cmd, capture_output=True, text=True, timeout=self.eval_timeout,
                    )
                except subprocess.TimeoutExpired:
                    return self._fail(
                        f"Process timed out after {self.eval_timeout}s", "eval_timeout",
                        {"queue_seconds": round(queue_seconds, 3),
                         "eval_seconds": round(time.perf_counter() - t_eval, 3)},
                    )
                eval_seconds = time.perf_counter() - t_eval

            timing = {"queue_seconds": round(queue_seconds, 3),
                      "eval_seconds": round(eval_seconds, 3)}

            if not out_json.exists():
                # The grader itself did not get far enough to write results.
                tail = (proc.stderr or proc.stdout or "")[-1500:]
                return self._fail(f"grader wrote no results:\n{tail}", "harness_error", timing)
            try:
                runs = json.loads(out_json.read_text())
                run = runs[0]
            except Exception as e:
                return self._fail(f"unreadable grader output: {e}", "harness_error", timing)

        score_us = run.get("score_us")
        phase = run.get("phases", {}).get(cfg["mode"], {})

        if score_us is None or not phase.get("passed", False):
            # A genuine failure of the candidate: wrong output, compile error, OOM, crash.
            failed_phase = run.get("failed_phase", cfg["mode"])
            tail = (phase.get("stderr") or "")[-1500:]
            self._last_timing = timing
            return {
                "reward": 0.0,
                "msg": f"FAILED in the '{failed_phase}' phase -- no score.\n{tail}",
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
        state_ctx = self.initial_state.to_prompt(
            TARGET_US, metric_name="runtime (microseconds)", maximize=False, language="python",
        )
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

--- Current kernel to improve upon ---
{state_ctx}

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""


class TrimulH100Env(_TrimulEnvBase):
    """TTT-Discover's prompt verbatim. The faithful baseline; run this on an H100."""
    hardware_rule = _HW_RULE_H100


class TrimulA100Env(_TrimulEnvBase):
    """Same task, prompt retargeted at an A100 so candidates are not handicapped by the wrong card."""
    hardware_rule = _HW_RULE_A100
