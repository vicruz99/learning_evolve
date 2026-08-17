"""Problem registry: name -> environment class + evaluation defaults.

The ``num_cpus_per_task`` / ``eval_timeout`` defaults match TTT-Discover's ``discover_*`` runners
(erdos: 1 cpu / 1100s, circle_packing: 1 / 530, ac1|ac2: 2 / 1100).
"""
from __future__ import annotations

from dataclasses import dataclass

from envs.base import Environment
from envs.erdos_min_overlap import ErdosMinOverlapEnv
from envs.circle_packing import CirclePackingEnv
from envs.ac_inequalities import AutoCorrInequalityEnv
from envs.kernel_trimul import TrimulA100Env, TrimulH100Env
from envs.toy_ee import ToyEeEnv


@dataclass(frozen=True)
class ProblemSpec:
    env_type: type[Environment]
    problem_type: str
    num_cpus_per_task: int
    eval_timeout: int
    metric_name: str          # native metric shown in prompts / context block
    maximize: bool            # True if higher raw score is better (matches env.is_maximize())
    entrypoint: str           # function the generated program must define (for solution-file headers)


REGISTRY: dict[str, ProblemSpec] = {
    "erdos":             ProblemSpec(ErdosMinOverlapEnv,    "",    1, 1100, "C₅ bound",     False, "run"),
    "circle_packing_26": ProblemSpec(CirclePackingEnv,      "26",  1,  530, "sum of radii", True,  "run_packing"),
    "circle_packing_32": ProblemSpec(CirclePackingEnv,      "32",  1,  530, "sum of radii", True,  "run_packing"),
    "ac1":               ProblemSpec(AutoCorrInequalityEnv, "ac1", 2, 1100, "upper bound",  False, "propose_candidate"),
    "ac2":               ProblemSpec(AutoCorrInequalityEnv, "ac2", 2, 1100, "lower bound",  True,  "construct_function"),
    # Kernel optimisation (TTT-Discover's gpu_mode task). Graded by a LOCAL GPU subprocess, not the
    # Ray sandbox: num_cpus_per_task is nominal (no CPU group is taken). Only ONE eval runs at a time
    # per card, across threads AND processes -- see envs.kernel_trimul._gpu_guard for why that is not
    # negotiable.
    #
    # eval_timeout 1500 is a BACKSTOP, not a budget. Grading runs in TTT-Discover's `leaderboard`
    # mode, which is two eval.py invocations (18 correctness shapes, then 7 timed ones), and
    # task.yml caps each at 1200 s -- evaluate.py enforces those itself and reports a clean failure.
    # The outer timeout sits above one phase cap (a hung phase fails cleanly at 1200 s and, since a
    # failed test phase skips the benchmark phase, no clean path reaches ~1300 s), so `eval_timeout`
    # in the logs always means "the grader hung", never "a slow kernel used its allowance". It was
    # 2700 (above the 2400 sum of both caps), but the 2026-08-14 w093 incident showed a single hang
    # stalls its whole parent group for the full allowance, so it now sits as low as that semantic
    # permits; the real hang protection is the process-group reaping in kernel_trimul. A healthy
    # eval is 36 s; the slowest PASSING eval over 2400 h100 candidates was 1127 s.
    #
    # TWO PROBLEMS, ONE TASK: the prompts differ only in which card they name, but that drives block
    # sizes and an H100-legal config dies on an A100's smaller shared memory. Scores are comparable
    # WITHIN an architecture, never across one -- TTT-Discover's best kernel does 2198 us on an A100
    # and 1161 us on an H100. Pick the one matching the card you are grading on.
    "trimul_a100":       ProblemSpec(TrimulA100Env, "trimul_a100", 1, 1500, "runtime (us)", False, "custom_kernel"),
    "trimul_h100":       ProblemSpec(TrimulH100Env, "trimul_h100", 1, 1500, "runtime (us)", False, "custom_kernel"),
    # Synthetic smoke-test problem (in-process grading, no sandbox). metric_name is deliberately
    # generic ("score") so the "ee"-counting mechanism stays hidden from the model. eval_timeout is
    # unused by the in-process evaluator; entrypoint is documentary only (no code is executed).
    "toy":               ProblemSpec(ToyEeEnv,               "toy",  1,   30, "score",        True,  "sentence"),
}


def get_problem(name: str) -> ProblemSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
