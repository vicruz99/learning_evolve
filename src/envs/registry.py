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
    # Synthetic smoke-test problem (in-process grading, no sandbox). metric_name is deliberately
    # generic ("score") so the "ee"-counting mechanism stays hidden from the model. eval_timeout is
    # unused by the in-process evaluator; entrypoint is documentary only (no code is executed).
    "toy":               ProblemSpec(ToyEeEnv,               "toy",  1,   30, "score",        True,  "sentence"),
}


def get_problem(name: str) -> ProblemSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown problem '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
