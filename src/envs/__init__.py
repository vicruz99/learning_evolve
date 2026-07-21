"""Problem definitions for the ICL harness.

Each problem is a vendored TTT-Discover ``Environment`` subclass (prompts, seed solutions, and
reward evaluators kept verbatim) sitting on a slim, tinker-free base. Use :data:`REGISTRY` /
:func:`get_problem` to look one up by name.
"""
from envs.base import Environment, EnvConfig, VerifyResult, RolloutResult
from envs.erdos_min_overlap import ErdosMinOverlapEnv
from envs.circle_packing import CirclePackingEnv
from envs.ac_inequalities import AutoCorrInequalityEnv
from envs.toy_ee import ToyEeEnv
from envs.registry import REGISTRY, ProblemSpec, get_problem

__all__ = [
    "Environment",
    "EnvConfig",
    "VerifyResult",
    "RolloutResult",
    "ErdosMinOverlapEnv",
    "CirclePackingEnv",
    "AutoCorrInequalityEnv",
    "ToyEeEnv",
    "REGISTRY",
    "ProblemSpec",
    "get_problem",
]
