"""Sandboxed code execution + reward evaluation (tinker-free).

Vendored from TTT-Discover: candidate solutions run in a hardened subprocess dispatched via a
Ray remote task, with CPUs allocated by a detached ``cpu_scheduler`` actor. Call
:func:`init_ray` once before constructing any evaluator.
"""
from sandbox.base_reward_evaluator import BaseRewardEvaluator
from sandbox.sandbox_reward_evaluator import SandboxRewardEvaluator
from sandbox.ray_setup import init_ray

__all__ = ["BaseRewardEvaluator", "SandboxRewardEvaluator", "init_ray"]
