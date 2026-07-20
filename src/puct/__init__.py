"""Shared search/buffer for test-time discovery.

Vendored (tinker-free) from TTT-Discover: the solution ``State`` and the ``PUCTSampler``
buffer + parent-selection strategy. Reused by the ICL loop and, later, the RL/SFT variants.
"""
from puct.state import State, state_from_dict, to_json_serializable
from puct.sampler import (
    StateSampler,
    PUCTSampler,
    create_sampler,
    get_or_create_sampler_with_default,
    create_initial_state,
)

__all__ = [
    "State",
    "state_from_dict",
    "to_json_serializable",
    "StateSampler",
    "PUCTSampler",
    "create_sampler",
    "get_or_create_sampler_with_default",
    "create_initial_state",
]
