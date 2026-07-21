"""Ranking keys for context selection.

A *ranking key* maps a ``State`` to a scalar where **higher = more preferred**. These are the raw
signals the selection engines rank/blend/normalize:

- ``quality``  : the buffered ``value`` (already normalized so higher = better).
- ``jump``     : improvement over the immediate parent (``value - parent_value``); seeds -> 0.
- ``recent``   : creation ``timestep`` (newest ranks highest).

``random`` is not a key (it has no per-state scalar); the engines handle it by shuffling.
"""
from __future__ import annotations

from puct.state import State


def quality(s: State) -> float:
    """Buffered value (higher = better; minimize envs store the negated raw score)."""
    return float(s.value) if s.value is not None else float("-inf")


def jump(s: State) -> float:
    """Improvement over the immediate parent: ``value - parent_values[0]``.

    Seeds (and any state without a recorded parent value) get ``0.0`` -- no measurable improvement,
    so they neither help nor dominate a jump-based ranking.
    """
    if s.value is None or not s.parent_values:
        return 0.0
    return float(s.value) - float(s.parent_values[0])


def recent(s: State) -> float:
    """Creation step; newest first. Seeds have ``timestep == -1`` so they rank last."""
    return float(s.timestep)


KEYS = {"quality": quality, "jump": jump, "recent": recent}


def get_key(name: str):
    if name not in KEYS:
        raise KeyError(f"Unknown ranking key '{name}'. Available: {sorted(KEYS)}")
    return KEYS[name]
