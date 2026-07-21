from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import uuid

import numpy as np


def to_json_serializable(obj):
    """Convert numpy arrays and other non-JSON-serializable types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_json_serializable(v) for v in obj]
    return obj


class State(ABC):
    id: str  # unique identifier for this state
    timestep: int  # the training step this state was first visited at
    value: float  # Expected value of starting from this state (higher = better)
    code: str  # the code that generated the construction
    construction: list[Any]  # construction of the state
    parent_values: list[float]  # list of ancestor values (most recent first) for terminal value estimation
    parents: list[dict]  # list of parent refs [{"id": ..., "timestep": ...}, ...] (most recent first)
    observation: str  # stdout/logs from the code that created this state
    strategy: str  # the <strategy>...</strategy> reasoning block the model emitted alongside the code

    def __init__(
        self,
        timestep: int,
        construction: list[Any],
        code: str,
        value: float = None,
        parent_values: list[float] = None,
        parents: list[dict] = None,
        id: str = None,
        observation: str = "",
        strategy: str = "",
    ):
        self.id = id if id is not None else str(uuid.uuid4())
        self.timestep = timestep
        self.value = value
        self.construction = to_json_serializable(construction)
        self.code = code
        self.parent_values = parent_values if parent_values is not None else []
        self.parents = parents if parents is not None else []
        self.observation = observation
        self.strategy = strategy

    def to_dict(self) -> dict:
        return {
            "type": "State",
            "id": self.id,
            "timestep": self.timestep,
            "value": self.value,
            "parent_values": self.parent_values,
            "parents": self.parents,
            "observation": self.observation,
            "construction": to_json_serializable(self.construction),
            "code": self.code,
            "strategy": self.strategy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> State:
        return cls(
            timestep=d["timestep"],
            construction=d["construction"],
            code=d["code"],
            value=d.get("value"),
            parent_values=d.get("parent_values", []),
            parents=d.get("parents", []),
            id=d.get("id"),
            observation=d.get("observation", ""),
            strategy=d.get("strategy", ""),
        )

    def to_prompt(self, target, metric_name: str = "value", maximize: bool = True, language: str = ""):
        value_ctx = f"You are iteratively optimizing {metric_name}."
        improvement_direction = "higher" if maximize else "lower"

        has_code = self.code and self.code.strip()
        if has_code:
            value_ctx += f"\nHere is the last code we ran:\n"
            if language:
                value_ctx += f"```{language}\n{self.code}\n```"
            else:
                value_ctx += f"{self.code}"
        else:
            value_ctx += f"\nNo previous code available."
        # Value context: show before/after if we have parent values
        if self.parent_values and self.value is not None and self.construction:
            before_value = self.parent_values[0] if maximize else -self.parent_values[0]
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nHere is the {metric_name} before and after running the code above ({improvement_direction} is better): {before_value:.6f} -> {after_value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        elif self.value is not None:
            after_value = self.value if maximize else -self.value
            current_gap = target - after_value if maximize else after_value - target
            value_ctx += f"\nCurrent {metric_name} (higher is better): {after_value:.6f}"
            value_ctx += f"\nTarget: {target}. Current gap: {current_gap:.6f}. Further improvements will also be generously rewarded."
        else:
            value_ctx += f"\nTarget {metric_name}: {target}"
        # Show previous stdout if available
        if self.observation and self.observation.strip():
            stdout = self.observation.strip()
            if len(stdout) > 500:
                stdout = "\n\n\t\t ...(TRUNCATED)...\n" + stdout[-500:]
            value_ctx += f"\n\n--- Previous Program Output ---\n{stdout}\n--- End Output ---"
        
        return value_ctx


def _state_class_by_name(name: str) -> type:
    """Resolve a state type name to a State subclass (used when env_type is not available)."""
    def _all_subclasses(cls: type) -> set[type]:
        return set(cls.__subclasses__()) | {s for c in cls.__subclasses__() for s in _all_subclasses(c)}
    for cls in [State] + list(_all_subclasses(State)):
        if cls.__name__ == name:
            return cls
    raise ValueError(f"Unknown state type: {name}")


def state_from_dict(d: dict | None, state_type: type | None = None) -> State | None:
    """Deserialize state from dict. Prefer passing state_type from env_type.state_type when available."""
    if d is None:
        return None
    cls = state_type if state_type is not None else _state_class_by_name(d.get("type", "State"))
    return cls.from_dict(d)
