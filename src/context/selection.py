"""ICL context selection strategies.

The frozen model conditions on a block of *past solutions* injected into the prompt. This module
picks which solutions go in and renders them. Strategy v1 (the only one implemented): the **n best
past solutions** by buffered value.

`State.value` is normalized so higher is always better (minimize envs store the negated raw score),
so "best" is simply the top-n by value. We render each in the problem's native metric direction
(`maximize` controls the sign shown to the model).

Written to be pluggable: add new `select_*` functions and swap them in the loop.
"""
from __future__ import annotations

from puct.state import State

# Rough chars-per-token estimate for the budget guard (avoids a tokenizer dependency).
_CHARS_PER_TOKEN = 4.0


def select_best_n(states: list[State], n: int, exclude_id: str | None = None) -> list[State]:
    """Top-``n`` states by value (higher = better), optionally excluding one id (the current parent).

    Shortfall is graceful: if fewer than ``n`` candidates exist, ALL of them are returned (no error).
    Early on, the buffer holds only the ``groups_per_batch`` seeds, so you get up to
    ``groups_per_batch - 1`` context solutions until the buffer fills.
    """
    cands = [s for s in states if s.value is not None and s.id != exclude_id]
    cands.sort(key=lambda s: s.value, reverse=True)
    return cands[:n]


def select_recent_n(states: list[State], n: int, exclude_id: str | None = None) -> list[State]:
    """Most recent ``n`` states by creation step (``timestep``), newest first.

    Ties (same timestep) keep buffer order; initial seeds have timestep=-1 so they rank last.
    """
    cands = [s for s in states if s.value is not None and s.id != exclude_id]
    cands.sort(key=lambda s: s.timestep, reverse=True)
    return cands[:n]


def _select_diverse_n(states: list[State], n: int, exclude_id: str | None = None) -> list[State]:
    raise NotImplementedError(
        "The 'diverse' context strategy (high-reward + dissimilar mix) is not implemented yet."
    )


# Registry of context-selection strategies (selectable via ICLConfig.context_strategy).
STRATEGIES = {
    "best": select_best_n,
    "recent": select_recent_n,
    "diverse": _select_diverse_n,  # registered stub — raises NotImplementedError
}


def get_strategy(name: str):
    """Return the selector function for ``name`` (raises KeyError on unknown)."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown context strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]


def _fenced(code: str, language: str = "python") -> str:
    code = (code or "").strip()
    if not code:
        return "(no code)"
    if code.startswith("```"):
        return code  # already fenced (e.g. seed programs / parsed children keep separators)
    return f"```{language}\n{code}\n```"


def _native_score(state: State, maximize: bool) -> float:
    """Return the score in the problem's native direction (value is stored higher=better)."""
    return state.value if maximize else -state.value


def render_solution(state: State, idx: int, metric_name: str, maximize: bool, language: str = "python") -> str:
    score = _native_score(state, maximize)
    return f"[Solution {idx}] {metric_name} = {score:.6f}\n{_fenced(state.code, language)}"


def build_context_block(
    states: list[State],
    metric_name: str = "score",
    maximize: bool = True,
    language: str = "python",
    max_context_tokens: int | None = None,
) -> str:
    """Render selected states into a prompt block, trimming lowest-ranked to fit a token budget.

    Args:
        states: already-selected, already-ranked (best first) states.
        max_context_tokens: approximate cap on the block's size (chars/4 heuristic). None = no cap.
    """
    if not states:
        return ""

    header = "\n\n--- Best solutions found so far (highest-scoring first) ---\n"
    footer = "\n--- End of best solutions ---\n"

    rendered: list[str] = []
    budget_chars = None if max_context_tokens is None else int(max_context_tokens * _CHARS_PER_TOKEN)
    used = len(header) + len(footer)
    for i, s in enumerate(states, 1):
        piece = "\n" + render_solution(s, i, metric_name, maximize, language) + "\n"
        if budget_chars is not None and used + len(piece) > budget_chars and rendered:
            break  # keep at least one solution; stop before overflowing
        rendered.append(piece)
        used += len(piece)

    return header + "".join(rendered) + footer
