"""ICL context-selection strategies (which past solutions go in the prompt, and how they render)."""
from context.selection import (
    SelectionParams,
    SelectionResult,
    select_best_n,
    select_recent_n,
    build_context_block,
    render_solution,
    STRATEGIES,
    get_strategy,
    select,
    dedupe_seeds,
)

__all__ = [
    "SelectionParams",
    "SelectionResult",
    "select_best_n",
    "select_recent_n",
    "build_context_block",
    "render_solution",
    "STRATEGIES",
    "get_strategy",
    "select",
    "dedupe_seeds",
]
