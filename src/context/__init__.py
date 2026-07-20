"""ICL context-selection strategies (which past solutions go in the prompt, and how they render)."""
from context.selection import (
    select_best_n,
    select_recent_n,
    build_context_block,
    render_solution,
    STRATEGIES,
    get_strategy,
)

__all__ = [
    "select_best_n",
    "select_recent_n",
    "build_context_block",
    "render_solution",
    "STRATEGIES",
    "get_strategy",
]
