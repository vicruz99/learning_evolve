"""ICL context selection: which past solutions go in the prompt, and how they render.

The frozen model conditions on a block of *past solutions* injected into the prompt. This module
decides **which** buffered states to show (the *strategy*) and **what** of each to show (rendering).
Those two concerns are orthogonal:

* **Selection strategy** (``STRATEGIES``) -> returns a :class:`SelectionResult` (a positive block and
  an optional secondary block). Four engines cover every strategy; most named strategies are one
  engine with different parameters:

  =================  ========  ================================  ==========================
  strategy           engine    key(s) / pools                    diversity
  =================  ========  ================================  ==========================
  ``random``         topk      random shuffle                    none
  ``best``           topk      quality                           none
  ``recent``         topk      recency                           none
  ``biggest_jump``   topk      jump (improvement over parent)    none
  ``best_worst``     mix       quality + worst (fraction ``x``)  none
  ``best_jump``      mix       quality + biggest-jump (``x``)    none
  ``per_lineage``    lineage   quality (+worst via ``x``)        hard: <=1 per lineage-path
  ``best_diverse``   mmr       quality                           soft: tree-distance penalty
  ``informative``    mmr       quality + alpha*jump              soft: tree-distance penalty
  ``contrastive``    mmr       quality + low->mid negatives      soft: tree-distance penalty
  =================  ========  ================================  ==========================

* **Rendering** (``build_context_block`` / ``render_solution``): per-run flags ``include_code`` and
  ``include_strategy`` control whether each solution shows its code, its ``<strategy>`` reasoning, or
  both ("strategy-only" = strategy on, code off).

``State.value`` is normalized so higher is always better (minimize envs store the negated raw score);
we render each solution in the problem's native metric direction (``maximize`` controls the sign).

Docs with intuition for every strategy live in ``docs/strategies/``.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass, field

from puct.state import State
from context.ranking import quality, jump, recent, get_key
from context.lineage import same_lineage, tree_distance, lineage_similarity

# Rough chars-per-token estimate for the budget guard (avoids a tokenizer dependency).
_CHARS_PER_TOKEN = 4.0

_LABEL = {
    "quality": "Best solutions found so far (highest-scoring first)",
    "jump": "Solutions with the largest improvement over their parent",
    "recent": "Most recently generated solutions",
    "random": "A random sample of past solutions",
}


# ----------------------------------------------------------------------------- params / result
@dataclass
class SelectionParams:
    """Knobs shared across strategies (each strategy reads only the ones it needs)."""
    mix_fraction: float = 0.5       # x: share of n_context taken from the primary/"best" pool
    mmr_lambda: float = 0.7         # MMR quality<->diversity trade-off (1 = pure quality, 0 = pure spread)
    jump_alpha: float = 0.5         # informative: blend of value (alpha) vs jump (1-alpha) in the MMR quality term
    context_seed: int | None = None  # seed for the `random` strategy (reproducibility)


@dataclass
class SelectionResult:
    """The chosen context, split into a primary block and an optional secondary block.

    ``negatives`` holds the second pool (worst / biggest-jump / low-to-mid), empty for single-block
    strategies. Labels drive the section headers in the rendered prompt.
    """
    positives: list[State]
    negatives: list[State] = field(default_factory=list)
    positives_label: str = _LABEL["quality"]
    negatives_label: str | None = None

    def all(self) -> list[State]:
        return list(self.positives) + list(self.negatives)


# ----------------------------------------------------------------------------- primitives (kept: used directly by tests/loop)
def select_best_n(states: list[State], n: int, exclude_id: str | None = None) -> list[State]:
    """Top-``n`` states by value (higher = better), optionally excluding one id (the current parent).

    Shortfall is graceful: if fewer than ``n`` candidates exist, ALL of them are returned (no error).
    Early on, the buffer holds only the ``groups_per_batch`` seeds, so you get up to
    ``groups_per_batch - 1`` context solutions until the buffer fills.
    """
    cands = _candidates(states, exclude_id)
    cands.sort(key=quality, reverse=True)
    return cands[:n]


def select_recent_n(states: list[State], n: int, exclude_id: str | None = None) -> list[State]:
    """Most recent ``n`` states by creation step (``timestep``), newest first."""
    cands = _candidates(states, exclude_id)
    cands.sort(key=recent, reverse=True)
    return cands[:n]


# ----------------------------------------------------------------------------- shared internals
def _candidates(states: list[State], exclude_id: str | None) -> list[State]:
    return [s for s in states if s.value is not None and s.id != exclude_id]


def _minmax(vals: dict[str, float]) -> dict[str, float]:
    """Min-max normalize to [0,1]; a flat set (all equal) maps to 1.0 so it contributes neutrally."""
    if not vals:
        return {}
    lo, hi = min(vals.values()), max(vals.values())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in vals}
    return {k: (v - lo) / (hi - lo) for k, v in vals.items()}


def _quality_vector(cands: list[State], mode: str, alpha: float) -> dict[str, float]:
    """Per-state quality in [0,1] for the MMR objective. ``value`` or ``alpha*value + (1-alpha)*jump``."""
    vnorm = _minmax({s.id: quality(s) for s in cands})
    if mode == "value":
        return vnorm
    jnorm = _minmax({s.id: jump(s) for s in cands})
    return {s.id: alpha * vnorm[s.id] + (1.0 - alpha) * jnorm[s.id] for s in cands}


def _mmr_select(cands: list[State], n: int, qual: dict[str, float], lam: float) -> list[State]:
    """Greedy Maximal Marginal Relevance: pick high-quality states that are far apart in the tree.

    Each step maximizes ``lam*quality - (1-lam)*max_similarity_to_already_picked`` where similarity is
    lineage-based (:func:`lineage_similarity`). No embeddings; the similarity fn is pluggable.
    """
    pool = list(cands)
    picked: list[State] = []
    while pool and len(picked) < n:
        if not picked:
            best = max(pool, key=lambda s: qual[s.id])
        else:
            best = max(pool, key=lambda s: lam * qual[s.id]
                       - (1.0 - lam) * max(lineage_similarity(s, p) for p in picked))
        picked.append(best)
        pool.remove(best)
    return picked


def _greedy_lineage(ranked: list[State], n: int) -> list[State]:
    """Walk ``ranked`` (already in preference order); take a state only if it is NOT on the same
    root-to-node path as one already taken. This is the user's iterative lineage skip."""
    picked: list[State] = []
    for s in ranked:
        if len(picked) >= n:
            break
        if any(same_lineage(s, p) for p in picked):
            continue
        picked.append(s)
    return picked


def _split_bins(seq: list[State], k: int) -> list[list[State]]:
    """Split ``seq`` into ``k`` contiguous near-equal bins (for stratified low->mid sampling)."""
    if k <= 0:
        return []
    m = len(seq)
    return [seq[(i * m) // k:((i + 1) * m) // k] for i in range(k)]


def _stratified_diverse(pool: list[State], n: int) -> list[State]:
    """Pick ``n`` states spanning the pool's value range (low->mid), one lineage-diverse rep per bin."""
    if n <= 0 or not pool:
        return []
    ranked = sorted(pool, key=quality)          # ascending: worst -> least-bad
    if len(ranked) <= n:
        return ranked
    picked: list[State] = []
    for b in _split_bins(ranked, n):
        if not b:
            continue
        if not picked:
            choice = max(b, key=quality)         # top of the lowest bin
        else:
            # most lineage-distant from what we've taken; ties broken toward higher value
            choice = max(b, key=lambda s: (min(tree_distance(s, p) for p in picked), quality(s)))
        picked.append(choice)
    return picked[:n]


def _split_counts(n: int, x: float) -> tuple[int, int]:
    """Split ``n`` slots into (primary, secondary) by fraction ``x`` (clamped to [0,1])."""
    x = min(max(x, 0.0), 1.0)
    n_pos = round(x * n)
    return n_pos, n - n_pos


# ----------------------------------------------------------------------------- engines (as strategy factories)
def _topk_strategy(key_name: str):
    def strategy(states, n, params: SelectionParams, exclude_id=None) -> SelectionResult:
        cands = _candidates(states, exclude_id)
        if key_name == "random":
            rng = _random.Random(params.context_seed)
            rng.shuffle(cands)
            picked = cands[:n]
        else:
            picked = sorted(cands, key=get_key(key_name), reverse=True)[:n]
        return SelectionResult(positives=picked, positives_label=_LABEL[key_name])
    return strategy


def _mix_strategy(secondary_key: str, neg_label: str):
    def strategy(states, n, params: SelectionParams, exclude_id=None) -> SelectionResult:
        cands = _candidates(states, exclude_id)
        n_pos, n_neg = _split_counts(n, params.mix_fraction)
        positives = sorted(cands, key=quality, reverse=True)[:n_pos]
        pos_ids = {s.id for s in positives}
        rest = [s for s in cands if s.id not in pos_ids]
        if secondary_key == "worst":
            secondary = sorted(rest, key=quality)[:n_neg]              # lowest value first
        else:  # "jump"
            secondary = sorted(rest, key=jump, reverse=True)[:n_neg]   # biggest improvement first
        return SelectionResult(positives, secondary, _LABEL["quality"],
                               neg_label if secondary else None)
    return strategy


def _per_lineage_strategy(states, n, params: SelectionParams, exclude_id=None) -> SelectionResult:
    cands = _candidates(states, exclude_id)
    n_pos, n_neg = _split_counts(n, params.mix_fraction)
    positives = _greedy_lineage(sorted(cands, key=quality, reverse=True), n_pos)
    pos_ids = {s.id for s in positives}
    rest = [s for s in cands if s.id not in pos_ids]
    negatives = _greedy_lineage(sorted(rest, key=quality), n_neg) if n_neg > 0 else []
    return SelectionResult(positives, negatives, _LABEL["quality"],
                           "Lower-scoring attempts (one per lineage)" if negatives else None)


def _mmr_strategy(quality_mode: str):
    def strategy(states, n, params: SelectionParams, exclude_id=None) -> SelectionResult:
        cands = _candidates(states, exclude_id)
        qual = _quality_vector(cands, quality_mode, params.jump_alpha)
        positives = _mmr_select(cands, n, qual, params.mmr_lambda)
        return SelectionResult(positives, positives_label=_LABEL["quality"])
    return strategy


def _contrastive_strategy(states, n, params: SelectionParams, exclude_id=None) -> SelectionResult:
    cands = _candidates(states, exclude_id)
    n_pos, n_neg = _split_counts(n, params.mix_fraction)
    qual = _quality_vector(cands, "value", params.jump_alpha)
    positives = _mmr_select(cands, n_pos, qual, params.mmr_lambda)
    pos_ids = {s.id for s in positives}
    pos_min = min((quality(s) for s in positives), default=float("inf"))
    pool = [s for s in cands if s.id not in pos_ids and quality(s) < pos_min]
    negatives = _stratified_diverse(pool, n_neg)
    return SelectionResult(positives, negatives, _LABEL["quality"],
                           "Lower-scoring attempts, spanning low->mid, for contrast" if negatives else None)


# ----------------------------------------------------------------------------- registry
STRATEGIES = {
    "random": _topk_strategy("random"),
    "best": _topk_strategy("quality"),
    "recent": _topk_strategy("recent"),
    "biggest_jump": _topk_strategy("jump"),
    "best_worst": _mix_strategy("worst", "Lower-scoring attempts, for contrast"),
    "best_jump": _mix_strategy("jump", _LABEL["jump"]),
    "per_lineage": _per_lineage_strategy,
    "best_diverse": _mmr_strategy("value"),
    "informative": _mmr_strategy("value_jump"),
    "contrastive": _contrastive_strategy,
}


def get_strategy(name: str):
    """Return the selection function for ``name`` (raises KeyError on unknown)."""
    if name not in STRATEGIES:
        raise KeyError(f"Unknown context strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name]


def select(name: str, states, n, params: SelectionParams | None = None, exclude_id=None) -> SelectionResult:
    """Convenience: run a named strategy with default params if none supplied."""
    return get_strategy(name)(states, n, params or SelectionParams(), exclude_id=exclude_id)


# ----------------------------------------------------------------------------- rendering
def _fenced(code: str, language: str = "python") -> str:
    code = (code or "").strip()
    if not code:
        return "(no code)"
    if code.startswith("```"):
        return code  # already fenced (seed programs / parsed children keep separators)
    return f"```{language}\n{code}\n```"


def _native_score(state: State, maximize: bool) -> float:
    """Score in the problem's native direction (value is stored higher = better)."""
    return state.value if maximize else -state.value


def render_solution(state: State, idx: int, metric_name: str, maximize: bool, language: str = "python",
                    include_code: bool = True, include_strategy: bool = False) -> str:
    """Render one solution. ``include_code`` / ``include_strategy`` pick code, reasoning, or both.

    Guard: if strategy display is requested but the state has no ``<strategy>`` text, we fall back to
    code so a solution is never rendered empty.
    """
    lines = [f"[Solution {idx}] {metric_name} = {_native_score(state, maximize):.6f}"]
    show_strategy = include_strategy and bool((state.strategy or "").strip())
    show_code = include_code or not show_strategy
    if show_strategy:
        lines.append("Strategy:\n" + state.strategy.strip())
    if show_code:
        lines.append(_fenced(state.code, language))
    return "\n".join(lines)


def build_context_block(
    selection,
    metric_name: str = "score",
    maximize: bool = True,
    language: str = "python",
    max_context_tokens: int | None = None,
    include_code: bool = True,
    include_strategy: bool = False,
) -> str:
    """Render a :class:`SelectionResult` (or a plain list of states) into a prompt block.

    Positives and negatives become two labeled sections with continuous ``[Solution N]`` numbering.
    ``max_context_tokens`` caps the block size (chars/4 heuristic), trimming lowest-ranked first while
    keeping at least one solution.
    """
    if isinstance(selection, list):
        selection = SelectionResult(positives=selection)
    if not selection.positives and not selection.negatives:
        return ""

    header = "\n\n--- Past solutions provided as context ---\n"
    footer = "\n--- End of context ---\n"
    budget_chars = None if max_context_tokens is None else int(max_context_tokens * _CHARS_PER_TOKEN)
    used = len(header) + len(footer)

    pieces: list[str] = []
    idx = 1
    n_solutions = 0        # count SOLUTIONS (not section headers) so we always keep at least one
    for label, group in ((selection.positives_label, selection.positives),
                         (selection.negatives_label, selection.negatives)):
        if not group:
            continue
        sec = f"\n[{label}]\n"
        section_open = False
        for s in group:
            piece = "\n" + render_solution(s, idx, metric_name, maximize, language,
                                           include_code, include_strategy) + "\n"
            # cost of adding this solution (plus its section header if the section isn't open yet)
            extra = len(piece) + (0 if section_open else len(sec))
            if budget_chars is not None and n_solutions >= 1 and used + extra > budget_chars:
                return header + "".join(pieces) + footer   # trim the tail; keep what fits
            if not section_open:
                pieces.append(sec)
                used += len(sec)
                section_open = True
            pieces.append(piece)
            used += len(piece)
            n_solutions += 1
            idx += 1

    return header + "".join(pieces) + footer
