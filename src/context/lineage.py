"""Lineage / tree-structure helpers for diversity-aware context selection.

Every ``State`` carries its full ancestor chain in ``state.parents`` (a list of ``{"id","timestep"}``
dicts, most-recent-first) and the matching ``state.parent_values``. From that we can reconstruct the
search *tree* and reason about how "close" two solutions are:

- **same lineage (a path relationship):** one is an ancestor/descendant of the other -> they sit on
  the same root-to-node path. Used by the ``per_lineage`` strategy's hard skip.
- **tree-edge distance:** number of edges between two nodes via their most-recent common ancestor
  (parent-child = 1, siblings = 2, cousins = 4, ...). Used by the ``mmr`` strategies as a *soft*
  similarity ``sim = 1 / (1 + dist)``.

Both are computed purely from the stored ancestor ids -- no global tree object needed.
"""
from __future__ import annotations

from puct.state import State


def ancestor_ids(state: State) -> list[str]:
    """``[state.id, parent.id, grandparent.id, ...]`` -- the node plus its chain to the seed.

    Order is most-recent-first (the node itself, then successively older ancestors), matching the
    order ``state.parents`` is stored in. Depth of ``state`` == ``len(...) - 1``.
    """
    ids = [state.id]
    ids.extend(str(p["id"]) for p in (state.parents or []) if p.get("id"))
    return ids


def same_lineage(a: State, b: State) -> bool:
    """True iff ``a`` and ``b`` lie on the same root-to-node path (one is ancestor of the other).

    Note this is the *path* relationship the user specified for ``per_lineage``: siblings/cousins
    (which merely share a common ancestor) are NOT the same lineage. Use :func:`tree_distance` if
    you want to also account for horizontal (sibling) closeness.
    """
    if a.id == b.id:
        return True
    return a.id in set(ancestor_ids(b)) or b.id in set(ancestor_ids(a))


def _depth(state: State) -> int:
    return len(ancestor_ids(state)) - 1


def tree_distance(a: State, b: State) -> int:
    """Number of tree edges between ``a`` and ``b`` (parent-child=1, siblings=2, cousins=4, ...).

    ``dist = depth(a) + depth(b) - 2*depth(MRCA)`` where MRCA is the deepest shared ancestor. If the
    two share no ancestor at all (different seeds), they are effectively in disjoint trees; we return
    ``depth(a) + depth(b) + 2`` so cross-seed pairs read as maximally far.
    """
    if a.id == b.id:
        return 0
    anc_a = ancestor_ids(a)          # most-recent-first, so index == distance up from the node
    anc_b = ancestor_ids(b)
    depth_a, depth_b = len(anc_a) - 1, len(anc_b) - 1
    set_b = set(anc_b)
    # anc_a is ordered nearest-first, so the first hit is the deepest (most-recent) common ancestor.
    for i, aid in enumerate(anc_a):
        if aid in set_b:
            depth_mrca = depth_a - i
            return (depth_a - depth_mrca) + (depth_b - depth_mrca)
    return depth_a + depth_b + 2     # disjoint trees (different seeds)


def lineage_similarity(a: State, b: State) -> float:
    """Soft closeness in ``[0, 1]``: ``1 / (1 + tree_distance)`` (1.0 = same node, ->0 far apart)."""
    return 1.0 / (1.0 + tree_distance(a, b))
