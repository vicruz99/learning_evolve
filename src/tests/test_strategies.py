"""Unit tests for context-selection strategies + rendering (pure; no ray/server)."""
import pytest

from puct import State
from context import (
    get_strategy, select, select_best_n, select_recent_n, build_context_block,
    STRATEGIES, SelectionParams,
)


# --- a small search tree -------------------------------------------------------------------------
#   S (seed, 0.0)
#   ├── A (0.5)
#   │   ├── B (0.9)      B, C are siblings (children of A)
#   │   └── C (0.8)
#   └── D (0.7)          D is a different branch off the seed
def _mk(value, ts, id, parents, parent_values, strategy=""):
    return State(timestep=ts, construction=[value], code=f"# {id}\nx={value}",
                 value=value, id=id, parents=parents, parent_values=parent_values, strategy=strategy)


def _tree():
    S = _mk(0.0, -1, "S", [], [])
    A = _mk(0.5, 0, "A", [{"id": "S", "timestep": -1}], [0.0])
    B = _mk(0.9, 1, "B", [{"id": "A", "timestep": 0}, {"id": "S", "timestep": -1}], [0.5, 0.0])
    C = _mk(0.8, 1, "C", [{"id": "A", "timestep": 0}, {"id": "S", "timestep": -1}], [0.5, 0.0])
    D = _mk(0.7, 1, "D", [{"id": "S", "timestep": -1}], [0.0])
    return [S, A, B, C, D]


def _ids(states):
    return [s.id for s in states]


# --- primitives ----------------------------------------------------------------------------------
def test_primitive_best_and_recent():
    st = _tree()
    assert _ids(select_best_n(st, 2)) == ["B", "C"]
    assert _ids(select_recent_n(st, 2)) in (["B", "C"], ["C", "B"])  # same timestep -> buffer order


def test_primitive_exclude_and_shortfall():
    st = _tree()
    assert "B" not in _ids(select_best_n(st, 5, exclude_id="B"))
    assert len(select_best_n(st, 99)) == 5      # request more than buffer holds -> all


# --- topk engine ---------------------------------------------------------------------------------
def test_best():
    r = select("best", _tree(), 2)
    assert _ids(r.positives) == ["B", "C"] and not r.negatives


def test_biggest_jump():
    # jumps: A=.5, D=.7, B=.4, C=.3, S=0 -> top two by improvement over parent are D, A
    r = select("biggest_jump", _tree(), 2)
    assert _ids(r.positives) == ["D", "A"]


def test_random_is_seeded():
    p = SelectionParams(context_seed=123)
    assert _ids(select("random", _tree(), 3, p).positives) == _ids(select("random", _tree(), 3, p).positives)


# --- mix engine ----------------------------------------------------------------------------------
def test_best_worst_mix():
    r = select("best_worst", _tree(), 4, SelectionParams(mix_fraction=0.5))
    assert _ids(r.positives) == ["B", "C"]      # 2 best
    assert _ids(r.negatives) == ["S", "A"]      # 2 lowest of the rest
    assert r.negatives_label


def test_best_jump_mix():
    r = select("best_jump", _tree(), 4, SelectionParams(mix_fraction=0.5))
    assert _ids(r.positives) == ["B", "C"]
    assert _ids(r.negatives) == ["D", "A"]      # biggest-improvement of the rest


# --- per_lineage engine --------------------------------------------------------------------------
def test_per_lineage_skips_ancestors():
    # Greedy on quality: pick B; C is a sibling (not same path) -> kept; D is a separate branch -> kept;
    # A and S are ANCESTORS of B, so they are skipped.
    r = select("per_lineage", _tree(), 3, SelectionParams(mix_fraction=1.0))
    assert _ids(r.positives) == ["B", "C", "D"]
    assert "A" not in _ids(r.positives) and "S" not in _ids(r.positives)


# --- mmr engine ----------------------------------------------------------------------------------
def test_best_diverse_lambda_extremes():
    st = _tree()
    # lambda=1 -> pure quality: B then C (the sibling, higher value)
    assert _ids(select("best_diverse", st, 2, SelectionParams(mmr_lambda=1.0)).positives) == ["B", "C"]
    # lambda=0 -> pure spread: B then the FARTHEST node in the tree (the separate branch D), not the sibling
    assert _ids(select("best_diverse", st, 2, SelectionParams(mmr_lambda=0.0)).positives) == ["B", "D"]


def test_informative_uses_jump():
    # With alpha=0 the MMR quality term is pure improvement-over-parent, so the biggest jumper (D) leads.
    r = select("informative", _tree(), 1, SelectionParams(jump_alpha=0.0, mmr_lambda=1.0))
    assert _ids(r.positives) == ["D"]


def test_contrastive_negatives_below_positives():
    r = select("contrastive", _tree(), 4, SelectionParams(mix_fraction=0.5, mmr_lambda=0.7))
    assert r.positives
    pos_min = min(s.value for s in r.positives)
    assert all(s.value < pos_min for s in r.negatives)


# --- rendering flags -----------------------------------------------------------------------------
def test_render_code_only_default():
    block = build_context_block(select_best_n(_tree(), 2), metric_name="score", maximize=True)
    assert "```python" in block and "Strategy:" not in block
    assert "score" in block


def test_render_strategy_only():
    st = [_mk(0.9, 1, "B", [{"id": "A", "timestep": 0}], [0.5], strategy="pack tighter near corners")]
    block = build_context_block(st, include_code=False, include_strategy=True)
    assert "Strategy:" in block and "pack tighter near corners" in block
    assert "```python" not in block


def test_render_strategy_falls_back_to_code_when_absent():
    # strategy-only requested but the state has no <strategy> text -> must still render code (never empty)
    block = build_context_block(select_best_n(_tree(), 1), include_code=False, include_strategy=True)
    assert "```python" in block


def test_budget_keeps_at_least_one_solution():
    # A tiny token budget must still yield >=1 rendered solution (not just a section header).
    big = [_mk(0.9 - 0.01 * i, 1, f"X{i}", [{"id": "A", "timestep": 0}], [0.5]) for i in range(20)]
    tiny = build_context_block(select_best_n(big, 20), metric_name="score", max_context_tokens=40)
    assert "[Solution 1]" in tiny
    full = build_context_block(select_best_n(big, 20), metric_name="score")
    assert len(tiny) < len(full)                 # and it genuinely trimmed the tail


def test_budget_no_dangling_negative_header():
    # If the budget is exhausted by positives, the negatives section header must not appear alone.
    st = _tree()
    r = select("best_worst", st, 4, SelectionParams(mix_fraction=0.5))
    blk = build_context_block(r, metric_name="score", max_context_tokens=30)
    assert "[Solution 1]" in blk
    assert "Lower-scoring attempts" not in blk or "[Solution 3]" in blk  # header only if a neg follows


def test_two_block_numbering_is_continuous():
    r = select("best_worst", _tree(), 4, SelectionParams(mix_fraction=0.5))
    block = build_context_block(r)
    for n in ("[Solution 1]", "[Solution 2]", "[Solution 3]", "[Solution 4]"):
        assert n in block


# --- strategy-text capture (schema + parser) -----------------------------------------------------
def test_parse_strategy_block_and_roundtrip():
    from envs.base import parse_strategy_block
    completion = ("<strategy>place 4 corner circles then greedily fill</strategy>\n"
                  "```python\ndef run(): return 1\n```")
    strat = parse_strategy_block(completion)
    assert strat == "place 4 corner circles then greedily fill"
    assert parse_strategy_block("no tags here") == ""
    # survives State (de)serialization
    s = State(timestep=0, construction=[1], code="x=1", value=0.5, strategy=strat)
    assert State.from_dict(s.to_dict()).strategy == strat


# --- registry ------------------------------------------------------------------------------------
def test_registry_has_all_strategies():
    assert set(STRATEGIES) == {
        "random", "best", "recent", "biggest_jump", "best_worst", "best_jump",
        "per_lineage", "best_diverse", "informative", "contrastive",
    }
    with pytest.raises(KeyError):
        get_strategy("nope")
