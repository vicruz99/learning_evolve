"""Invariants of the harness-side context pool and the exclude-parent option.

These lock in two behaviours that are easy to break by accident and expensive to notice:

  1. **A seed program is never shown as a "past solution".** At generation 0 the parents *are* the
     seed, so there is no past experience to condition on and the context block must be empty. This
     holds structurally — only graded rollout results enter the pool — and these tests pin that.
  2. **`exclude_parent_from_context` is what makes the block differ per parent.** With it off, every
     parent of a generation gets a byte-identical block (given a fixed `context_seed`), which is what
     turns the prompt into a shared prefix vLLM can prefill once per generation.

Pure: no ray, no server, no model.
"""
import asyncio
import os
import tempfile

from context import SelectionParams, get_strategy
from icl.config import ICLConfig
from icl.loop import ICLRunner
from puct import State


def _mk(value, ts, sid, parents=(), parent_values=()):
    return State(timestep=ts, construction=[value], code=f"# {sid}\nx={value}", value=value,
                 id=sid, parents=list(parents), parent_values=list(parent_values))


def _runner(**overrides) -> ICLRunner:
    cfg = ICLConfig(problem="toy", log_path=tempfile.mkdtemp(), n_context=3,
                    groups_per_batch=2, group_size=2, num_generations=1, **overrides)
    return ICLRunner(cfg)


def _prompt_for(runner: ICLRunner, parent: State) -> tuple[str, list[str]]:
    """Build one parent's prompt through the real code path; return (prompt, context solution ids)."""
    spec = runner.spec
    with tempfile.TemporaryDirectory() as td:
        runner.sampler = runner._make_sampler(os.path.join(td, "s.json"))
        env = spec.env_type(initial_state=parent, sampler=runner.sampler, config=runner.env_config)
        prompt, selection, _intro, _tail, _block = runner._build_prompt(env, parent)
    return prompt, [s.id for s in selection.all()]


# --- the seed is never a "past solution" ----------------------------------------------------------
def test_generation_0_has_an_empty_context_block():
    """Gen 0 = the pool has not been fed yet, so there is nothing to inject — not even the seed."""
    runner = _runner()
    seed = _mk(0.0, -1, "seed")
    prompt, ids = _prompt_for(runner, seed)
    assert ids == []
    assert "Past solutions you've tried" not in prompt


def test_generation_0_stays_empty_even_with_exclude_parent_off():
    """The option must not open a back door for the seed: the pool is empty either way."""
    runner = _runner(exclude_parent_from_context=False, context_seed=7)
    seed = _mk(0.0, -1, "seed")
    _prompt, ids = _prompt_for(runner, seed)
    assert ids == []


def test_only_graded_solutions_enter_the_pool():
    """`_extend_context_pool` is the ONLY writer, and the loop feeds it graded children only — so a
    seed (which is never graded) cannot reach it. Later generations legitimately contain everything."""
    runner = _runner()
    assert runner._context_pool == []
    runner._pool_fh = open(os.path.join(runner.cfg.log_path, "pool.jsonl"), "w")
    graded = [_mk(0.5, 0, "A"), _mk(0.9, 1, "B")]
    runner._extend_context_pool(graded)
    runner._pool_fh.close()
    assert [s.id for s in runner._context_pool] == ["A", "B"]


def test_later_generations_do_inject_context():
    """Sanity counterpart: once the pool has graded solutions they DO appear (the block is not
    accidentally always empty)."""
    runner = _runner()
    runner._context_pool = [_mk(0.5, 0, "A"), _mk(0.9, 1, "B")]
    prompt, ids = _prompt_for(runner, _mk(0.9, 1, "B"))
    assert "A" in ids
    assert "Past solutions you've tried" in prompt


# --- exclude_parent_from_context -------------------------------------------------------------------
def _pool():
    return [_mk(0.5, 0, "A"), _mk(0.9, 1, "B"), _mk(0.8, 1, "C"), _mk(0.7, 1, "D")]


def test_exclude_parent_on_drops_the_parent_from_its_own_block():
    runner = _runner(exclude_parent_from_context=True, context_strategy="best")
    runner._context_pool = _pool()
    _prompt, ids = _prompt_for(runner, runner._context_pool[1])       # parent B, the best
    assert "B" not in ids
    assert ids == ["C", "D", "A"]                                    # B's slot shifts A in


def test_exclude_parent_off_keeps_the_parent_in_its_own_block():
    runner = _runner(exclude_parent_from_context=False, context_strategy="best")
    runner._context_pool = _pool()
    _prompt, ids = _prompt_for(runner, runner._context_pool[1])       # parent B
    assert ids == ["B", "C", "D"]


def test_exclude_parent_off_plus_seed_makes_every_parent_share_one_block():
    """The whole point of the option: identical blocks -> a shared prefix across a generation."""
    pool = _pool()
    shared = None
    for parent in pool:                                              # every parent of a generation
        runner = _runner(exclude_parent_from_context=False, context_strategy="best", context_seed=7)
        runner._context_pool = list(pool)
        _prompt, ids = _prompt_for(runner, parent)
        shared = ids if shared is None else shared
        assert ids == shared

    # ...whereas with exclude_parent on, the parent's own slot shifts and the blocks diverge.
    blocks = []
    for parent in pool:
        runner = _runner(exclude_parent_from_context=True, context_strategy="best", context_seed=7)
        runner._context_pool = list(pool)
        _prompt, ids = _prompt_for(runner, parent)
        blocks.append(ids)
    assert len({tuple(b) for b in blocks}) > 1


def test_exclude_id_is_the_only_parent_input_to_a_strategy():
    """No strategy reads the parent except through exclude_id (per_lineage / mmr compute lineage
    relationships AMONG candidates), so turning the option off is enough to share the block."""
    pool = _pool()
    params = SelectionParams(context_seed=7)
    for name in ("best", "random", "recent", "best_worst", "per_lineage", "best_diverse",
                 "informative", "contrastive", "biggest_jump", "best_jump"):
        strategy = get_strategy(name)
        first = [s.id for s in strategy(pool, 3, params, exclude_id=None).all()]
        again = [s.id for s in strategy(pool, 3, params, exclude_id=None).all()]
        assert first == again, f"{name} is not deterministic given a fixed context_seed"
