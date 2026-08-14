"""Unit tests for the parent sources — pure (no ray, no model server).

Best-of-N     = ``--parent-source initial --n-context 0``  (no history at all)
TTT w/o RL    = ``--parent-source puct    --n-context 0``  (history only via parent selection)
greedy        = ``--parent-source best``                   (history only as best-so-far)
context-only  = ``--parent-source none``                   (history only through the prompt block)

What is pinned here is the property that makes each of them the arm it claims to be: which solution
(if any) reaches the model as "the current solution to improve upon".
"""
import numpy as np

from envs.base import EnvConfig
from icl.config import ICLConfig
from puct import PUCTSampler, State
from run_icl import build_parser


class _FakeEnv:
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(timestep=-1, construction=[0.0], code="SEED", value=0.0)


class _FakeAcEnv(_FakeEnv):
    """An env with construction_length_limits, which is what triggers the random AC construction."""
    construction_length_limits = (1000, 8000)


def _sampler(tmp_path, env_type=_FakeEnv, **kw) -> PUCTSampler:
    tmp_path.mkdir(parents=True, exist_ok=True)          # callers pass fresh subdirs for isolation
    return PUCTSampler(file_path=str(tmp_path / "puct_sampler.json"), env_type=env_type,
                       problem_type="test", max_buffer_size=100, batch_size=2, **kw)


# ---- Best-of-N: parents always come from the seed, never from the buffer -------------------------
def test_sample_initial_states_ignores_a_populated_buffer(tmp_path):
    sampler = _sampler(tmp_path)
    parent = sampler.sample_states(1)[0]
    # A high-scoring child that PUCT would certainly pick next.
    child = State(timestep=0, construction=[1.0], code="EVOLVED", value=99.0)
    sampler.update_states([child], [parent], save=False)
    assert any(s.code == "EVOLVED" for s in sampler._states)          # it is in the buffer

    picked = sampler.sample_initial_states(3)
    assert len(picked) == 3
    assert all(p.code == "SEED" for p in picked)                       # ...and still not chosen
    assert all(p.value == 0.0 for p in picked)
    # Contrast: the PUCT source *would* have taken it.
    assert sampler.sample_states(1)[0].code == "EVOLVED"


def test_sample_initial_states_returns_distinct_fresh_objects(tmp_path):
    sampler = _sampler(tmp_path)
    picked = sampler.sample_initial_states(3)
    assert len({id(p) for p in picked}) == 3
    assert len({p.id for p in picked}) == 3          # distinct ids: per-parent bookkeeping stays sane


def test_sample_initial_states_keeps_tracker_state_consistent(tmp_path):
    """The tracker reads _last_sampled_* / _last_puct_stats after every selection."""
    sampler = _sampler(tmp_path)
    picked = sampler.sample_initial_states(2)
    assert sampler._last_sampled_states == picked
    assert len(sampler._last_puct_stats) == 2
    assert sampler.get_sample_stats().get("puct/buffer_size") == len(sampler._states)


def test_best_of_n_still_records_the_buffer(tmp_path):
    """Best-of-N must not read the buffer, but it must keep writing it — best-so-far comes from there."""
    sampler = _sampler(tmp_path)
    parents = sampler.sample_initial_states(2)
    kids = [State(timestep=0, construction=[float(i)], code=f"C{i}", value=float(i))
            for i in (1, 2)]
    sampler.update_states(kids, parents, save=False)
    assert max(s.value for s in sampler._states) == 2.0


# ---- greedy: parents are always the buffer's best-so-far -----------------------------------------
def test_sample_best_states_returns_the_top_state_for_every_slot(tmp_path):
    sampler = _sampler(tmp_path)
    parent = sampler.sample_states(1)[0]
    kids = [State(timestep=0, construction=[float(i)], code=f"C{i}", value=float(i)) for i in (1, 5, 3)]
    sampler.update_states(kids, [parent] * 3, save=False)

    picked = sampler.sample_best_states(4)
    assert len(picked) == 4
    assert {p.code for p in picked} == {"C5"}
    # One state, one identity: PUCT bookkeeping must accumulate on it rather than on four clones.
    assert len({p.id for p in picked}) == 1


def test_sample_best_states_falls_back_to_the_seed_on_an_empty_buffer(tmp_path):
    sampler = _sampler(tmp_path)
    sampler._states = []
    picked = sampler.sample_best_states(2)
    assert [p.code for p in picked] == ["SEED", "SEED"]


def test_sample_best_states_is_greedy_where_puct_explores(tmp_path):
    """The point of the arm: no exploration bonus, so a visited best stays the parent."""
    sampler = _sampler(tmp_path)
    parent = sampler.sample_states(1)[0]
    top = State(timestep=0, construction=[9.0], code="TOP", value=9.0)
    mid = State(timestep=0, construction=[8.0], code="MID", value=8.0)
    sampler.update_states([top, mid], [parent, parent], save=False)
    # Expand TOP enough times: PUCT's 1/(1+n) term eventually pushes selection elsewhere...
    for _ in range(30):
        sampler.record_failed_rollout(top)
    assert sampler.sample_states(1)[0].code != "TOP"
    # ...while `best` keeps handing back the same solution.
    assert sampler.sample_best_states(1)[0].code == "TOP"


def test_sample_best_states_keeps_tracker_state_consistent(tmp_path):
    sampler = _sampler(tmp_path)
    picked = sampler.sample_best_states(3)
    assert sampler._last_sampled_states == picked
    assert len(sampler._last_puct_stats) == 3
    assert len(sampler._last_sampled_indices) == 3


# ---- no parent: the prompt shows no current solution ---------------------------------------------
def test_env_config_shows_the_parent_solution_by_default():
    assert EnvConfig(problem_type="26", log_path="/tmp").show_parent_solution is True


def test_no_parent_source_drops_the_current_solution_from_the_prompt():
    """The whole point of `none`: no parent code, no parent value, no improve-upon framing —
    but the objective and the target survive, so the arm differs in one thing only."""
    from envs.circle_packing import CirclePackingEnv

    parent = State(timestep=0, construction=[], code="def run_packing(): pass", value=2.1)
    def _tail(show: bool) -> str:
        cfg = EnvConfig(problem_type="26", log_path="/tmp", show_parent_solution=show)
        return CirclePackingEnv(initial_state=parent, sampler=object(), config=cfg).improvement_task()

    shown, hidden = _tail(True), _tail(False)
    assert "def run_packing(): pass" in shown and "improve upon" in shown
    assert "def run_packing(): pass" not in hidden
    assert "improve upon" not in hidden.lower()
    assert "2.636" in hidden                      # the target still reaches the model
    assert "run_packing function" in hidden       # ...and so do the rules


def test_no_parent_tail_is_identical_across_parents():
    """With no trailing solution the tail is parent-independent, so the WHOLE prompt is a shared
    prefix — the prefix-caching contract of envs/base.improvement_task, taken to its limit."""
    from envs.circle_packing import CirclePackingEnv

    cfg = EnvConfig(problem_type="26", log_path="/tmp", show_parent_solution=False)
    tails = [CirclePackingEnv(initial_state=State(timestep=0, construction=[], code=f"code {i}",
                                                  value=float(i)),
                              sampler=object(), config=cfg).improvement_task()
             for i in range(3)]
    assert len(set(tails)) == 1


# ---- seeding: the sampler's only stochastic surface is the AC initial construction ----------------
def test_seed_makes_the_ac_initial_construction_reproducible(tmp_path):
    a = _sampler(tmp_path / "a", env_type=_FakeAcEnv, rng_seed=7).sample_initial_states(2)
    b = _sampler(tmp_path / "b", env_type=_FakeAcEnv, rng_seed=7).sample_initial_states(2)
    c = _sampler(tmp_path / "c", env_type=_FakeAcEnv, rng_seed=8).sample_initial_states(2)
    assert [len(s.construction) for s in a] == [len(s.construction) for s in b]
    assert np.allclose([s.construction[0] for s in a], [s.construction[0] for s in b])
    # A different seed gives a different replicate (both length and value are drawn).
    assert ([len(s.construction) for s in c], [s.construction[0] for s in c]) != \
           ([len(s.construction) for s in a], [s.construction[0] for s in a])


def test_unseeded_sampler_still_works(tmp_path):
    picked = _sampler(tmp_path, env_type=_FakeAcEnv).sample_initial_states(1)
    assert 1000 <= len(picked[0].construction) <= 8000


# ---- CLI plumbing --------------------------------------------------------------------------------
def test_cli_exposes_the_baseline_flags():
    a = build_parser().parse_args(["--problem", "toy", "--parent-source", "initial",
                                  "--seed", "3", "--n-context", "0"])
    assert (a.parent_source, a.seed, a.n_context) == ("initial", 3, 0)


def test_cli_defaults_to_puct_and_no_seed():
    a = build_parser().parse_args(["--problem", "toy"])
    assert (a.parent_source, a.seed) == ("puct", None)


def test_cli_exposes_every_parent_source():
    for source in ("puct", "initial", "best", "none"):
        assert build_parser().parse_args(
            ["--problem", "toy", "--parent-source", source]).parent_source == source


# ---- loop wiring: parent_source picks both the sampler call AND the prompt shape ------------------
def test_loop_dispatches_each_parent_source_to_the_right_sampler_call():
    from icl.loop import ICLRunner

    class _SpySampler:
        def __init__(self):
            self.calls = []

        def __getattr__(self, name):
            def _record(n):
                self.calls.append((name, n))
                return []
            return _record

    for source, expected in (("puct", "sample_states"), ("initial", "sample_initial_states"),
                             ("best", "sample_best_states"), ("none", "sample_initial_states")):
        runner = ICLRunner.__new__(ICLRunner)
        runner.cfg = ICLConfig(problem="toy", log_path="/tmp/x", parent_source=source)
        runner.sampler = _SpySampler()
        runner._sample_parents(4)
        assert runner.sampler.calls == [(expected, 4)], source


def test_only_the_none_source_hides_the_parent_solution():
    from icl.loop import ICLRunner

    for source in ("puct", "initial", "best"):
        assert ICLRunner(ICLConfig(problem="toy", log_path="/tmp/x",
                                   parent_source=source)).env_config.show_parent_solution is True
    assert ICLRunner(ICLConfig(problem="toy", log_path="/tmp/x",
                               parent_source="none")).env_config.show_parent_solution is False


def test_config_defaults_match_the_cli():
    cfg = ICLConfig(problem="toy", log_path="/tmp/x")
    assert (cfg.parent_source, cfg.seed) == ("puct", None)
