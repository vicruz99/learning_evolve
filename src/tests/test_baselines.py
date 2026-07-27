"""Unit tests for the two no-past-experience baselines — pure (no ray, no model server).

Best-of-N     = ``--parent-source initial --n-context 0``  (no history at all)
TTT w/o RL    = ``--parent-source puct    --n-context 0``  (history only via parent selection)

What is pinned here is the property that makes the first one a *baseline*: parents never come from the
buffer, so no past solution can influence the next proposal.
"""
import numpy as np

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


def test_config_defaults_match_the_cli():
    cfg = ICLConfig(problem="toy", log_path="/tmp/x")
    assert (cfg.parent_source, cfg.seed) == ("puct", None)
