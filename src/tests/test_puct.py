"""Unit test for the vendored PUCT buffer — pure (no ray, no tinker, no model server)."""
import sys

from puct import PUCTSampler, State, state_from_dict


class _FakeEnv:
    """Minimal env_type: PUCTSampler only needs create_initial_state + state_type here."""
    state_type = State

    @classmethod
    def create_initial_state(cls, problem_type: str) -> State:
        return State(timestep=-1, construction=[0.0], code="", value=0.0)


def test_import_is_tinker_free():
    assert "tinker" not in sys.modules


def test_sample_update_flush_cycle(tmp_path):
    sampler = PUCTSampler(
        file_path=str(tmp_path / "puct_sampler.json"),
        env_type=_FakeEnv,
        problem_type="test",
        max_buffer_size=100,
        batch_size=1,
        topk_children=2,
    )
    assert len(sampler._states) == 1  # the seeded initial state
    assert sampler._T == 0

    # One generation: sample the parent, add two children with different values.
    parents = sampler.sample_states(1)
    assert len(parents) == 1
    parent = parents[0]

    children = [
        State(timestep=0, construction=[1.0], code="```python\na=1\n```", value=0.5),
        State(timestep=0, construction=[2.0], code="```python\na=2\n```", value=0.9),
    ]
    sampler.update_states(children, [parent, parent], save=False)

    # Buffer grew by the two children; visit count / T advanced for the parent.
    assert len(sampler._states) == 3
    assert sampler._T == 1                       # one parent expanded this generation
    assert sampler._n[parent.id] == 1
    assert sampler._m[parent.id] == 0.9          # best reachable child value

    sampler.flush(step=1)

    # Next selection should now be able to pick the best child (value 0.9).
    picked = sampler.sample_states(1)
    assert picked[0].value is not None

    # Persistence round-trips.
    from puct._json_io import _read_json_or_default
    store = _read_json_or_default(str(tmp_path / "puct_sampler_step_000001.json"), default=None)
    assert store is not None
    assert len(store["states"]) == 3
    reloaded = [state_from_dict(s, state_type=State) for s in store["states"]]
    assert {round(s.value, 3) for s in reloaded} == {0.0, 0.5, 0.9}


def test_failed_rollout_advances_visits():
    sampler = PUCTSampler(
        file_path="/tmp/le_icl_puct_test.json",
        env_type=_FakeEnv,
        problem_type="test",
        batch_size=1,
    )
    parent = sampler.sample_states(1)[0]
    t0 = sampler._T
    sampler.record_failed_rollout(parent)
    assert sampler._T == t0 + 1
    assert sampler._n[parent.id] == 1
