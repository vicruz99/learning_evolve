"""Unit tests for context-selection strategies (pure; no ray/server)."""
import pytest

from puct import State
from context import get_strategy, select_best_n, select_recent_n, STRATEGIES


def _s(value, timestep):
    return State(timestep=timestep, construction=[value], code="", value=value)


def test_best_orders_by_value():
    states = [_s(0.1, 0), _s(0.9, 1), _s(0.5, 2)]
    assert [s.value for s in select_best_n(states, 2)] == [0.9, 0.5]


def test_recent_orders_by_timestep():
    states = [_s(0.1, 0), _s(0.9, 1), _s(0.5, 2)]
    assert [s.timestep for s in select_recent_n(states, 2)] == [2, 1]


def test_exclude_id():
    a = _s(0.9, 1)
    out = select_best_n([a, _s(0.5, 2)], 5, exclude_id=a.id)
    assert a not in out and len(out) == 1


def test_shortfall_returns_all_available():
    """Requesting more than the buffer holds returns all available (graceful, no error)."""
    states = [_s(0.1, 0), _s(0.5, 1)]
    assert len(select_best_n(states, 5)) == 2
    assert len(select_recent_n(states, 5)) == 2


def test_registry_and_stubs():
    assert get_strategy("best") is select_best_n
    assert get_strategy("recent") is select_recent_n
    assert set(STRATEGIES) >= {"best", "recent", "diverse"}
    with pytest.raises(KeyError):
        get_strategy("nope")
    with pytest.raises(NotImplementedError):
        get_strategy("diverse")([], 3)
