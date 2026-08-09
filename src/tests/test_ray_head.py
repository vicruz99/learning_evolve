"""Unit tests for how the Ray head is SIZED (pure; starts nothing).

Every probe plan_head() makes is a machine measurement, so the whole file runs against a fake box:
64 physical cores / 128 logical, 256 G of RAM, no cgroup or LSF ceiling. That is the only way these
assertions mean the same thing on the laptop, on guadiana and inside an LSF job.
"""
import pytest

from sandbox import ray_head
from sandbox.ray_head import GiB, plan_head

CORES, LOGICAL, NODE_RAM = 64, 128, 256 * GiB


@pytest.fixture(autouse=True)
def fake_box(monkeypatch):
    """A machine that grants us all 64 of its cores and confines nothing."""
    monkeypatch.delenv("LSB_DJOB_NUMPROC", raising=False)
    monkeypatch.delenv("LSB_JOBID", raising=False)
    monkeypatch.setattr(ray_head, "granted_cores", lambda: (CORES, LOGICAL))
    monkeypatch.setattr(ray_head, "machine_cores", lambda: CORES)
    monkeypatch.setattr(ray_head, "_cgroup_cpu_limits", lambda: (None, "no quota", None))
    monkeypatch.setattr(ray_head, "check_smt_grouping", lambda _tpt: None)
    monkeypatch.setattr(ray_head, "_cgroup_memory_limit", lambda: (None, "no cgroup limit"))
    monkeypatch.setattr(ray_head, "_lsf_memlimit", lambda: (None, "not an LSF job"))
    monkeypatch.setattr(ray_head, "_meminfo_total", lambda: NODE_RAM)
    monkeypatch.setattr(ray_head, "_shm_free", lambda: NODE_RAM // 2)


def _joined(plan) -> str:
    return " | ".join(plan.notes + plan.warnings)


# ---- the override itself -------------------------------------------------------------------------
def test_num_cpus_override_is_the_final_count_with_no_reserve_subtracted():
    """The point of the knob: "let Ray see 16" means 16, not 16 minus a driver reserve."""
    plan = plan_head(threads_per_task=1, max_parallel=4, cfg={"num_cpus": 16})
    assert plan.num_cpus == 16
    assert "OVERRIDDEN to 16" in _joined(plan)
    assert not plan.warnings                      # 16 <= 64 granted: nothing to warn about
    assert "--num-cpus=16" in plan.argv()


def test_without_the_override_the_reserve_still_applies():
    """The detected path is untouched: 64 granted - (1 supervisor + 2 drivers) = 61 -> 60 at tpt=2."""
    plan = plan_head(threads_per_task=2, max_parallel=2, cfg={})
    assert plan.num_cpus == 60
    assert "reserved 3 cpu(s)" in _joined(plan)


def test_override_beats_reserve_cpus_when_both_are_set():
    plan = plan_head(threads_per_task=1, max_parallel=1, cfg={"num_cpus": 8, "reserve_cpus": 32})
    assert plan.num_cpus == 8


def test_override_rounds_down_to_whole_eval_slots():
    """cpu_scheduler drops the trailing partial group, so 17 cpus at 2/task can only back 16."""
    plan = plan_head(threads_per_task=2, max_parallel=1, cfg={"num_cpus": 17})
    assert plan.num_cpus == 16
    assert "rounded down to a multiple of 2" in _joined(plan)


def test_override_below_one_eval_slot_is_refused():
    with pytest.raises(ValueError, match="less than one 4-cpu eval slot"):
        plan_head(threads_per_task=4, max_parallel=1, cfg={"num_cpus": 3})


# ---- warnings when the operator asks for more than the box grants --------------------------------
def test_override_above_the_affinity_mask_warns_about_cpu_starvation():
    """Ray admits num_cpus/tpt tasks; cpu_scheduler can only build affinity/tpt groups. The surplus
    spins in get_cpu_group() until it fails, which is not obvious from anything Ray reports."""
    plan = plan_head(threads_per_task=1, max_parallel=1, cfg={"num_cpus": 200})
    assert plan.num_cpus == 200
    assert any("cpu_starvation" in w and "128 logical CPU" in w for w in plan.warnings)


def test_override_above_the_physical_cores_warns_about_sharing_cores():
    """Between cores and logical CPUs there is no starvation -- just two evals per core at half
    speed, which silently shifts runtimes and timeout rates."""
    plan = plan_head(threads_per_task=1, max_parallel=1, cfg={"num_cpus": 96})
    assert plan.num_cpus == 96
    assert any("half speed" in w for w in plan.warnings)
    assert not any("cpu_starvation" in w for w in plan.warnings)


# ---- what the override implies for memory --------------------------------------------------------
def test_memory_fair_share_follows_the_override_not_the_grant():
    """Taking 16 of 64 cores must not still claim the whole node's RAM: with no cgroup limit, Ray's
    heap is a fair share of MemTotal, and the fair share of a hand-picked core count is that count."""
    pinned = plan_head(threads_per_task=1, max_parallel=1, cfg={"num_cpus": 16})
    full = plan_head(threads_per_task=1, max_parallel=1, cfg={})
    assert pinned.memory_bytes < full.memory_bytes
    assert "16/64 physical cores (from sweep.ray.num_cpus)" in _joined(pinned)
    # ... but still enough heap for every slot it admits, or memory becomes the binding constraint.
    assert pinned.memory_bytes >= 16 * ray_head.TASK_MEMORY


def test_explicit_memory_gb_still_wins_over_the_derived_share():
    plan = plan_head(threads_per_task=1, max_parallel=1, cfg={"num_cpus": 16, "memory_gb": 8})
    assert plan.memory_bytes == 8 * GiB
