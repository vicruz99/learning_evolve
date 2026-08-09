"""Unit tests for verified resume points (pure; no ray/server/LLM).

The bug these pin down: --resume read summary.json's ``status`` and nothing else, so a run whose
generations had been deleted was reported "complete" and skipped, and a run resumed once had its
summary rewritten from an empty per-generation list — after which it claimed fewer generations than it
had done, or claimed to be complete at 3 of 15.
"""
import csv
import json
import os

import pytest

from results.resume import inspect_run, prior_state, rewind, tail_exists


# ---- a run directory, written the way ExperimentTracker writes one --------------------------------
def _write_run(root, gens: int, *, want: int = 3, groups: int = 2, group_size: int = 2,
               n_context: int = 4, status: str = "complete", summary_gens: int | None = None,
               valid_per_gen: int = 2, drop_group_in: int | None = None,
               snapshot_steps: "list[int] | None" = None, pool_lines: int | None = None) -> str:
    """A run dir with ``gens`` finished generations. ``drop_group_in`` reproduces the LLM-failure
    shape: icl.loop swallows the error and records a group with no children."""
    run_dir = str(root)
    os.makedirs(os.path.join(run_dir, "solutions"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "buffer"), exist_ok=True)
    json.dump({"num_generations": want, "groups_per_batch": groups, "group_size": group_size,
               "n_context": n_context, "_meta": {"created_at": "2026-01-01T00:00:00"}},
              open(os.path.join(run_dir, "config.json"), "w"))

    per_gen, events, sols, progress = [], [], [], []
    sol_seq = 0
    for g in range(gens):
        gen_dir = os.path.join(run_dir, "generations", f"gen_{g:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        parents = []
        for slot in range(groups):
            children = [] if (g == drop_group_in and slot == 0) else [
                {"child": c, "correctness": 1.0 if c < valid_per_gen else 0.0} for c in range(group_size)
            ]
            parents.append({"slot": slot, "parent_sol": "seed", "children": children})
        for _ in range(valid_per_gen):
            sol_seq += 1
            sol = f"sol_{sol_seq:06d}"
            sols.append({"sol": sol, "state_id": f"st{sol_seq}", "gen": g, "raw_score": 1.0 + sol_seq,
                         "value": 1.0 + sol_seq, "correctness": 1.0})
            open(os.path.join(run_dir, "solutions", sol + ".py"), "w").write("# code\n")
        stats = {"generation": g, "valid_candidates": valid_per_gen,
                 "failed_candidates": groups * group_size - valid_per_gen,
                 "gen_best_score": 1.0 + sol_seq, "best_so_far_score": 1.0 + sol_seq,
                 "wall_seconds": 10.0, "failure_types": {"invalid": 1},
                 "usage": {"requests": 1, "completions": groups * group_size,
                           "prompt_tokens": 100, "completion_tokens": 200}}
        json.dump({"generation": g, "stats": stats, "parents": parents},
                  open(os.path.join(gen_dir, "meta.json"), "w"))
        per_gen.append(stats)
        events += [{"generation": g, "child": c} for c in range(groups * group_size)]
        progress.append([g, valid_per_gen])

    with open(os.path.join(run_dir, "events.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    with open(os.path.join(run_dir, "solutions", "manifest.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in sols)
    with open(os.path.join(run_dir, "progress.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["generation", "valid_candidates"])
        w.writerows(progress)
    n_pool = len(sols) if pool_lines is None else pool_lines
    with open(os.path.join(run_dir, "buffer", "context_pool.jsonl"), "w") as f:
        f.writelines(json.dumps({"id": f"st{i + 1}"}) + "\n" for i in range(n_pool))
    for step in (snapshot_steps if snapshot_steps is not None else [gens]):
        json.dump({"step": step, "states": [{"id": "s"}]},
                  open(os.path.join(run_dir, "buffer", f"puct_sampler_step_{step:06d}.json"), "w"))
    json.dump({"status": status, "best": {"score": 1.0 + sol_seq, "rank_value": 1.0 + sol_seq},
               "updated_at": "2026-01-02T00:00:00", "totals": {},
               "per_generation": per_gen[:summary_gens] if summary_gens is not None else per_gen},
              open(os.path.join(run_dir, "summary.json"), "w"))
    return run_dir


# ---- what "complete" means ------------------------------------------------------------------------
def test_a_finished_run_is_complete(tmp_path):
    prog = inspect_run(_write_run(tmp_path / "r", 3, want=3))
    assert prog.complete and prog.good_generations == 3 and prog.resume_step == 3
    assert prog.damage == []


def test_deleting_the_data_under_a_complete_summary_is_not_complete(tmp_path):
    """The reported bug: the run dir was cleared, summary.json still said complete, --resume skipped
    it. Completeness now has to be backed by the generations themselves."""
    run_dir = _write_run(tmp_path / "r", 3, want=3)
    import shutil
    shutil.rmtree(os.path.join(run_dir, "generations"))
    shutil.rmtree(os.path.join(run_dir, "buffer"))
    prog = inspect_run(run_dir)
    assert not prog.complete
    assert prog.good_generations == 0 and prog.resume_step == 0
    assert prog.summary_status == "complete"                       # the lie is still on disk


def test_a_missing_run_dir_is_a_full_restart(tmp_path):
    prog = inspect_run(str(tmp_path / "gone"), num_generations=3)
    assert not prog.complete and prog.resume_step == 0 and not prog.exists


def test_summary_rewritten_by_an_older_resume_does_not_shorten_the_run(tmp_path):
    """15 generations on disk, a summary claiming 3 (what the old tracker wrote after a resume):
    the run is complete and must not be rewound to 3."""
    run_dir = _write_run(tmp_path / "r", 15, want=15, summary_gens=3)
    prog = inspect_run(run_dir)
    assert prog.complete and prog.good_generations == 15
    assert any("summary.json records 3" in d for d in prog.damage)


# ---- where a resume picks up ----------------------------------------------------------------------
def test_a_partial_trailing_generation_is_not_resumable_past(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[3])
    os.makedirs(os.path.join(run_dir, "generations", "gen_0003", "parent_00"))   # died inside gen 3
    prog = inspect_run(run_dir)
    assert prog.good_generations == 3 and prog.resume_step == 3 and not prog.complete
    assert any("gen 3: no meta.json" in d for d in prog.damage)


def test_a_generation_whose_llm_calls_failed_is_corrupt(tmp_path):
    """icl.loop records an empty group when a request fails, so the generation looks finished. The
    user's rule: if the LLM was not reachable, that generation is not to be built on."""
    run_dir = _write_run(tmp_path / "r", 3, want=3, drop_group_in=2, snapshot_steps=[2, 3])
    prog = inspect_run(run_dir)
    assert prog.good_generations == 2 and prog.resume_step == 2 and not prog.complete
    assert any("recorded no candidates" in d for d in prog.damage)


def test_a_short_group_is_corrupt_too(tmp_path):
    run_dir = _write_run(tmp_path / "r", 2, want=3, snapshot_steps=[1, 2])
    meta_path = os.path.join(run_dir, "generations", "gen_0001", "meta.json")
    meta = json.load(open(meta_path))
    meta["parents"][1]["children"] = meta["parents"][1]["children"][:1]      # one chunk never returned
    json.dump(meta, open(meta_path, "w"))
    prog = inspect_run(run_dir)
    assert prog.good_generations == 1 and prog.resume_step == 1
    assert any("fewer than 2 candidates" in d for d in prog.damage)


def test_a_truncated_context_pool_rewinds_to_what_it_covers(tmp_path):
    """Resuming on a short pool used to silently shrink every later prompt (the pool is reloaded, not
    rebuilt), which would break the arm comparison rather than the run."""
    run_dir = _write_run(tmp_path / "r", 3, want=5, valid_per_gen=2, pool_lines=4, snapshot_steps=[2, 3])
    prog = inspect_run(run_dir)
    assert prog.good_generations == 2 and prog.resume_step == 2
    assert any("context_pool.jsonl holds 4" in d for d in prog.damage)


def test_a_run_with_no_context_keeps_its_pool_out_of_the_decision(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=3, n_context=0, pool_lines=0)
    assert inspect_run(run_dir).good_generations == 3


def test_without_a_snapshot_the_run_starts_over(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[])
    prog = inspect_run(run_dir)
    assert prog.good_generations == 3 and prog.resume_step == 0
    assert any("no usable PUCT snapshot" in d for d in prog.damage)


def test_a_torn_snapshot_does_not_count(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[2])
    open(os.path.join(run_dir, "buffer", "puct_sampler_step_000003.json"), "w").write('{"states": [')
    prog = inspect_run(run_dir)
    assert prog.resume_step == 2 and 3 not in prog.snapshots


# ---- rewinding the discarded tail -----------------------------------------------------------------
def test_rewind_moves_the_tail_aside_and_leaves_the_prefix_consistent(tmp_path):
    run_dir = _write_run(tmp_path / "r", 4, want=6, snapshot_steps=[2, 4])
    assert tail_exists(run_dir, 2)
    moved = rewind(run_dir, 2)
    assert moved

    stale = [d for d in os.listdir(run_dir) if d.startswith("stale_")]
    assert len(stale) == 1
    stale_dir = os.path.join(run_dir, stale[0])

    # generations, their solutions and the orphaned snapshot are moved, not deleted
    assert sorted(os.listdir(os.path.join(run_dir, "generations"))) == ["gen_0000", "gen_0001"]
    assert sorted(os.listdir(os.path.join(stale_dir, "generations"))) == ["gen_0002", "gen_0003"]
    assert os.path.exists(os.path.join(stale_dir, "buffer", "puct_sampler_step_000004.json"))
    assert os.path.exists(os.path.join(run_dir, "buffer", "puct_sampler_step_000002.json"))
    assert sorted(os.listdir(os.path.join(stale_dir, "solutions"))) == [
        "manifest.jsonl", "sol_000005.py", "sol_000006.py", "sol_000007.py", "sol_000008.py"]

    # the append-only files keep exactly the kept generations
    gens = {json.loads(l)["generation"] for l in open(os.path.join(run_dir, "events.jsonl"))}
    assert gens == {0, 1}
    assert {json.loads(l)["gen"] for l in open(os.path.join(run_dir, "solutions", "manifest.jsonl"))} == {0, 1}
    rows = list(csv.DictReader(open(os.path.join(run_dir, "progress.csv"))))
    assert [r["generation"] for r in rows] == ["0", "1"]
    # the context pool is trimmed to the solutions the kept generations produced
    assert sum(1 for _ in open(os.path.join(run_dir, "buffer", "context_pool.jsonl"))) == 4

    assert not tail_exists(run_dir, 2)
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    assert summary["status"] == "rewound" and len(summary["per_generation"]) == 2
    assert json.load(open(os.path.join(stale_dir, "summary.json")))["status"] == "complete"


def test_rewind_to_zero_clears_everything(tmp_path):
    run_dir = _write_run(tmp_path / "r", 2, want=4)
    rewind(run_dir, 0)
    assert not tail_exists(run_dir, 0)
    assert sum(1 for _ in open(os.path.join(run_dir, "events.jsonl"))) == 0
    assert sum(1 for _ in open(os.path.join(run_dir, "buffer", "context_pool.jsonl"))) == 0
    assert inspect_run(run_dir).good_generations == 0


def test_rewind_is_idempotent(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[2, 3])
    rewind(run_dir, 2)
    assert rewind(run_dir, 2, stamp="second") == []


# ---- the numbers a resumed run has to carry forward ----------------------------------------------
def test_prior_state_rebuilds_the_books_of_the_kept_generations(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=5, groups=2, group_size=2, valid_per_gen=2)
    prior = prior_state(run_dir, 2)
    assert [s["generation"] for s in prior.per_generation] == [0, 1]
    assert prior.succeeded == 4 and prior.failed == 4 and prior.candidates == 8
    assert prior.usage["completion_tokens"] == 400 and prior.usage["requests"] == 2
    assert prior.failure_types == {"invalid": 2}
    assert prior.sol_seq == 4                     # so the next solution is sol_000005, not a collision
    assert prior.best["sol"] == "sol_000004" and prior.worst_valid["sol"] == "sol_000001"
    assert prior.state_to_sol["st1"] == "sol_000001"
    assert prior.started_at == "2026-01-01T00:00:00"


def test_prior_state_of_a_fresh_run_is_empty(tmp_path):
    prior = prior_state(_write_run(tmp_path / "r", 2), 0)
    assert prior.per_generation == [] and prior.sol_seq == 0 and prior.best is None


@pytest.mark.parametrize("keep", [1, 2])
def test_rewind_then_prior_state_agree(tmp_path, keep):
    """What rewind leaves on disk is exactly what prior_state reports — the invariant the tracker
    relies on to keep writing one cumulative summary."""
    run_dir = _write_run(tmp_path / "r", 3, want=5, snapshot_steps=[1, 2, 3])
    rewind(run_dir, keep)
    prior = prior_state(run_dir, keep)
    assert len(prior.per_generation) == keep
    assert prior.sol_seq == sum(1 for _ in open(os.path.join(run_dir, "solutions", "manifest.jsonl")))
    assert prior.succeeded == sum(1 for _ in open(
        os.path.join(run_dir, "buffer", "context_pool.jsonl")))


def test_without_a_target_generation_count_only_the_summary_can_claim_completion(tmp_path):
    """With no num_generations anywhere, generations on disk alone must not read as a finished run —
    a run killed at generation 3 of 15 has 3 perfectly good generations."""
    run_dir = _write_run(tmp_path / "r", 3, want=3)
    os.remove(os.path.join(run_dir, "config.json"))
    os.remove(os.path.join(run_dir, "summary.json"))
    prog = inspect_run(run_dir)
    assert prog.good_generations == 3 and prog.resume_step == 3 and not prog.complete
    # the sweep manifest is where that number comes from in practice
    assert inspect_run(run_dir, num_generations=3).complete


def test_snapshot_verdicts_are_cached_per_file_identity(tmp_path):
    """--status re-inspects every run of a sweep on a timer, and each inspection used to re-parse
    every buffer snapshot -- the whole PUCT buffer, program code and all. Snapshots are immutable once
    written, so the re-parse can only ever reach the same verdict."""
    import results.resume as R

    run_dir = _write_run(tmp_path / "r", 3, want=3, snapshot_steps=[1, 2, 3])
    R._SNAPSHOT_CACHE.clear()
    reads = []
    real_read = R._read_json
    R._read_json = lambda p: (reads.append(p), real_read(p))[1]
    try:
        assert R._snapshot_steps(run_dir) == {1, 2, 3}
        first = len([p for p in reads if "puct_sampler_step" in p])
        assert R._snapshot_steps(run_dir) == {1, 2, 3}
        assert len([p for p in reads if "puct_sampler_step" in p]) == first == 3

        # A file rewritten in place must MISS the cache: a torn snapshot that is later completed has
        # to be re-read, or a run would be refused a resume point it now has.
        path = os.path.join(run_dir, "buffer", "puct_sampler_step_000002.json")
        with open(path, "w") as f:
            f.write('{"step": 2, "states": []}')            # now empty -> no longer resumable
        assert R._snapshot_steps(run_dir) == {1, 3}
    finally:
        R._read_json = real_read


def test_a_run_that_produced_no_valid_solution_is_never_complete(tmp_path):
    """Every check in this module is structural — full groups, full complement of children,
    meta.json written — and a broken evaluator passes all of them: its candidates came back and
    failed, which is what an ordinary generation looks like from the outside. Such a run verified as
    COMPLETE, so --resume skipped it and --status showed it green."""
    run_dir = _write_run(tmp_path / "r", 15, want=15, valid_per_gen=0, status="complete",
                         snapshot_steps=list(range(16)))
    prog = inspect_run(run_dir, 15)

    assert prog.good_generations == 15          # structurally it really is 15 generations
    assert not prog.complete                    # ...but it holds nothing, so it is not finished
    assert any("no result whatever its summary says" in d for d in prog.damage)


def test_a_run_with_any_yield_at_all_is_still_judged_normally(tmp_path):
    run_dir = _write_run(tmp_path / "r", 3, want=3, valid_per_gen=1)
    prog = inspect_run(run_dir, 3)
    assert prog.complete and prog.damage == []
