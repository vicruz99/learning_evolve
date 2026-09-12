"""Unit tests for sft/online/select.py (pure; no model, no server)."""
import json
import os
import subprocess
import sys

import pytest

from sft.online.select import (EXIT_EMPTY, dedup, run_select, select_pct_threshold,
                               select_top_frac, window)
from online_fixtures import write_config, write_generation

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G, GS = 2, 5           # 2 parents x 5 children = 10 candidates per generation


def _cands(scores, gen=0):
    return [{"generation": gen, "parent_slot": i // GS, "child": i % GS, "raw_score": s,
             "answer": f"a{gen}-{i}"} for i, s in enumerate(scores) if s is not None]


# ---- top_frac -------------------------------------------------------------------------------------
def test_top_frac_rounds_and_keeps_at_least_one():
    c = _cands([0.1, 0.9, 0.5, 0.7, 0.3, None, None, None, None, None])       # 5 valid
    chosen, info = select_top_frac(c, 0.4, maximize=True)
    assert [x["raw_score"] for x in chosen] == [0.9, 0.7]                      # round(2.0) = 2
    chosen, _ = select_top_frac(c, 0.05, maximize=True)
    assert [x["raw_score"] for x in chosen] == [0.9]                           # max(1, round(0.25))
    chosen, _ = select_top_frac([], 0.4, maximize=True)
    assert chosen == []


def test_top_frac_pools_across_a_window_and_respects_minimize():
    c = _cands([0.9, 0.8, None, None, None, None, None, None, None, None], gen=0) + \
        _cands([0.1, 0.2, None, None, None, None, None, None, None, None], gen=1)
    chosen, info = select_top_frac(c, 0.5, maximize=True)
    assert sorted(x["raw_score"] for x in chosen) == [0.8, 0.9]                # both from gen 0
    assert info["per_generation"][0]["selected"] == 2 and info["per_generation"][1]["selected"] == 0
    chosen, _ = select_top_frac(c, 0.5, maximize=False)
    assert sorted(x["raw_score"] for x in chosen) == [0.1, 0.2]


# ---- pct_threshold --------------------------------------------------------------------------------
def test_pct_threshold_per_generation_with_ties_included():
    g0 = _cands([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], gen=0)   # 10 valid, k=2
    chosen, info = select_pct_threshold(g0, 80, maximize=True)
    assert sorted(x["raw_score"] for x in chosen) == [0.9, 0.9]
    assert info["per_generation"][0] == {"valid": 10, "k": 2, "threshold": 0.9, "selected": 2,
                                         "degenerate": False}
    # a tie AT the threshold is kept: k=2 -> threshold 0.8, three candidates >= 0.8
    g1 = _cands([0.9, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], gen=1)
    chosen, info = select_pct_threshold(g1, 80, maximize=True)
    assert sorted(x["raw_score"] for x in chosen) == [0.8, 0.8, 0.9]
    assert info["per_generation"][1]["selected"] == 3 and not info["per_generation"][1]["degenerate"]


def test_pct_threshold_small_generation_gives_top_one():
    g = _cands([0.3, 0.5, 0.4, None, None, None, None, None, None, None])       # 3 valid < 5
    chosen, info = select_pct_threshold(g, 80, maximize=True)
    assert [x["raw_score"] for x in chosen] == [0.5]
    assert info["per_generation"][0]["k"] == 1


def test_pct_threshold_all_equal_scores_is_degenerate_and_truncated():
    g = _cands([0.5] * 10)
    chosen, info = select_pct_threshold(g, 80, maximize=True)
    assert len(chosen) == 2 and info["per_generation"][0]["degenerate"] is True
    # deterministic tie-break: lowest (parent_slot, child) first
    assert [(x["parent_slot"], x["child"]) for x in chosen] == [(0, 0), (0, 1)]


def test_pct_threshold_unions_generations():
    c = _cands([0.9, 0.8] + [0.1] * 8, gen=0) + _cands([0.3, 0.2] + [0.1] * 8, gen=1)
    chosen, info = select_pct_threshold(c, 80, maximize=True)
    assert sorted(x["raw_score"] for x in chosen) == [0.2, 0.3, 0.8, 0.9]
    assert set(info["per_generation"]) == {0, 1}
    # a generation whose p80 lands inside a block of ties is degenerate: truncated to k, not "everything"
    c = _cands([0.9] + [0.1] * 9, gen=0) + _cands([0.2] + [0.1] * 9, gen=1)
    chosen, info = select_pct_threshold(c, 80, maximize=True)
    assert sorted(x["raw_score"] for x in chosen) == [0.1, 0.1, 0.2, 0.9]
    assert info["per_generation"][0]["degenerate"] and info["per_generation"][1]["degenerate"]


# ---- dedup ----------------------------------------------------------------------------------------
def test_dedup_drops_trained_and_within_round_duplicates_keeping_best():
    c = _cands([0.9, 0.8, 0.7])
    c[1]["answer"] = c[0]["answer"]                    # within-round duplicate, lower score
    from sft.build_dataset import answer_sha
    kept, stats = dedup(c, exclude={answer_sha(c[2]["answer"])}, maximize=True)
    assert [x["raw_score"] for x in kept] == [0.9]
    assert stats == {"cross_round": 1, "within": 1}


# ---- end to end on a fabricated run dir ------------------------------------------------------------
def _run(tmp_path, gens):
    run = str(tmp_path / "run")
    write_config(run, groups=G, group_size=GS, want=12)
    for g, scores in enumerate(gens):
        write_generation(run, g, scores, groups=G, group_size=GS)
    return run


def test_run_select_writes_dataset_and_manifest(tmp_path):
    run = _run(tmp_path, [[0.9, 0.8, 0.7, 0.6, 0.5, None, None, None, None, None],
                          [0.95, 0.1, None, None, None, None, None, None, None, None]])
    out = str(tmp_path / "round_00")
    m = run_select(run, out, 0, 1, "top_frac", 0.3)
    assert m["counts"]["selected"] == 2 and m["k"] == 2                       # round(0.3 * 7)
    rows = [json.loads(l) for l in open(os.path.join(out, "train.jsonl"))]
    assert sorted(r["raw_score"] for r in rows) == [0.9, 0.95]
    assert all(r["answer_sha"] and r["reasoning"] == "think" for r in rows)
    assert m["window"] == {"gen_lo": 0, "gen_hi": 1, "eligible_in_window": 7,
                           "generations_in_window": [0, 1]}
    assert len(m["selected_shas"]) == 2


def test_run_select_window_excludes_other_generations_and_excluded_shas(tmp_path):
    run = _run(tmp_path, [[0.9] + [None] * 9, [0.5, 0.4] + [None] * 8])
    out = str(tmp_path / "r")
    m = run_select(run, out, 1, 1, "pct_threshold", 80)
    assert m["counts"]["selected"] == 1 and m["generations_covered"] == [1]
    shas = tmp_path / "trained.txt"
    shas.write_text("\n".join(m["selected_shas"]) + "\n")
    m2 = run_select(run, str(tmp_path / "r2"), 1, 1, "pct_threshold", 80, exclude_shas=str(shas))
    assert m2["counts"]["selected"] == 0 and m2["dedup"]["cross_round"] == 1
    assert os.path.exists(tmp_path / "r2" / "manifest.json")                  # a skipped round leaves a record


def test_run_select_refuses_a_run_without_reasoning(tmp_path):
    run = str(tmp_path / "run")
    write_config(run, groups=G, group_size=GS, want=3, save_reasoning=False)
    write_generation(run, 0, [0.5] * 10, groups=G, group_size=GS, reasoning=False)
    with pytest.raises(RuntimeError):
        run_select(run, str(tmp_path / "o"), 0, 0, "top_frac", 0.5)


def test_cli_exit_codes(tmp_path):
    run = _run(tmp_path, [[0.9] + [None] * 9])
    cmd = [sys.executable, "-m", "sft.online.select", "--run", run, "--out", str(tmp_path / "o"),
           "--gen-lo", "0", "--gen-hi", "0", "--select", "top_frac", "--frac", "0.5"]
    assert subprocess.run(cmd, cwd=SRC, capture_output=True).returncode == 0
    cmd[cmd.index("--gen-lo") + 1] = "5"
    cmd[cmd.index("--gen-hi") + 1] = "5"
    cmd[cmd.index("--out") + 1] = str(tmp_path / "o2")
    assert subprocess.run(cmd, cwd=SRC, capture_output=True).returncode == EXIT_EMPTY
