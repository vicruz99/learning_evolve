"""Fabricated ICL run directories for the online-SFT unit tests.

Writes the subset of ``ExperimentTracker``'s layout that ``sft.build_dataset.collect`` and
``results.resume.inspect_run`` read: ``config.json``, ``events.jsonl`` with prompt / completion /
reasoning files, per-generation ``meta.json`` with full groups, PUCT snapshots and the context pool.
Scores are supplied per generation so tests can shape percentiles and ties exactly.
"""
from __future__ import annotations

import json
import os


def write_config(run_dir: str, *, groups: int, group_size: int, want: int,
                 model_name: str = "base", maximize: bool = True, save_reasoning: bool = True) -> None:
    os.makedirs(run_dir, exist_ok=True)
    json.dump({"num_generations": want, "groups_per_batch": groups, "group_size": group_size,
               "n_context": 0, "problem": "ac2", "model_name": model_name,
               "save_reasoning": save_reasoning,
               "_meta": {"created_at": "2026-01-01T00:00:00", "metric_name": "lower bound",
                         "maximize": maximize}},
              open(os.path.join(run_dir, "config.json"), "w"))


def write_generation(run_dir: str, gen: int, scores: list, *, groups: int, group_size: int,
                     answer_of=None, finish_reason: str = "stop", reasoning: bool = True) -> None:
    """Append one finished generation. ``scores[i]`` is child i's raw_score, or None for a failure.

    ``answer_of(gen, i)`` gives the answer text (defaults to a unique string per candidate), so
    duplicate answers across generations can be produced on purpose.
    """
    assert len(scores) == groups * group_size
    gen_dir = os.path.join(run_dir, "generations", f"gen_{gen:04d}")
    os.makedirs(os.path.join(run_dir, "buffer"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "solutions"), exist_ok=True)
    parents = []
    events = []
    valid = 0
    for slot in range(groups):
        pdir = os.path.join(gen_dir, f"parent_{slot:02d}")
        os.makedirs(pdir, exist_ok=True)
        prompt_rel = os.path.relpath(os.path.join(pdir, "prompt.txt"), run_dir)
        open(os.path.join(run_dir, prompt_rel), "w").write(f"prompt g{gen} p{slot}")
        children = []
        for c in range(group_size):
            i = slot * group_size + c
            score = scores[i]
            ok = score is not None
            valid += int(ok)
            comp_rel = os.path.relpath(os.path.join(pdir, f"child_{c:02d}.txt"), run_dir)
            answer = answer_of(gen, i) if answer_of else f"answer g{gen} i{i}"
            open(os.path.join(run_dir, comp_rel), "w").write(answer)
            if reasoning:
                open(os.path.join(run_dir, comp_rel[:-4] + ".reasoning.txt"), "w").write("think")
            children.append({"child": c, "correctness": 1.0 if ok else 0.0})
            events.append({"generation": gen, "parent_slot": slot, "child": c,
                           "correctness": 1.0 if ok else 0.0, "raw_score": score,
                           "reward": score, "finish_reason": finish_reason,
                           "completion_file": comp_rel, "prompt_file": prompt_rel,
                           "reasoning_tokens": 10, "answer_tokens": 5})
        parents.append({"slot": slot, "parent_sol": "seed", "children": children})
    stats = {"generation": gen, "valid_candidates": valid,
             "failed_candidates": groups * group_size - valid, "wall_seconds": 1.0,
             "usage": {"completion_tokens": 100}}
    with open(os.path.join(run_dir, "events.jsonl"), "a") as f:
        f.writelines(json.dumps(e) + "\n" for e in events)
    with open(os.path.join(run_dir, "buffer", "context_pool.jsonl"), "a") as f:
        f.writelines(json.dumps({"id": f"g{gen}i{i}"}) + "\n" for i in range(valid))
    json.dump({"step": gen + 1, "states": [{"id": "s"}]},
              open(os.path.join(run_dir, "buffer", f"puct_sampler_step_{gen + 1:06d}.json"), "w"))
    if gen == 0:
        json.dump({"step": 0, "states": [{"id": "s"}]},
                  open(os.path.join(run_dir, "buffer", "puct_sampler_step_000000.json"), "w"))
    # meta.json is written LAST, as the tracker does: its presence is what marks the generation done.
    json.dump({"generation": gen, "stats": stats, "parents": parents},
              open(os.path.join(gen_dir, "meta.json"), "w"))
