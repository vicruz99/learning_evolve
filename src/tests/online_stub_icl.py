"""Stand-in for run_icl.py used by tests/test_orchestrate.py (set ONLINE_RUN_ICL_CMD to run it).

Accepts the driver's flags, writes ``--resume-step .. --num-generations-1`` finished generations in
the fabricated layout of tests/online_fixtures.py, and records which ``--model`` served each
generation in ``<log-path>/online_stub_models.jsonl`` so the test can assert the adapter schedule.
Environment knobs: ``STUB_FAIL_AT_GEN`` (exit 1 before writing that generation),
``STUB_SLEEP`` (seconds to linger before exiting, to look like a live driver).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from online_fixtures import write_config, write_generation   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--log-path", required=True)
    ap.add_argument("--resume-step", type=int, default=0)
    ap.add_argument("--num-generations", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--groups-per-batch", type=int, default=2)
    ap.add_argument("--group-size", type=int, default=5)
    a, _rest = ap.parse_known_args()

    run = a.log_path
    if not os.path.exists(os.path.join(run, "config.json")):
        write_config(run, groups=a.groups_per_batch, group_size=a.group_size,
                     want=a.num_generations, model_name=a.model)
    fail_at = os.environ.get("STUB_FAIL_AT_GEN")
    n = a.groups_per_batch * a.group_size
    for g in range(a.resume_step, a.num_generations):
        if fail_at is not None and int(fail_at) == g:
            return 1
        # scores: deterministic, distinct per (seed, gen, i); child 0 of each generation fails
        scores = [None if i % 5 == 0 else round(0.5 + 0.01 * ((a.seed * 7 + g * 13 + i * 3) % 40), 3)
                  for i in range(n)]
        write_generation(run, g, scores, groups=a.groups_per_batch, group_size=a.group_size)
        with open(os.path.join(run, "online_stub_models.jsonl"), "a") as f:
            f.write(json.dumps({"generation": g, "model": a.model}) + "\n")
    time.sleep(float(os.environ.get("STUB_SLEEP", "0")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
