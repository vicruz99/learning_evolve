# -*- coding: utf-8 -*-
"""Re-score a finished run with its own eval.py, and report the best VALID candidate.

Never trust a score the agent reported about itself: the pilot had runs whose ledger
claimed improvements that did not survive a re-grade. This re-runs the grading function
over every .npy the run produced, so the number in the report is one we computed.

usage: score.py <run_dir> [<run_dir> ...]
"""
import importlib.util
import os
import sys

import numpy as np


def load_eval(run):
    spec = importlib.util.spec_from_file_location("run_eval", os.path.join(run, "eval.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.evaluate_sequence


def candidates(run):
    for dirpath, _, files in os.walk(run):
        for fn in sorted(files):
            if fn.endswith(".npy"):
                yield os.path.join(dirpath, fn)


def main(run):
    run = os.path.abspath(run)
    try:
        ev = load_eval(run)
    except Exception as exc:
        print("%-34s eval.py unusable: %s" % (os.path.basename(run), exc))
        return
    best, best_p, n_valid, n_total = None, None, 0, 0
    for p in candidates(run):
        n_total += 1
        try:
            seq = [float(x) for x in np.load(p).ravel()]
            s = float(ev(seq))
        except Exception:
            continue
        n_valid += 1
        # AC1 minimises, AC2 maximises; keep both extremes and let the caller read the task.
        if best is None or s > best[1]:
            best = (p, s)
        if best_p is None or s < best_p[1]:
            best_p = (p, s)
    if best is None:
        print("%-34s no valid candidate (%d .npy seen)" % (os.path.basename(run), n_total))
        return
    print("%-34s %d/%d valid   max=%.6f (%s)   min=%.6f (%s)"
          % (os.path.basename(run), n_valid, n_total,
             best[1], os.path.relpath(best[0], run),
             best_p[1], os.path.relpath(best_p[0], run)))


if __name__ == "__main__":
    for d in sys.argv[1:]:
        main(d)
