#!/usr/bin/env python3
import argparse
import os

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Domain guidance ported verbatim from TTT-Discover
# (discover/examples/erdos_min_overlap/env.py::get_question). TTT harness-mechanics rules
# (the `run(...)` signature requirement, return-fences, no-lambdas, filesystem/IO rules) are
# dropped; Shinka appends its own format instructions automatically.
search_task_sys_msg = r"""You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdos minimum overlap problem constant C5.

## Problem

Find a step function h: [0, 2] -> [0, 1] that **minimizes** the overlap integral:

$$C_5 = \max_k \int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) in [0, 1] for all x
2. integral of h(x) dx over [0, 2] = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 <= h[i] <= 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C5 = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C5 values are better** - they provide tighter upper bounds on the Erdos constant.

## Budget & Resources
- **Time budget**: ~1000s for your code to run (search internally within this budget)
- **CPUs**: 2 available

## Available helpers
- `evaluate_erdos_solution(h_values, c5_bound, n_points)` is available (do not redefine it); it verifies and returns the C5 bound.
- `initial_h_values` (an initial construction) is available as a global.

**Lower is better**. Current record: C5 <= 0.38092. Our goal is to find a construction that shows C5 <= 0.38080.

You may want to start your search from the current construction (`initial_h_values`).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.

Reason about how you could further improve this construction. Ideally, try to do something different than a naive algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparameters, etc.
Unless you make a meaningful improvement, you will not be rewarded."""


def main(
    config_path: str,
    results_dir: str | None = None,
    embedding_model: str | None = None,
):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Internal search budget (seconds) for the evolved program's optimization loop.
    eval_budget_s = config.get("eval_budget_s")
    if eval_budget_s is not None:
        os.environ["TTT_BUDGET_S"] = str(eval_budget_s)
    job_time = config.get("job_time", "00:20:00")

    config["evo_config"]["task_sys_msg"] = search_task_sys_msg
    if results_dir is not None:
        config["evo_config"]["results_dir"] = results_dir
    if embedding_model is not None:
        config["evo_config"]["embedding_model"] = embedding_model
    evo_config = EvolutionConfig(**config["evo_config"])
    job_config = LocalJobConfig(
        eval_program_path="evaluate.py",
        time=job_time,
        # Cap OMP/BLAS threads per eval subprocess (prompt promises 2 CPUs);
        # without this each eval's numeric libraries default to every core.
        numeric_threads_per_job=config.get("numeric_threads_per_job", 2),
    )
    db_config = DatabaseConfig(**config["db_config"])

    runner = ShinkaEvolveRunner(
        evo_config=evo_config,
        job_config=job_config,
        db_config=db_config,
        max_evaluation_jobs=config.get("max_evaluation_jobs"),
        max_proposal_jobs=config.get("max_proposal_jobs"),
        max_db_workers=config.get("max_db_workers"),
        debug=False,
        verbose=True,
    )
    runner.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="shinka_dev.yaml")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override evo_config.results_dir (e.g. one dir per replicate).")
    parser.add_argument("--embedding_model", type=str, default=None,
                        help="Override evo_config.embedding_model, e.g. "
                             "local/nomic-embed-text:latest@http://localhost:11434/v1 "
                             "(the CPU ollama server, see RUNS.md).")
    args = parser.parse_args()
    main(args.config_path, args.results_dir, args.embedding_model)
