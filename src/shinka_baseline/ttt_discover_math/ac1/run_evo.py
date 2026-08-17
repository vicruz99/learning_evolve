#!/usr/bin/env python3
import argparse
import os

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Domain guidance ported verbatim from TTT-Discover
# (discover/examples/ac_inequalities/env.py::get_question ac1 branch, plus AC1_EVAL_FUNCTION
# and AC1_LITERATURE from prompt.py). TTT harness-mechanics rules (propose_candidate signature,
# no-lambdas, no-IO, return-fences, print-statement note) are dropped; Shinka appends its own
# format instructions automatically.
search_task_sys_msg = r"""Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that minimizes the following evaluation function:

```python
import numpy as np

def evaluate_sequence(sequence: list[float]) -> float:
    \"\"\"
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    \"\"\"
    if not isinstance(sequence, list):
        return np.inf
    if not sequence:
        return np.inf
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf
        if np.isnan(x) or np.isinf(x):
            return np.inf
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    sequence = [min(1000.0, x) for x in sequence]
    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)
    if sum_a < 0.01:
        return np.inf
    return float(2 * n * max_b / (sum_a**2))
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Starting from a nonnegative step function $f=(a_0,\dots,a_{n-1})$ normalized so that $\sum_j a_j=\sqrt{2n}$, set $M=\|f*f\|_\infty$. Next compute $g_0=(b_0,\dots,b_{n-1})$ by solving a linear program, i.e.\ maximizing $\sum_j b_j$ subject to $b_j\ge0$ and $\|f*g_0\|_\infty\le M$; as is standard, the optimum is attained at an extreme point determined by an active set of binding inequalities, here corresponding to important constraints where the convolution bound $(f*g_0)(x)\le M$ is tight and limiting. Rescale $g_0$ to match the normalization, $g=\frac{\sqrt{2n}}{\sum_j b_j}g_0$, and update $f\leftarrow (1-t)f+t g$ for a small $t>0$. Repeating this step produces a sequence with nonincreasing $\|f*f\|_\infty$, and the iteration is continued until it stabilizes.
```

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have ~1000 seconds to run (search internally within this budget), and after that it has to have returned the best sequence it found. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Target upper bound: 1.5030 (lower is better).

You may want to start your search from one of the constructions we have found so far, which you can access through the `height_sequence_1` global variable.
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparameters, etc.
Unless you make a meaningful improvement, you will not be rewarded.

Guidance:
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- You can use up to 2 CPUs."""


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
