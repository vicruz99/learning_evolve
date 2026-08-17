#!/usr/bin/env python3
import argparse
import os

import yaml

from shinka.core import ShinkaEvolveRunner, EvolutionConfig
from shinka.database import DatabaseConfig
from shinka.launch import LocalJobConfig

# Domain guidance ported verbatim from TTT-Discover
# (discover/examples/ac_inequalities/env.py::get_question ac2 branch, plus ae_verifier_program
# and AC2_LITERATURE from prompt.py). TTT harness-mechanics rules (construct_function signature,
# no-lambdas, no-IO, return-fences, print-statement note) are dropped; Shinka appends its own
# format instructions automatically.
search_task_sys_msg = r"""Act as an expert software developer and inequality specialist specializing in creating step functions with certain properties.

Your task is to generate the sequence of non-negative heights of a step function, that maximizes the following evaluation function:

```python
import numpy as np
def evaluate_sequence(sequence: list[float]) -> float:
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")
    if not sequence:
        raise ValueError("Empty sequence")
    for x in sequence:
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")
    sequence = [float(x) for x in sequence]
    sequence = [max(0, x) for x in sequence]
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    sequence = [min(1000.0, x) for x in sequence]
    convolution_2 = np.convolve(sequence, sequence)
    # Calculate the 2-norm squared: ||f*f||_2^2
    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points) # Width of each interval
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):  # Iterate through intervals
        y1 = y_points[i]
        y2 = y_points[i+1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared
    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)
    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound
```

A previous state of the art used the following approach. You can use it as inspiration, but you are not required to use it, and you are encouraged to explore.
```latex
Their procedure is a coarse-to-fine optimization of the score. It starts with a stochastic global search that repeatedly perturbs the current best candidate and keeps the perturbation whenever it improves (Q), with the perturbation scale gradually reduced over time. Once a good basin is found, they switch to a deterministic local improvement step, performing projected gradient ascent (move in the gradient direction and project back to the feasible region). To reach higher resolution, they lift a good low-resolution solution to a higher-dimensional one by simply repeating its entries and then rerun the local refinement. Iterating this explore-refine-upscale cycle yields their final high-resolution maximizer and the improved lower bound.
```

Your task is to write a search function that searches for the best sequence of coefficients. Your function will have ~1000 seconds to run (search internally within this budget), and after that it has to have returned the best sequence it found. All numbers in your sequence have to be positive or zero. Larger sequences with 1000s of items often have better attack surface, but too large sequences with 100s of thousands of items may be too slow to search.

You may code up any search method you want, and you are allowed to call the evaluate_sequence() function as many times as you want. You have access to it, you don't need to code up the evaluate_sequence() function.

Target lower bound: 0.97 (higher is better).

You may want to start your search from one of the constructions we have found so far, which you can access through the `height_sequence_1` global variable.
However, you are encouraged to explore solutions that use other starting points to prevent getting stuck in a local minimum.

Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparameters, etc.
Unless you make a meaningful improvement, you will not be rewarded; if you are stuck you should think about how to get unstuck.

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

    # Internal search budget (seconds) for evolved variants that add time-awareness.
    eval_budget_s = config.get("eval_budget_s")
    if eval_budget_s is not None:
        os.environ["TTT_BUDGET_S"] = str(eval_budget_s)
    job_time = config.get("job_time", "00:30:00")

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
