"""Evaluator for the AC1 autocorrelation inequality, ported to ShinkaEvolve.

`evaluate_sequence` == TTT `evaluate_sequence_ac1` (returns np.inf on invalid). AC1 is a
MINIMIZE objective (lower upper bound is better), so we report combined_score =
1/(1e-8 + bound), the same reward shaping TTT-Discover uses.
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


def evaluate_sequence(sequence: list) -> float:
    """Returns np.inf if the input is invalid (== TTT evaluate_sequence_ac1)."""
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


def validate_ac1(run_output) -> Tuple[bool, Optional[str]]:
    """Shinka validate_fn: reproduces discover's verify_ac1_solution with a message."""
    try:
        value = evaluate_sequence(run_output)
        if value == np.inf or np.isnan(value):
            return False, "Invalid solution (evaluate_sequence returned inf/nan)."
    except Exception as e:
        return False, f"Invalid solution: {e}"
    return True, f"Valid. Upper bound = {value}"


def aggregate_ac1_metrics(results: List[Any], results_dir: str) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    sequence = results[0]
    result = evaluate_sequence(sequence)  # true metric: upper bound (lower is better)

    metrics = {
        "combined_score": float(1.0 / (1e-8 + result)),  # maximize -> minimize upper bound
        "public": {
            "upper_bound": float(result),
            "sequence_length": int(len(sequence)) if hasattr(sequence, "__len__") else -1,
        },
        "private": {
            "upper_bound": float(result),
        },
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(extra_file, sequence=np.asarray(sequence, dtype=float), upper_bound=result)
    except Exception as e:
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator_with_context(r):
        return aggregate_ac1_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_ac1",
        num_runs=1,
        validate_fn=validate_ac1,
        aggregate_metrics_fn=_aggregator_with_context,
    )

    if correct:
        print("Evaluation and Validation completed successfully.")
    else:
        print(f"Evaluation or Validation failed: {error_msg}")

    print("Metrics:")
    for key, value in metrics.items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: <string_too_long_to_display>")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AC1 evaluator (shinka.eval)")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
