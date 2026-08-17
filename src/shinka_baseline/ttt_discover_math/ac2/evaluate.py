"""Evaluator for the AC2 autocorrelation inequality, ported to ShinkaEvolve.

`evaluate_sequence` == TTT `evaluate_sequence_ac2` (the `ae_verifier_program`; raises on
invalid). AC2 is a MAXIMIZE objective (higher lower-bound is better), so combined_score is the
metric itself, matching TTT-Discover.
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


def evaluate_sequence(sequence: list) -> float:
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

    num_points = len(convolution_2)
    x_points = np.linspace(-0.5, 0.5, num_points + 2)
    x_intervals = np.diff(x_points)
    y_points = np.concatenate(([0], convolution_2, [0]))
    l2_norm_squared = 0.0
    for i in range(len(convolution_2) + 1):
        y1 = y_points[i]
        y2 = y_points[i + 1]
        h = x_intervals[i]
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


def validate_ac2(run_output) -> Tuple[bool, Optional[str]]:
    """Shinka validate_fn: reproduces discover's verify_ac2_solution with a message."""
    try:
        value = evaluate_sequence(run_output)
        if value == np.inf or np.isnan(value):
            return False, "Invalid solution (evaluate_sequence returned inf/nan)."
    except Exception as e:
        return False, f"Invalid solution: {e}"
    return True, f"Valid. Lower bound = {value}"


def aggregate_ac2_metrics(results: List[Any], results_dir: str) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    sequence = results[0]
    result = evaluate_sequence(sequence)  # true metric: lower bound (higher is better)

    metrics = {
        "combined_score": float(result),  # maximize lower bound directly
        "public": {
            "lower_bound": float(result),
            "sequence_length": int(len(sequence)) if hasattr(sequence, "__len__") else -1,
        },
        "private": {
            "lower_bound": float(result),
        },
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(extra_file, sequence=np.asarray(sequence, dtype=float), lower_bound=result)
    except Exception as e:
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator_with_context(r):
        return aggregate_ac2_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_ac2",
        num_runs=1,
        validate_fn=validate_ac2,
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
    parser = argparse.ArgumentParser(description="AC2 evaluator (shinka.eval)")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
