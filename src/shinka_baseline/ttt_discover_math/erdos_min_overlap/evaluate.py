"""Evaluator for the Erdos minimum-overlap problem, ported to ShinkaEvolve.

Verifier and scoring mirror discover/examples/erdos_min_overlap/env.py exactly so the score
matches TTT-Discover. This is a MINIMIZE objective (lower C5 is better), so we report
combined_score = 1/(1e-8 + c5), the same reward shaping TTT-Discover uses.
"""

import os
import argparse
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

from shinka.core import run_shinka_eval


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def validate_erdos(run_output) -> Tuple[bool, Optional[str]]:
    """Shinka validate_fn: reproduces discover's verify_erdos_solution with a message."""
    try:
        h_values, c5_bound, n_points = run_output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False, f"Invalid C5 bound: {c5_bound}"
    except Exception as e:
        return False, f"Invalid solution: {e}"
    return True, f"Valid. C5 bound = {c5_bound}"


def aggregate_erdos_metrics(results: List[Any], results_dir: str) -> Dict[str, Any]:
    if not results:
        return {"combined_score": 0.0, "error": "No results to aggregate"}

    h_values, c5_bound, n_points = results[0]
    c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)  # true metric (lower better)

    metrics = {
        "combined_score": float(1.0 / (1e-8 + c5_bound)),  # maximize -> minimize C5
        "public": {
            "c5_bound": float(c5_bound),
            "n_points": int(n_points),
        },
        "private": {
            "c5_bound": float(c5_bound),
        },
    }

    extra_file = os.path.join(results_dir, "extra.npz")
    try:
        np.savez(extra_file, h_values=np.asarray(h_values, dtype=float), c5_bound=c5_bound, n_points=n_points)
    except Exception as e:
        metrics["extra_npz_save_error"] = str(e)

    return metrics


def main(program_path: str, results_dir: str):
    print(f"Evaluating program: {program_path}")
    print(f"Saving results to: {results_dir}")
    os.makedirs(results_dir, exist_ok=True)

    def _aggregator_with_context(r):
        return aggregate_erdos_metrics(r, results_dir)

    metrics, correct, error_msg = run_shinka_eval(
        program_path=program_path,
        results_dir=results_dir,
        experiment_fn_name="run_erdos",
        num_runs=1,
        validate_fn=validate_erdos,
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
    parser = argparse.ArgumentParser(description="Erdos min-overlap evaluator (shinka.eval)")
    parser.add_argument("--program_path", type=str, default="initial.py")
    parser.add_argument("--results_dir", type=str, default="results")
    parsed_args = parser.parse_args()
    main(parsed_args.program_path, parsed_args.results_dir)
