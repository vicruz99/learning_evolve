"""Erdos minimum-overlap problem (C5 upper bound), ported from TTT-Discover.

TTT-Discover gives this problem a random initial *construction* (`initial_h_values`) but NO
seed algorithm (the model is told "Write code to optimize this construction"). So the
EVOLVE-BLOCK below is a minimal seed that just returns the given construction; evolution is
expected to write the actual optimizer, calling the fixed helpers exposed here.

Fixed (read-only) context, exactly mirroring what TTT-Discover injects into the sandbox:
  - `verify_c5_solution`, `evaluate_erdos_solution`  (ported from
    discover/examples/erdos_min_overlap/env.py)
  - `initial_h_values`  (the initial construction; see note on the RNG seed below)
The evolved code may call `evaluate_erdos_solution(...)` and start from `initial_h_values`.
"""

import os

import numpy as np


# --- Fixed verifier / evaluator (ported verbatim from TTT-Discover) ------------------------
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


# --- Initial construction (exposed as a global, like TTT-Discover) -------------------------
# NOTE: TTT-Discover regenerates this with an *unseeded* RNG on every rollout. Our ICL runs
# (src/envs/erdos_min_overlap.py::ErdosMinOverlapEnv.initial_state_seed) pin seed 12345 with
# this exact RNG call order; we use the same seed so both frameworks start from the identical
# construction.
def _make_initial_h_values(seed: int = 12345):
    rng = np.random.default_rng(seed)
    n_points = int(rng.integers(40, 100))
    construction = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation
    return construction


initial_h_values = _make_initial_h_values()


# EVOLVE-BLOCK-START
def run(seed=42, budget_s=1000, **kwargs):
    """Minimal seed: return the given initial construction and its C5 value.

    Evolution should replace this with an actual optimizer that lowers C5, e.g. by
    perturbing/refining the construction and scoring candidates with
    `evaluate_erdos_solution(h, c5, n)`. You may start from `initial_h_values` or explore
    other starting points.

    Returns:
        (h_values, c5_bound, n_points)
    """
    h = np.array(initial_h_values, dtype=float)
    n_points = len(h)
    dx = 2.0 / n_points
    correlation = np.correlate(h, 1.0 - h, mode="full") * dx
    c5_bound = float(np.max(correlation))
    return h.tolist(), c5_bound, n_points
# EVOLVE-BLOCK-END


# Fixed entry point Shinka invokes. Passes the internal search budget (TTT_BUDGET_S, default
# 1000s) to the evolved `run`, mirroring TTT-Discover's ~1000s search budget.
def run_erdos(seed=42, **kwargs):
    budget_s = float(os.environ.get("TTT_BUDGET_S", "1000"))
    return run(seed=seed, budget_s=budget_s)
