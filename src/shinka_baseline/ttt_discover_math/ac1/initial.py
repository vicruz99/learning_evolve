"""Autocorrelation inequality AC1 (minimize upper bound), ported from TTT-Discover.

Fixed (read-only) context, exactly mirroring what TTT-Discover injects into the sandbox:
  - `evaluate_sequence`  == TTT `evaluate_sequence_ac1` (returns np.inf on invalid input);
    ported from discover/examples/ac_inequalities/{env.py,prompt.py}. The evolved code may
    call it as many times as it likes.
  - `height_sequence_1`  the initial construction, regenerated with the same RNG (seed 12345)
    and call order as discover/examples/ac_inequalities/env.py::create_initial_state.

The EVOLVE-BLOCK holds TTT-Discover's seed search program `propose_candidate`
(discover/examples/ac_inequalities/prompt.py::example_ae_program_random_init),
byte-identical to what the ICL first prompt shows (incl. its own imports and comments).
"""

import os
import time

import numpy as np
from scipy import optimize

linprog = optimize.linprog


# --- Fixed evaluator (== TTT evaluate_sequence_ac1; returns inf on invalid) -----------------
def evaluate_sequence(sequence: list) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
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


# --- Initial construction (exposed as a global, like TTT-Discover) -------------------------
# Matches discover/examples/ac_inequalities/env.py::create_initial_state (seed 12345, call
# order preserved): a single random value repeated a random number of times.
def _make_height_sequence_1():
    rng = np.random.default_rng(12345)
    value = rng.random()
    length = rng.integers(1000, 8000)
    return [value] * int(length)


height_sequence_1 = _make_height_sequence_1()


# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import optimize
linprog = optimize.linprog


def get_good_direction_to_move_into(sequence):
    """Returns a better direction using LP to find g with larger sum while keeping conv bounded."""
    n = len(sequence)
    sum_sequence = np.sum(sequence)
    normalized_sequence = [x * np.sqrt(2 * n) / sum_sequence for x in sequence]
    rhs = np.max(np.convolve(normalized_sequence, normalized_sequence))
    g_fun = solve_convolution_lp(normalized_sequence, rhs)
    if g_fun is None:
        return None
    sum_g = np.sum(g_fun)
    normalized_g_fun = [x * np.sqrt(2 * n) / sum_g for x in g_fun]
    t = 0.01
    new_sequence = [(1 - t) * x + t * y for x, y in zip(sequence, normalized_g_fun)]
    return new_sequence


def solve_convolution_lp(f_sequence, rhs):
    """Solves the LP: maximize sum(b) s.t. conv(f, b) <= rhs, b >= 0."""
    n = len(f_sequence)
    c = -np.ones(n)
    a_ub = []
    b_ub = []
    for k in range(2 * n - 1):
        row = np.zeros(n)
        for i in range(n):
            j = k - i
            if 0 <= j < n:
                row[j] = f_sequence[i]
        a_ub.append(row)
        b_ub.append(rhs)
    a_ub_nonneg = -np.eye(n)
    b_ub_nonneg = np.zeros(n)
    a_ub = np.vstack([a_ub, a_ub_nonneg])
    b_ub = np.hstack([b_ub, b_ub_nonneg])
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, options={
        'time_limit': 10.0,   # seconds, make sure we don't get stuck
        'disp': False,
    })
    if result.success:
        return result.x
    return None


def propose_candidate(seed=42, budget_s=1000, **kwargs):
    np.random.seed(seed)
    deadline = time.time() + budget_s - 10
        
    if np.random.rand() < 0.5:
        # Start from the SOTA sequence (already available as height_sequence_1)
        best_sequence = list(height_sequence_1)
    else:
        # Start from random initialization, could help if height_sequence_1 is a local minimum
        best_sequence = [np.random.random()] * np.random.randint(100, 1000)
    curr_sequence = best_sequence.copy()
    best_score = evaluate_sequence(best_sequence)
    
    while time.time() < deadline:
        h_function = get_good_direction_to_move_into(curr_sequence)
        if h_function is None:
            # Random perturbation if LP fails
            idx = np.random.randint(len(curr_sequence))
            curr_sequence[idx] = max(0, curr_sequence[idx] + np.random.randn() * 0.01)
        else:
            curr_sequence = h_function
        
        try:
            curr_score = evaluate_sequence(curr_sequence)
            if curr_score < best_score:
                best_score = curr_score
                best_sequence = curr_sequence.copy()
        except:
            pass
    
    return best_sequence
# EVOLVE-BLOCK-END


# Fixed entry point Shinka invokes. Passes the internal search budget (TTT_BUDGET_S, default
# 1000s) to the evolved `propose_candidate`, mirroring TTT-Discover's ~1000s search budget.
def run_ac1(seed=42, **kwargs):
    budget_s = float(os.environ.get("TTT_BUDGET_S", "1000"))
    return propose_candidate(seed=seed, budget_s=budget_s)
