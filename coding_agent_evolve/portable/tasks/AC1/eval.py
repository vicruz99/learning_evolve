"""Grading function for AC1 (autocorrelation inequality, upper bound), copied byte-for-byte from
src/envs/ac_inequalities.py (evaluate_sequence_ac1). DO NOT MODIFY -- this is the
function your candidates are scored with.
"""
import numpy as np


def evaluate_sequence(sequence: list[float]) -> float:
    """
    Evaluates a sequence of coefficients with enhanced security checks.
    Returns np.inf if the input is invalid.
    """
    # --- Security Checks ---

    # Verify that the input is a list
    if not isinstance(sequence, list):
        return np.inf

    # Reject empty lists
    if not sequence:
        return np.inf

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            return np.inf

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            return np.inf

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    n = len(sequence)
    b_sequence = np.convolve(sequence, sequence)
    max_b = max(b_sequence)
    sum_a = np.sum(sequence)

    # Protect against the case where the sum is too close to zero
    if sum_a < 0.01:
        return np.inf

    return float(2 * n * max_b / (sum_a**2))


# --------------------------------------------------------------------------------------
# Command line: score a saved construction.
#     python eval.py candidate.npy
# Accepts .npy, .npz (first array), or .json (a bare list of numbers).
# --------------------------------------------------------------------------------------
def _load(path: str):
    import json
    if path.endswith(".json"):
        with open(path) as fh:
            return np.asarray(json.load(fh), dtype=np.float64)
    arr = np.load(path)
    if hasattr(arr, "files"):          # .npz
        arr = arr[arr.files[0]]
    return np.asarray(arr, dtype=np.float64).ravel()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("usage: python eval.py <candidate.npy|.npz|.json>", file=sys.stderr)
        raise SystemExit(2)
    seq = _load(sys.argv[1]).tolist()
    score = float(evaluate_sequence(seq))
    print(f"n           = {len(seq)}")
    print(f"SCORE       = {score!r}   (lower is better)")
    if not np.isfinite(score):
        print("INVALID: the evaluator rejected this construction.", file=sys.stderr)
        raise SystemExit(1)
