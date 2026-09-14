"""Grading function for AC2 (autocorrelation inequality, lower bound), copied byte-for-byte from
src/envs/ac_inequalities.py (evaluate_sequence_ac2). DO NOT MODIFY -- this is the
function your candidates are scored with.
"""
import numpy as np


def evaluate_sequence(sequence: list[float]) -> float:
    # Verify that the input is a list
    if not isinstance(sequence, list):
        raise ValueError("Invalid sequence type")

    # Reject empty lists
    if not sequence:
        raise ValueError("Empty sequence")

    # Check each element in the list for validity
    for x in sequence:
        # Reject boolean types (as they are a subclass of int) and
        # any other non-integer/non-float types (like strings or complex numbers).
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise ValueError("Invalid sequence element type")

        # Reject Not-a-Number (NaN) and infinity values.
        if np.isnan(x) or np.isinf(x):
            raise ValueError("Invalid sequence element value")

    # Convert all elements to float for consistency
    sequence = [float(x) for x in sequence]

    # Protect against negative numbers
    sequence = [max(0, x) for x in sequence]

    # Check if sum of sequence will be too close to zero
    if np.sum(sequence) < 0.01:
        raise ValueError("Sum of sequence is too close to zero.")
    
    # Protect against numbers that are too large
    sequence = [min(1000.0, x) for x in sequence]

    convolution_2 = np.convolve(sequence, sequence)
    # --- Security Checks ---

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
        # Integral of (mx + c)^2 = h/3 * (y1^2 + y1*y2 + y2^2) where m = (y2-y1)/h, c = y1 - m*x1, interval is [x1, x2], y1 = mx1+c, y2=mx2+c
        interval_l2_squared = (h / 3) * (y1**2 + y1 * y2 + y2**2)
        l2_norm_squared += interval_l2_squared

    # Calculate the 1-norm: ||f*f||_1
    norm_1 = np.sum(np.abs(convolution_2)) / (len(convolution_2) + 1)

    # Calculate the infinity-norm: ||f*f||_inf
    norm_inf = np.max(np.abs(convolution_2))
    C_lower_bound = l2_norm_squared / (norm_1 * norm_inf)
    return C_lower_bound


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
    print(f"SCORE       = {score!r}   (higher is better)")
    if not np.isfinite(score):
        print("INVALID: the evaluator rejected this construction.", file=sys.stderr)
        raise SystemExit(1)
