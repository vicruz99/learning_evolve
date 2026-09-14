"""Grading function for the Erdos minimum overlap problem, copied byte-for-byte from
src/envs/erdos_min_overlap.py (verify_c5_solution). DO NOT MODIFY -- this is the
function your candidates are scored with.
"""
import numpy as np


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
    h = _load(sys.argv[1])
    n_points = h.shape[0]
    dx = 2.0 / n_points
    computed = float(np.max(np.correlate(h, 1.0 - h, mode="full") * dx))
    # Re-run the real verifier with the value we just computed, so the [0,1] range check,
    # the renormalisation check and the 1e-4 agreement check all fire exactly as they will
    # when your solution is graded.
    try:
        c5 = float(verify_c5_solution(h, computed, n_points))
    except ValueError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(f"n_points    = {n_points}")
    print(f"sum(h)      = {float(h.sum())!r}   (target {n_points / 2.0!r})")
    print(f"C5          = {c5!r}   (lower is better)")
