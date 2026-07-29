# sol_000265 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=1deb4dea sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, differential_evolution

N_CIRCLES = 26

# Precompute the constant structure of the LP constraint matrix
# Structure: r_i + r_j <= d_ij  (pairwise)
#            r_i <= b_i        (boundary)
def _setup_lp_matrices():
    n = N_CIRCLES
    n_pairs = n * (n - 1) // 2
    n_constraints = n_pairs + n
    A = np.zeros((n_constraints, n))
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A[idx, i] = 1.0
            A[idx, j] = 1.0
            idx += 1
    for i in range(n):
        A[idx, i] = 1.0
        idx += 1
    return A, np.ones(n)

A_LP, c_LP = _setup_lp_matrices()

def compute_max_sum_radii(centers):
    """
    Given fixed centers, solve the LP to find optimal radii maximizing sum(r_i).
    Returns (sum_radii, radii_array).
    """
    n = N_CIRCLES
    x, y = centers[:, 0], centers[:, 1]
    # Maximum allowed radius by boundaries
    b = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    
    # Pairwise Euclidean distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Build RHS vector for LP
    b_ub = np.zeros(A_LP.shape[0])
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            b_ub[idx] = dists[i, j]
            idx += 1
    for i in range(n):
        b_ub[idx] = b[i]
        idx += 1
        
    try:
        # Maximize sum(r) <=> Minimize -sum(r)
        res = linprog(c_LP, A_ub=A_LP, b_ub=b_ub, 
                      bounds=[(0, None)] * n, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def objective(centers_flat):
    """Objective function for DE: negative of the max sum of radii."""
    centers = centers_flat.reshape(N_CIRCLES, 2)
    val, _ = compute_max_sum_radii(centers)
    return -val

def run_packing():
    # Bounds for 26 centers in [0,1]^2
    bounds = [(0, 1)] * (2 * N_CIRCLES)
    
    # Global optimization of center positions
    res = differential_evolution(
        objective, 
        bounds, 
        popsize=25, 
        maxiter=1000, 
        seed=42, 
        workers=1,
        tol=1e-9,
        atol=1e-9
    )
    
    centers = res.x.reshape(N_CIRCLES, 2)
    
    # Compute optimal radii for the found centers
    _, radii = compute_max_sum_radii(centers)
    
    # Numerical safety: clamp radii to non-negative
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)
