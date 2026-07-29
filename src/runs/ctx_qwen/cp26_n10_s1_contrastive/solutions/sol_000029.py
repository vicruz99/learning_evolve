# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000015 (state cc21d5f7) state=46a9682b sum of radii=1.539661 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, differential_evolution

N_CIRCLES = 26

# Precompute the constant structure of the inequality constraint matrix A_ub.
# Rows 0 to 4N-1: Boundary constraints (r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y)
# Rows 4N onwards: Pairwise constraints (r_i + r_j <= dist_ij)
_A_Ub_CONST = np.zeros((4 * N_CIRCLES + N_CIRCLES * (N_CIRCLES - 1) // 2, N_CIRCLES))
for i in range(N_CIRCLES):
    _A_Ub_CONST[4*i, i] = 1.0
    _A_Ub_CONST[4*i+1, i] = 1.0
    _A_Ub_CONST[4*i+2, i] = 1.0
    _A_Ub_CONST[4*i+3, i] = 1.0
p = 4 * N_CIRCLES
for i in range(N_CIRCLES):
    for j in range(i + 1, N_CIRCLES):
        _A_Ub_CONST[p, i] = 1.0
        _A_Ub_CONST[p, j] = 1.0
        p += 1

def solve_lp(centers):
    """
    Given fixed centers, solves the LP to find radii that maximize sum(radii).
    Returns (max_sum_radii, optimal_radii).
    """
    n = centers.shape[0]
    c = -np.ones(n)  # Maximize sum(r) <=> Minimize -sum(r)
    
    # Construct right-hand side vector b_ub dynamically based on centers
    b = np.empty(4 * n + n * (n - 1) // 2)
    idx = 0
    for i in range(n):
        b[idx] = centers[i, 0]; idx += 1
        b[idx] = 1.0 - centers[i, 0]; idx += 1
        b[idx] = centers[i, 1]; idx += 1
        b[idx] = 1.0 - centers[i, 1]; idx += 1
        
    # Compute pairwise Euclidean distances efficiently
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            b[idx] = dists[i, j]
            idx += 1
            
    bounds = [(0, None)] * n
    res = linprog(c, A_ub=_A_Ub_CONST, b_ub=b, bounds=bounds, method='highs')
    
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(n)

def de_objective(x_flat):
    """Objective for Differential Evolution: minimize negative sum of radii."""
    centers = x_flat.reshape(N_CIRCLES, 2)
    s, _ = solve_lp(centers)
    return -s

def run_packing():
    n = N_CIRCLES
    # Keep centers slightly away from exact boundaries to allow positive radii
    bounds = [(0.02, 0.98)] * (2 * n)
    
    best_centers = None
    best_sum = 0.0
    
    # Stage 1: Global search for center coordinates using Differential Evolution
    # Running multiple seeds increases the chance of finding the global optimum
    for seed in range(3):
        try:
            res = differential_evolution(
                de_objective, bounds,
                popsize=15, maxiter=300,
                mutation=(0.5, 1.0), recombination=0.7,
                seed=seed + 200, polishing=False
            )
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_centers = res.x.reshape(n, 2)
        except Exception:
            pass
            
    if best_centers is None:
        best_centers = np.random.rand(n, 2) * 0.9 + 0.05
        
    # Stage 2: Local refinement / Hill Climbing on centers
    # Fine-tunes positions to squeeze out additional radius sum
    for _ in range(60):
        improved = False
        idx = np.random.randint(0, n)
        cx, cy = best_centers[idx]
        for _ in range(8):
            dx = np.random.uniform(-0.015, 0.015)
            dy = np.random.uniform(-0.015, 0.015)
            new_c = best_centers.copy()
            new_c[idx, 0] = np.clip(cx + dx, 0.02, 0.98)
            new_c[idx, 1] = np.clip(cy + dy, 0.02, 0.98)
            s_new, _ = solve_lp(new_c)
            if s_new > best_sum + 1e-9:
                best_centers = new_c
                best_sum = s_new
                improved = True
                break
        if not improved:
            break
            
    # Final LP solve to extract radii corresponding to the optimized centers
    final_sum, radii = solve_lp(best_centers)
    return best_centers, radii, float(final_sum)
