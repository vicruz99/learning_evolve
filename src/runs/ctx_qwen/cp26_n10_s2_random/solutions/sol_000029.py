# sol_000029 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 3dc87422) state=c59903fd sum of radii=2.620759 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
# Precompute indices for pairwise constraints to speed up evaluation
TRI_U_IND = np.triu_indices(N_CIRCLES, k=1)

def compute_constraints(x):
    """Computes all constraint values. Must be >= 0 for validity."""
    n = N_CIRCLES
    c = x[:2*n].reshape(n, 2)
    r = x[2*n:]
    
    # Boundary constraints: circle inside [0,1]x[0,1]
    b1 = c[:, 0] - r
    b2 = 1.0 - c[:, 0] - r
    b3 = c[:, 1] - r
    b4 = 1.0 - c[:, 1] - r
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    # Extract only upper triangle pairs to avoid duplicates and self-comparison
    ovl = dist[TRI_U_IND] - r_sum[TRI_U_IND]
    
    # Non-negative radii
    nr = r
    
    return np.concatenate([b1, b2, b3, b4, ovl, nr])

def compute_objective(x):
    """Objective: minimize negative sum of radii."""
    n = N_CIRCLES
    return -np.sum(x[2*n:])

def generate_initial_guess(seed):
    """Generates a feasible initial configuration using a perturbed hexagonal grid."""
    rng = np.random.default_rng(seed)
    n = N_CIRCLES
    r_init = 0.09
    
    # Generate hexagonal lattice points
    centers = []
    y = r_init
    row = 0
    while len(centers) < n + 5:
        x = r_init if row % 2 == 0 else r_init * 1.5
        while x < 1.0 - r_init:
            centers.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
        
    centers = np.array(centers[:n+5])
    # Randomly select n points to break perfect symmetry
    idx = rng.choice(len(centers), n, replace=False)
    centers = centers[idx]
    
    # Add small random perturbation
    centers += rng.normal(0, 0.005, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    r = np.full(n, r_init)
    return np.concatenate([centers.flatten(), r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    best_obj = np.inf
    best_vars = None
    
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.0, 0.5)] * n
    
    # Run multiple restarts to escape local minima
    for seed in range(15):
        x0 = generate_initial_guess(seed)
        
        cons = {'type': 'ineq', 'fun': compute_constraints}
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP',
                           bounds=bounds, constraints=cons,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            if res.fun < best_obj:
                # Verify constraints are satisfied within tolerance
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-8):
                    best_obj = res.fun
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    # Fallback if optimization fails completely
    if best_vars is None:
        best_vars = generate_initial_guess(0)
        
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    
    # Post-processing: ensure strict validity against validator tolerance
    c_vals = compute_constraints(best_vars)
    min_val = np.min(c_vals)
    if min_val < -1e-9:
        margin = min_val + 1e-9
        radii -= margin
        radii = np.maximum(radii, 0.0)
        
    sum_r = np.sum(radii)
    return centers, radii, float(sum_r)
