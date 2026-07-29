# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 50e7db78) state=c0ab670f sum of radii=2.112500 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calculate_penalty(centers, radii, mu):
    """Calculate penalty for overlaps and boundary violations."""
    n = centers.shape[0]
    penalty = 0.0
    
    # Boundary violations: max(0, violation)^2
    v_l = radii - centers[:, 0]
    v_r = radii - (1.0 - centers[:, 0])
    v_b = radii - centers[:, 1]
    v_t = radii - (1.0 - centers[:, 1])
    
    penalty += np.sum(np.maximum(v_l, 0.0)**2)
    penalty += np.sum(np.maximum(v_r, 0.0)**2)
    penalty += np.sum(np.maximum(v_b, 0.0)**2)
    penalty += np.sum(np.maximum(v_t, 0.0)**2)
    
    # Overlap violations: max(0, r_i + r_j - dist_ij)^2
    # Vectorized computation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    
    # Exclude self-interactions by setting diagonal to infinity
    np.fill_diagonal(dist, np.inf)
    
    sum_r = radii[:, np.newaxis] + radii[np.newaxis, :]
    violations = np.maximum(0.0, sum_r - dist)
    
    # Sum over upper triangle (divide by 2 to avoid double counting)
    penalty += np.sum(violations**2) / 2.0
    
    return mu * penalty

def objective(z, n, mu):
    """Objective: minimize negative sum of radii + penalty."""
    r = z[2*n:]
    centers = np.column_stack((z[:n], z[n:2*n]))
    return -np.sum(r) + calculate_penalty(centers, r, mu)

def run_packing():
    n = 26
    
    # 1. Initialization: Hexagonal lattice
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.08) + np.arange(n) * 1e-4  # Small perturbation to break symmetry
    
    r_guess = 0.09
    dx = 2 * r_guess
    dy = np.sqrt(3) * r_guess
    
    idx = 0
    y = r_guess
    for row in range(6):
        x = r_guess + (row % 2) * (dx / 2)
        for col in range(5):
            if idx >= n:
                break
            centers[idx] = [x, y]
            x += dx
            idx += 1
        y += dy
        
    # Ensure initial points are within bounds
    centers = np.clip(centers, 0, 1)
    
    # Flatten to 1D array for optimizer
    z0 = np.concatenate([centers[:, 0], centers[:, 1], radii])
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    # 2. Iterative optimization with increasing penalty
    res = None
    # Sequence of penalty strengths to gradually enforce constraints
    penalty_schedule = [1000, 5000, 20000, 50000]
    
    for mu in penalty_schedule:
        current_z = z0 if res is None else res.x
        res = minimize(
            objective, 
            current_z, 
            args=(n, mu), 
            method='L-BFGS-B', 
            bounds=bounds, 
            options={'maxiter': 10000, 'ftol': 1e-14, 'gtol': 1e-8}
        )
        z0 = res.x
        
    # 3. Extract results
    centers_opt = np.column_stack((res.x[:n], res.x[n:2*n]))
    radii_opt = res.x[2*n:]
    total_sum = np.sum(radii_opt)
    
    return centers_opt, radii_opt, total_sum
