# sol_000043 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 33c0c451) state=5b437714 sum of radii=1.638141 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N_CIRCLES = 26

def get_boundary_max_r(centers):
    """Compute max allowed radius for each center based on square boundaries."""
    x, y = centers[:, 0], centers[:, 1]
    return np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))

def repair_radii_lp(centers):
    """Solve LP to find optimal radii for fixed centers."""
    n = centers.shape[0]
    c_obj = -np.ones(n)  # Maximize sum(r) => minimize -sum(r)
    
    max_rs = get_boundary_max_r(centers)
    max_rs = np.maximum(max_rs, 0.0)
    bounds_r = [(0.0, mr) for mr in max_rs]
    
    # Pairwise constraints: r_i + r_j <= dist_ij
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            b_ub[idx] = math.hypot(dx, dy)
            idx += 1
            
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return res.x
        else:
            return np.full(n, 1e-6)
    except Exception:
        return np.full(n, 1e-6)

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2 * N_CIRCLES:])

def constraints_fn(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    Variable layout: [x1..xN, y1..yN, r1..rN]
    """
    cx = x[:N_CIRCLES]
    cy = x[N_CIRCLES:2 * N_CIRCLES]
    r = x[2 * N_CIRCLES:]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])
    
    # Overlap constraints: dist_ij >= r_i + r_j
    i_idx, j_idx = np.triu_indices(N_CIRCLES, k=1)
    dx = cx[i_idx] - cx[j_idx]
    dy = cy[i_idx] - cy[j_idx]
    dists = np.hypot(dx, dy)
    c_overlap = dists - (r[i_idx] + r[j_idx])
    
    return np.concatenate([c_bound, c_overlap])

def hex_init(seed):
    """Create a hexagonal lattice initialization with slight perturbation."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N_CIRCLES, 2))
    row_counts = [6, 5, 6, 5, 4]  # Sums to 26
    y_vals = np.linspace(0.12, 0.88, 5)
    
    idx = 0
    for r_idx, count in enumerate(row_counts):
        y = y_vals[r_idx]
        x_start = 0.08
        x_end = 0.92
        if r_idx % 2 == 1:
            x_start += 0.04
            x_end -= 0.04
        x_vals = np.linspace(x_start, x_end, count)
        for x in x_vals:
            if idx < N_CIRCLES:
                centers[idx] = [x, y]
                idx += 1
                
    centers += rng.randn(*centers.shape) * 0.01
    centers = np.clip(centers, 0.05, 0.95)
    return centers

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Uses alternating SLSQP + LP repair with multiple restarts and perturbations.
    """
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    bounds = [(0.0, 1.0)] * (2 * N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints_fn}
    
    # Phase 1: Multiple structured restarts
    for seed in range(30):
        centers0 = hex_init(seed)
        r0 = np.full(N_CIRCLES, 0.06)
        x0 = np.concatenate([centers0.ravel(), r0])
        
        # SLSQP pass 1
        res1 = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                        options={'maxiter': 3000, 'ftol': 1e-12})
        
        c1 = res1.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
        r1_lp = repair_radii_lp(c1)
        s1 = np.sum(r1_lp)
        
        if s1 > best_sum:
            best_sum = s1
            best_centers = c1.copy()
            best_radii = r1_lp.copy()
            
        # SLSQP pass 2 from LP-repaired state
        x0_ref = np.concatenate([c1.ravel(), r1_lp])
        res2 = minimize(objective, x0_ref, method='SLSQP', bounds=bounds, constraints=cons,
                        options={'maxiter': 3000, 'ftol': 1e-14})
        
        c2 = res2.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
        r2_lp = repair_radii_lp(c2)
        s2 = np.sum(r2_lp)
        
        if s2 > best_sum:
            best_sum = s2
            best_centers = c2.copy()
            best_radii = r2_lp.copy()
            
    # Phase 2: Perturbation and local search from best found
    for _ in range(15):
        if best_centers is None:
            break
            
        np.random.seed(_)
        pert = best_centers + np.random.randn(N_CIRCLES, 2) * 0.003
        pert = np.clip(pert, 0.02, 0.98)
        r_pert = repair_radii_lp(pert)
        x0_p = np.concatenate([pert.ravel(), r_pert])
        
        res_p = minimize(objective, x0_p, method='SLSQP', bounds=bounds, constraints=cons,
                         options={'maxiter': 2000, 'ftol': 1e-13})
        c_p = res_p.x[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
        r_p_lp = repair_radii_lp(c_p)
        s_p = np.sum(r_p_lp)
        
        if s_p > best_sum:
            best_sum = s_p
            best_centers = c_p.copy()
            best_radii = r_p_lp.copy()
            
    # Fallback safety check
    if best_centers is None:
        best_centers = hex_init(0)
        best_radii = repair_radii_lp(best_centers)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
