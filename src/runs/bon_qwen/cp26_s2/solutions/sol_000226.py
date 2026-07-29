# sol_000226 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=ff83802e sum of radii=1.462770 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_repulsion_forces(centers, r, n):
    """Compute pairwise repulsion and boundary forces."""
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Avoid division by zero
    safe_dists = np.where(dists > 1e-8, dists, 1e-8)
    mask = dists < 2 * r
    rep_force_mag = np.where(mask, (2 * r - dists) / safe_dists, 0.0)
    
    forces = np.zeros_like(centers)
    # Apply pairwise repulsion
    forces += np.einsum('ijk,ij->ik', diff, rep_force_mag)
    
    # Apply boundary repulsion
    forces += np.where(centers < r, (r - centers) * 5.0, 0.0)
    forces += np.where(centers > 1.0 - r, (centers - (1.0 - r)) * -5.0, 0.0)
    
    return forces, dists

def optimize_centers(n, seed=42):
    """Run repulsion simulation to find well-spread centers."""
    rng = np.random.default_rng(seed)
    centers = rng.random((n, 2)) * 0.7 + 0.15
    
    r = 0.04
    step = 0.015
    best_r = r
    best_centers = centers.copy()
    
    for it in range(30000):
        forces, dists = compute_repulsion_forces(centers, r, n)
        centers += step * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        if it % 250 == 0:
            step *= 0.985
            # Validate current configuration
            valid = True
            if np.any(centers < r) or np.any(centers > 1.0 - r):
                valid = False
            elif np.min(dists) < 2 * r - 1e-6:
                valid = False
                
            if valid:
                best_r = r
                best_centers = centers.copy()
                r += 0.00025
                step = max(step, 0.003)
            else:
                r = max(r - 0.0001, 0.01)
                step = max(step, 0.002)
                
    return best_centers

def solve_radii_lp(centers):
    """Solve LP to maximize sum of radii for fixed centers."""
    n = centers.shape[0]
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    c = -np.ones(n)
    A_ub = []
    b_ub = []
    eps = 1e-10
    
    # Pairwise constraints: r_i + r_j <= dist(i,j) - eps
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j] - eps)
            
        # Boundary constraints: r_i <= min(x, 1-x, y, 1-y) - eps
        bounds_vals = [
            centers[i, 0],
            1.0 - centers[i, 0],
            centers[i, 1],
            1.0 - centers[i, 1]
        ]
        for b_val in bounds_vals:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b_val - eps)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    # Fallback: equal radii based on min distance
    min_dist = np.min(dists)
    min_bound = min(centers.min(axis=0), 1.0 - centers.max(axis=0)).min()
    r = min(min_dist / 2.0, min_bound)
    return np.full(n, max(r, 0.0))

def run_packing():
    n = 26
    centers = optimize_centers(n)
    radii = solve_radii_lp(centers)
    radii = np.maximum(radii, 0.0)
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii
