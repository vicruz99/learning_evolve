# sol_000041 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000007 (state 33c0c451) state=046a36a4 sum of radii=2.628190 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26
I, J = np.triu_indices(N, k=1)

def objective(x):
    """Objective function: maximize sum of radii (minimize negative sum)"""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Inequality constraints:
    - Pairwise distance >= sum of radii
    - Circle boundaries within [0,1]x[0,1]
    Returns array of constraint values (must be >= 0)
    """
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c_bound = np.concatenate([
        cx - r, 
        1.0 - cx - r, 
        cy - r, 
        1.0 - cy - r
    ])
    
    # Overlap constraints: dist(i,j) >= r[i] + r[j]
    dx = cx[I] - cx[J]
    dy = cy[I] - cy[J]
    dists = np.hypot(dx, dy)
    c_overlap = dists - r[I] - r[J]
    
    return np.concatenate([c_bound, c_overlap])

def make_hex_init(seed, density=0.8, pert=0.015):
    """Generate a hexagonal lattice initialization with controlled perturbation."""
    rng = np.random.RandomState(seed)
    s = 0.20 + 0.05 * density  # lattice spacing
    points = []
    row = 0
    while len(points) < N:
        y = 0.05 + row * s * np.sqrt(3)/2
        if y > 0.95: break
        x_start = 0.05 + (row % 2) * s/2
        col = 0
        while x_start + col * s <= 0.95 and len(points) < N:
            points.append([x_start + col * s, y])
            col += 1
        row += 1
        
    points = np.array(points[:N])
    if len(points) < N:
        while len(points) < N:
            points = np.vstack([points, rng.uniform(0.1, 0.9, 2)])
            
    points += rng.randn(*points.shape) * pert
    points = np.clip(points, 0.02, 0.98)
    
    r_init = np.full(N, 0.06 * density)
    return np.column_stack([points, r_init]).flatten()

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = 0.0
    best_x = None
    
    seeds = list(range(50))
    densities = [0.75, 0.85, 0.95, 1.05, 1.15]
    
    # Stage 1: Broad search with hexagonal initializations
    for seed in seeds:
        for d in densities:
            x0 = make_hex_init(seed, density=d, pert=0.012)
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=cons,
                    options={'maxiter': 8000, 'ftol': 1e-13, 'disp': False}
                )
                
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    cx = res.x[0::3]
                    cy = res.x[1::3]
                    r = res.x[2::3]
                    
                    # Quick validity check with tolerance
                    valid = True
                    if np.any(r < -1e-7): valid = False
                    elif np.any(cx < r - 1e-7) or np.any(cx + r > 1 + 1e-7): valid = False
                    elif np.any(cy < r - 1e-7) or np.any(cy + r > 1 + 1e-7): valid = False
                    else:
                        dx = cx[I] - cx[J]
                        dy = cy[I] - cy[J]
                        dists = np.hypot(dx, dy)
                        if np.any(dists < r[I] + r[J] - 1e-7): valid = False
                        
                    if valid:
                        best_sum = curr_sum
                        best_x = res.x.copy()
            except Exception:
                continue
                
    # Stage 2: Local refinement around the best solution found
    if best_x is not None:
        rng = np.random.RandomState(42)
        for _ in range(15):
            x_ref = best_x + rng.randn(len(best_x)) * 0.003
            try:
                res = minimize(
                    objective, x_ref, method='SLSQP', bounds=bounds, constraints=cons,
                    options={'maxiter': 5000, 'ftol': 1e-13}
                )
                if -res.fun > best_sum:
                    cx = res.x[0::3]; cy = res.x[1::3]; r = res.x[2::3]
                    if (np.all(r >= 0) and np.all(cx >= r) and np.all(cx+r <= 1) and 
                        np.all(cy >= r) and np.all(cy+r <= 1)):
                        dx = cx[I]-cx[J]; dy = cy[I]-cy[J]
                        if np.all(np.hypot(dx,dy) >= r[I]+r[J] - 1e-9):
                            best_sum = -res.fun
                            best_x = res.x.copy()
            except Exception:
                pass

    # Fallback initialization
    if best_x is None:
        best_x = make_hex_init(0, 0.9)
        
    # Extract and post-process to guarantee strict validity
    cx = best_x[0::3].copy()
    cy = best_x[1::3].copy()
    r = best_x[2::3].copy()
    
    # Enforce boundary constraints
    for i in range(N):
        max_r = min(cx[i], 1-cx[i], cy[i], 1-cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
    # Iteratively resolve overlaps
    for _ in range(100):
        changed = False
        for i in range(N):
            for j in range(i+1, N):
                dist = math.hypot(cx[i]-cx[j], cy[i]-cy[j])
                if dist < r[i] + r[j] - 1e-10:
                    excess = r[i] + r[j] - dist
                    r[i] -= excess / 2
                    r[j] -= excess / 2
                    changed = True
        if not changed:
            break
            
    r = np.maximum(r, 0.0)
    centers = np.column_stack([cx, cy])
    return centers, r, float(np.sum(r))
