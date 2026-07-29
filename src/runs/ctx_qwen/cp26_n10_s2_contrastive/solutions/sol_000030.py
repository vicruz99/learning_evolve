# sol_000030 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 62f34940) state=596cf150 sum of radii=2.621353 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def objective_func(x, n):
    """Minimize negative sum of radii to maximize total sum."""
    return -np.sum(x[2 * n :])

def constraint_func(x, n):
    """
    Compute all inequality constraints as a single vectorized array.
    Constraints must be >= 0.
    """
    cx = x[:n]
    cy = x[n : 2 * n]
    r = x[2 * n : 3 * n]

    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, same for y
    c_bound = np.concatenate([cx - r, 1.0 - cx - r, cy - r, 1.0 - cy - r])

    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    iu, ju = np.triu_indices(n, k=1)
    dx = cx[iu] - cx[ju]
    dy = cy[iu] - cy[ju]
    d2 = dx**2 + dy**2
    r_sum = r[iu] + r[ju]
    c_overlap = d2 - r_sum**2

    return np.concatenate([c_bound, c_overlap])

def create_hex_initial(n, noise_level):
    """Generate a hexagonal lattice initialization perturbed by noise."""
    centers = []
    r_init = 0.04  # Start small to guarantee feasibility
    row_spacing = r_init * np.sqrt(3)
    col_spacing = 2 * r_init
    y = r_init + 0.02
    row = 0
    
    while len(centers) < n and y + r_init <= 1.0:
        x = r_init + 0.02 if row % 2 == 0 else r_init + r_init + 0.02
        while x <= 1.0 - r_init - 0.02 and len(centers) < n:
            centers.append([x, y])
            x += col_spacing
        y += row_spacing
        row += 1
        
    cx = np.array([c[0] for c in centers[:n]])
    cy = np.array([c[1] for c in centers[:n]])
    r = np.full(n, r_init)
    
    cx += np.random.uniform(-noise_level, noise_level, n)
    cy += np.random.uniform(-noise_level, noise_level, n)
    cx = np.clip(cx, r_init, 1.0 - r_init)
    cy = np.clip(cy, r_init, 1.0 - r_init)
    
    return np.concatenate([cx, cy, r])

def create_grid_initial(n, noise_level):
    """Generate a square grid initialization perturbed by noise."""
    cx = np.repeat(np.linspace(0.15, 0.85, 5), 5)
    cy = np.tile(np.linspace(0.15, 0.85, 5), 5)
    
    # Add extra points if needed
    if len(cx) < n:
        cx = np.append(cx, 0.5)
        cy = np.append(cy, 0.5)
        
    cx = cx[:n] + np.random.uniform(-noise_level, noise_level, n)
    cy = cy[:n] + np.random.uniform(-noise_level, noise_level, n)
    r = np.full(n, 0.04)
    
    cx = np.clip(cx, 0.05, 0.95)
    cy = np.clip(cy, 0.05, 0.95)
    
    return np.concatenate([cx, cy, r])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0.0
    best_x = None
    
    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraint_func, 'args': (n,)}
    
    # Strategy 1: Hexagonal starts with varying noise
    for trial in range(8):
        np.random.seed(trial * 13 + 7)
        noise = 0.01 + trial * 0.005
        x0 = create_hex_initial(n, noise)
        
        res = minimize(
            objective_func, x0, method='SLSQP', args=(n,),
            bounds=bounds, constraints=cons,
            options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False}
        )
        
        if not np.isnan(res.fun) and not np.isinf(res.fun):
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_x = res.x.copy()

    # Strategy 2: Grid starts
    for trial in range(5):
        np.random.seed(trial * 17 + 3)
        noise = 0.01 + trial * 0.005
        x0 = create_grid_initial(n, noise)
        
        res = minimize(
            objective_func, x0, method='SLSQP', args=(n,),
            bounds=bounds, constraints=cons,
            options={'maxiter': 12000, 'ftol': 1e-12, 'disp': False}
        )
        
        if not np.isnan(res.fun) and not np.isinf(res.fun):
            s = -res.fun
            if s > best_sum:
                best_sum = s
                best_x = res.x.copy()

    # Strategy 3: Perturb and refine the best solution found
    if best_x is not None:
        for k in range(10):
            np.random.seed(k * 31 + 5)
            x_curr = best_x + np.random.randn(3 * n) * 0.003
            # Ensure radii stay positive after perturbation
            x_curr[2*n:] = np.maximum(x_curr[2*n:], 0.0)
            
            res = minimize(
                objective_func, x_curr, method='SLSQP', args=(n,),
                bounds=bounds, constraints=cons,
                options={'maxiter': 15000, 'ftol': 1e-13, 'disp': False}
            )
            
            if not np.isnan(res.fun) and not np.isinf(res.fun):
                s = -res.fun
                if s > best_sum:
                    best_sum = s
                    best_x = res.x.copy()
                    
    # Extract and clean up results
    cx = best_x[:n]
    cy = best_x[n : 2 * n]
    r = best_x[2 * n : 3 * n]
    
    # Ensure strict boundary compliance
    r = np.maximum(r, 1e-9)
    for i in range(n):
        max_r = min(cx[i], 1.0 - cx[i], cy[i], 1.0 - cy[i])
        if r[i] > max_r:
            r[i] = max_r
            
    # Iteratively resolve any residual overlaps from numerical tolerance
    for _ in range(100):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = math.hypot(cx[i] - cx[j], cy[i] - cy[j])
                if d < r[i] + r[j] - 1e-9:
                    excess = r[i] + r[j] - d
                    r[i] -= excess / 2.0
                    r[j] -= excess / 2.0
                    changed = True
        if not changed:
            break
            
    centers = np.column_stack([cx, cy])
    return centers, r, float(np.sum(r))
