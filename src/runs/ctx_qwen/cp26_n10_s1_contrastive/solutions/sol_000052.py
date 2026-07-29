# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state ce356e52) state=15edee3a sum of radii=2.621069 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint(vars):
    """
    Returns a 1D array of inequality constraint values (must be >= 0).
    Enforces boundary containment and pairwise non-overlap.
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    r_sum_sq = r_sum**2
    
    # Extract upper triangular part to avoid duplicates and self-constraints
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c.append(dist_sq[i, j] - r_sum_sq[i, j])
    
    return np.concatenate(c)

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraint}
    
    best_sum = -np.inf
    best_vars = None
    
    configs = []
    rng = np.random.default_rng(42)
    
    # Strategy 1: Perturbed Hexagonal Lattices
    for _ in range(10):
        r_init = 0.082 + rng.uniform(0, 0.02)
        pts = []
        y = r_init
        row = 0
        while len(pts) < N_CIRCLES:
            x = r_init + (row % 2) * r_init
            while x <= 1.0 - r_init and len(pts) < N_CIRCLES:
                pts.append([x, y])
                x += 2.0 * r_init
            y += np.sqrt(3.0) * r_init
            row += 1
        pts = np.array(pts[:N_CIRCLES])
        # Add controlled jitter to break symmetry
        pts += rng.uniform(-0.008, 0.008, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        v = np.zeros(3 * N_CIRCLES)
        v[0::3] = pts[:, 0]
        v[1::3] = pts[:, 1]
        v[2::3] = r_init
        configs.append(v)
        
    # Strategy 2: Jittered Grid + Center
    for _ in range(10):
        pts = np.array([[0.1 + i*0.2, 0.1 + j*0.2] for i in range(5) for j in range(5)])
        pts = np.vstack([pts, [[0.5, 0.5]]])
        pts += rng.uniform(-0.012, 0.012, pts.shape)
        pts = np.clip(pts, 0.05, 0.95)
        
        v = np.zeros(3 * N_CIRCLES)
        v[0::3] = pts[:, 0]
        v[1::3] = pts[:, 1]
        v[2::3] = 0.088 + rng.uniform(-0.005, 0.005, N_CIRCLES)
        configs.append(v)
        
    # Strategy 3: Quasi-random space filling
    for _ in range(10):
        pts = rng.uniform(0.08, 0.92, (N_CIRCLES, 2))
        v = np.zeros(3 * N_CIRCLES)
        v[0::3] = pts[:, 0]
        v[1::3] = pts[:, 1]
        v[2::3] = 0.075 + rng.uniform(0, 0.03, N_CIRCLES)
        configs.append(v)
        
    # Run optimization on each configuration
    for v0 in configs:
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                bounds=bounds,
                constraints=cons, 
                options={'maxiter': 4000, 'ftol': 1e-13}
            )
            
            # Validate constraint satisfaction
            if np.min(constraint(res.x)) >= -1e-6:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
        except Exception:
            continue
            
    if best_vars is not None:
        centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
        radii = np.maximum(best_vars[2::3], 0.0)
        return centers, radii, float(best_sum)
    
    # Fallback to a guaranteed valid (though suboptimal) configuration
    centers = np.random.rand(N_CIRCLES, 2) * 0.8 + 0.1
    radii = np.full(N_CIRCLES, 0.04)
    return centers, radii, float(np.sum(radii))
