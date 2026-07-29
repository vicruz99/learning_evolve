# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state bbbe9bd5) state=e4a8cbeb sum of radii=2.623915 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def get_constraints(vars):
    """
    Computes inequality constraints g(vars) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    c = vars.reshape(N_CIRCLES, 3)
    x = c[:, 0]
    y = c[:, 1]
    r = c[:, 2]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c1 = x - r
    c2 = 1.0 - x - r
    c3 = y - r
    c4 = 1.0 - y - r
    
    # Pairwise non-overlap: dist_sq - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, None] + r[None, :]
    
    # Extract upper triangular part to avoid duplicates
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c5 = dist_sq[i, j] - r_sum[i, j]**2
    
    return np.concatenate([c1, c2, c3, c4, c5])

def objective(vars):
    """Objective: maximize sum of radii <=> minimize negative sum."""
    return -np.sum(vars.reshape(N_CIRCLES, 3)[:, 2])

def make_hex_init():
    """Generates a hexagonal lattice initialization."""
    r = 0.04
    pts = []
    y = r
    row = 0
    while len(pts) < N_CIRCLES:
        x_off = r if row % 2 == 0 else 2 * r
        x = x_off
        while x <= 1.0 - r and len(pts) < N_CIRCLES:
            pts.append([x, y, r])
            x += 2 * r
        y += np.sqrt(3.0) * r
        row += 1
    return np.array(pts[:N_CIRCLES]).flatten()

def make_random_init(seed_val):
    """Generates a strictly feasible random initialization."""
    rng = np.random.RandomState(seed_val)
    c = rng.rand(N_CIRCLES, 2)
    
    # Distance to boundaries
    dists_b = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                         np.minimum(c[:, 1], 1.0 - c[:, 1]))
    
    # Distance to nearest other center
    diff = c[:, None, :] - c[None, :, :]
    dists_c = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists_c, np.inf)
    dists_min = np.min(dists_c, axis=1)
    
    # Set radius to minimum of boundary and half inter-circle distance
    # Scale by 0.6 to ensure strict feasibility for SLSQP start
    r = np.minimum(dists_b, dists_min / 2.0) * 0.6
    return np.column_stack([c, r]).flatten()

def run_packing():
    best_vars = None
    best_sum = -np.inf
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': get_constraints}
    
    # Phase 1: Diverse random starts to explore global landscape
    for seed in range(25):
        x0 = make_random_init(seed)
        res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'iprint': -1})
        if res.success:
            # Explicit feasibility check
            if np.min(get_constraints(res.x)) >= -1e-9:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = res.x.copy()
                    
    # Phase 2: Hexagonal structure start
    x0_hex = make_hex_init()
    res = minimize(objective, x0_hex, method='SLSQP', bounds=bounds,
                   constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'iprint': -1})
    if res.success and np.min(get_constraints(res.x)) >= -1e-9:
        if -res.fun > best_sum:
            best_sum = -res.fun
            best_vars = res.x.copy()
            
    # Phase 3: Local perturbation refinement
    if best_vars is not None:
        # Safety clipping to prevent boundary drift
        best_vars[::3] = np.clip(best_vars[::3], 0.001, 0.999)
        best_vars[1::3] = np.clip(best_vars[1::3], 0.001, 0.999)
        best_vars[2::3] = np.maximum(best_vars[2::3], 0.001)
        
        np.random.seed(42)
        for _ in range(15):
            x_pert = best_vars + np.random.randn(3 * N_CIRCLES) * 0.002
            res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12, 'iprint': -1})
            if res.success and np.min(get_constraints(res.x)) >= -1e-9:
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_vars = res.x.copy()
                    
        # Phase 4: High-precision polish
        res = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                       constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14, 'iprint': -1})
        if res.success and np.min(get_constraints(res.x)) >= -1e-9:
            best_vars = res.x
            
    centers = best_vars.reshape(N_CIRCLES, 3)[:, :2]
    radii = best_vars.reshape(N_CIRCLES, 3)[:, 2]
    return centers, radii, float(np.sum(radii))
