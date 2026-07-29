# sol_000051 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state ce356e52) state=72453efc sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint(vars):
    """Returns inequality constraints >= 0 for boundary and non-overlap."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    c = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Pairwise non-overlap: dist_sq - (r_i + r_j)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx**2 + dy**2
    
    r_sum = r[:, None] + r[None, :]
    r_sum_sq = r_sum**2
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    c_pair = dist_sq[mask] - r_sum_sq[mask]
    
    return np.concatenate([c, c_pair])

def get_hex_init(r_init):
    """Generate a hexagonal lattice initialization."""
    pts = []
    y = r_init
    row = 0
    while len(pts) < N:
        x = r_init if row % 2 == 0 else 2.0 * r_init
        while x <= 1.0 - r_init and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3.0) * r_init
        row += 1
    return np.array(pts[:N])

def get_spread_init(seed):
    """Generate evenly spread points using repulsion simulation."""
    np.random.seed(seed)
    pts = np.random.rand(N, 2) * 0.8 + 0.1
    for _ in range(300):
        diff = pts[:, None, :] - pts[None, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1e6)
        # 1/r^2 repulsion force
        forces = np.sum((diff / dists[:, :, None]**3), axis=1)
        # Wall repulsion
        forces += (pts < 0.15).astype(float) * 5.0
        forces -= (pts > 0.85).astype(float) * 5.0
        pts += 0.002 * forces
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def run_opt(vars0):
    """Run SLSQP optimization from a given starting vector."""
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    res = minimize(objective, vars0, method='SLSQP', bounds=bounds,
                   constraints={'type': 'ineq', 'fun': constraint},
                   options={'maxiter': 2000, 'ftol': 1e-10})
    return res.x, -res.fun

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    best_vars = None
    best_sum = -np.inf
    
    # Phase 1: Diverse initializations
    candidates = []
    
    # Hexagonal patterns with varying base radii
    for r_est in [0.085, 0.095, 0.105]:
        c = get_hex_init(r_est)
        candidates.append((c, np.full(N, r_est * 0.9)))
        
    # Force-spread patterns
    for s in range(4):
        c = get_spread_init(s)
        # Compute geometrically safe initial radii
        dists = np.sqrt(((c[:, None] - c[None, :])**2).sum(axis=2))
        np.fill_diagonal(dists, np.inf)
        mind = dists.min(axis=1)
        wall = np.minimum(np.minimum(c[:, 0], 1.0 - c[:, 0]), 
                          np.minimum(c[:, 1], 1.0 - c[:, 1]))
        r = np.minimum(mind / 2.0, wall) * 0.85
        candidates.append((c, r))
        
    # Run optimization on each candidate
    for c_init, r_init in candidates:
        v0 = np.zeros(3 * N)
        v0[0::3] = c_init[:, 0]
        v0[1::3] = c_init[:, 1]
        v0[2::3] = r_init
        
        v_opt, s_opt = run_opt(v0)
        if s_opt > best_sum:
            best_sum = s_opt
            best_vars = v_opt
            
        # Perturbation-based local search to escape local minima
        for _ in range(3):
            v_pert = best_vars + np.random.normal(0, 0.003, 3 * N)
            v_pert[0::3] = np.clip(v_pert[0::3], 0.0, 1.0)
            v_pert[1::3] = np.clip(v_pert[1::3], 0.0, 1.0)
            v_pert[2::3] = np.clip(v_pert[2::3], 0.0, 0.5)
            v_opt2, s_opt2 = run_opt(v_pert)
            if s_opt2 > best_sum:
                best_sum = s_opt2
                best_vars = v_opt2
                
    # Phase 2: High-precision refinement on the best found configuration
    if best_vars is not None:
        res_final = minimize(objective, best_vars, method='SLSQP',
                             bounds=[(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N,
                             constraints={'type': 'ineq', 'fun': constraint},
                             options={'maxiter': 5000, 'ftol': 1e-13})
        best_vars = res_final.x
        best_sum = -res_final.fun
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    return centers, radii, float(best_sum)
