# sol_000025 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000016 (state 3dc87422) state=3e064e02 sum of radii=2.254672 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

N = 26

def objective(vars):
    """Objective: Minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint_bounds(vars):
    """Boundary constraints: circles must stay inside [0,1]x[0,1]."""
    cx = vars[0::3]
    cy = vars[1::3]
    cr = vars[2::3]
    return np.concatenate([
        cx - cr,          # x >= r
        1.0 - cx - cr,    # x + r <= 1
        cy - cr,          # y >= r
        1.0 - cy - cr     # y + r <= 1
    ])

def constraint_overlap(vars):
    """Non-overlap constraints: distance between centers >= sum of radii."""
    cx = vars[0::3]
    cy = vars[1::3]
    cr = vars[2::3]
    
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sr = cr[:, None] + cr[None, :]
    
    # Only consider upper triangle to avoid duplicates and self-comparison
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    return dist[mask] - sr[mask]

def get_valid_radii(centers):
    """Compute the maximum valid radii for a fixed set of centers."""
    n = centers.shape[0]
    # Distance to boundaries
    rb = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Distance to nearest neighbor
    dists = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    # Resulting radii are limited by the tighter constraint
    return np.minimum(rb, rp)

def init_hex_centers(seed):
    """Generate an initial hexagonal lattice configuration."""
    np.random.seed(seed)
    c = []
    r0 = 0.09
    y = r0
    row = 0
    while len(c) < N:
        x = r0 + (row % 2) * r0
        while x <= 1 - r0 and len(c) < N:
            c.append([x, y])
            x += 2 * r0
        y += r0 * math.sqrt(3)
        row += 1
    c = np.array(c[:N])
    # Add small perturbation and clamp to ensure initial feasibility
    c += np.random.normal(0, 0.003, (N, 2))
    return np.clip(c, 0.12, 0.88)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize sum of radii."""
    best_sum = -1.0
    best_c = None
    best_r = None
    
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = [
        {'type': 'ineq', 'fun': constraint_bounds},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Multi-start optimization to escape local minima
    for seed in range(15):
        c0 = init_hex_centers(seed)
        r0 = np.full(N, 0.08)
        
        x0 = np.zeros(3 * N)
        x0[0::3] = c0[:, 0]
        x0[1::3] = c0[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
            
            # Extract optimized centers
            c_opt = np.column_stack((res.x[0::3], res.x[1::3]))
            # Strictly enforce constraints by recomputing max possible radii
            r_opt = get_valid_radii(c_opt)
            
            s = np.sum(r_opt)
            if s > best_sum:
                best_sum = s
                best_c = c_opt.copy()
                best_r = r_opt.copy()
        except Exception:
            continue
            
    return best_c, best_r, float(best_sum)
