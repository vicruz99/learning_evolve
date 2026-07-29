# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state 58c90071) state=766fe0af sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute pairwise indices for overlap constraints to speed up evaluations
TRI_U_IDX = np.triu_indices(N, k=1)

def compute_constraints(vars_flat):
    """
    Computes all boundary and non-overlap constraints.
    Returns an array where each element must be >= 0.
    """
    X = vars_flat.reshape(N, 3)
    xs, ys, rs = X[:, 0], X[:, 1], X[:, 2]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx_i, idx_j = TRI_U_IDX
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dr = rs[idx_i] + rs[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(c)

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def make_init(cfg, shifts, seed):
    """
    Generates a feasible initial configuration based on a hexagonal-like grid.
    Ensures strict feasibility by sizing radii to 95% of available space.
    """
    rng = np.random.default_rng(seed)
    pts = []
    y = 0.08
    dy = 0.16
    dx = 0.18
    
    for r_idx, count in enumerate(cfg):
        shift = shifts[r_idx] * dx / 2.0
        x = 0.08 + shift
        for _ in range(count):
            pts.append([x + rng.uniform(-0.01, 0.01), y + rng.uniform(-0.01, 0.01)])
            x += dx
        y += dy
        
    pts = np.array(pts[:N])
    
    # Compute strictly feasible initial radii
    rs = np.full(N, 0.5)
    for i in range(N):
        d_bound = min(pts[i,0], 1.0 - pts[i,0], pts[i,1], 1.0 - pts[i,1])
        d_neighbors = min([np.hypot(pts[i,0] - pts[j,0], pts[i,1] - pts[j,1]) 
                          for j in range(N) if i != j], default=2.0)
        rs[i] = min(d_bound, d_neighbors / 2.0) * 0.95
        
    x0 = np.zeros(N * 3)
    for i in range(N):
        x0[3*i] = pts[i,0]
        x0[3*i+1] = pts[i,1]
        x0[3*i+2] = rs[i]
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Diverse row configurations for hexagonal packing
    configs = [
        [5, 5, 5, 5, 5, 1], [5, 6, 5, 6, 4], [6, 5, 6, 5, 4],
        [5, 5, 6, 5, 5], [4, 6, 6, 6, 4], [5, 4, 5, 6, 6],
        [6, 6, 5, 5, 4], [4, 5, 5, 6, 6]
    ]
    
    inits = []
    seed = 42
    for cfg in configs:
        # Try both even-row shifted and odd-row shifted patterns
        for shift in [[i % 2 for i in range(len(cfg))], [(i + 1) % 2 for i in range(len(cfg))]]:
            inits.append(make_init(cfg, shift, seed))
            seed += 1
            
    # Phase 1: Multi-start optimization
    for x0 in inits:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12})
            if np.all(compute_constraints(res.x) >= -1e-6):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        best_x = inits[0]
    
    # Phase 2: Iterative refinement ("Grow and Push")
    # Gradually increase radii and re-optimize positions to escape local minima
    curr_x = best_x.copy()
    for step in range(15):
        # Grow radii slightly
        rs = curr_x[2::3].copy()
        rs *= 1.0015 
        curr_x[2::3] = rs
        
        # Perturb centers slightly to help resolver overlaps
        curr_x[:2*N] += np.random.normal(0, 0.0005, 2*N)
        
        try:
            res = minimize(compute_objective, curr_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12})
            if np.all(compute_constraints(res.x) >= -1e-6):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
                    curr_x = best_x.copy()
        except Exception:
            pass
            
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x.reshape(N, 3)[:, 2]
    
    # Final safety clamp
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
