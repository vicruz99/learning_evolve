# sol_000060 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000034 (state 766fe0af) state=aaa24c3a sum of radii=2.620919 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
TRiu_IDX = np.triu_indices(N, k=1)

def compute_constraints(vars_flat):
    """
    Computes all boundary and non-overlap constraints.
    Returns an array where each element must be >= 0.
    """
    X = vars_flat.reshape(N, 3)
    xs, ys, rs = X[:, 0], X[:, 1], X[:, 2]
    
    c = []
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    i_idx, j_idx = TRiu_IDX
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dr = rs[i_idx] + rs[j_idx]
    dists = np.sqrt(dx**2 + dy**2)
    c.append(dists - dr)
    
    return np.concatenate(c)

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Returns variable bounds for x, y, r for each circle."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)])
    return b

def make_hex_init(rows_counts, shift_pattern, seed, scale=0.95):
    """
    Generates a feasible initial configuration based on a hexagonal-like grid.
    Ensures strict feasibility by sizing radii to 90% of available space.
    """
    rng = np.random.default_rng(seed)
    pts = []
    y = 0.1
    dy = 0.1732  # approx sqrt(3)/2 * 0.2
    dx = 0.2
    
    for r_idx, count in enumerate(rows_counts):
        shift = shift_pattern[r_idx] * dx / 2.0
        x = 0.1 + shift
        for _ in range(count):
            pts.append([x + rng.uniform(-0.005, 0.005), y + rng.uniform(-0.005, 0.005)])
            x += dx
        y += dy
        
    pts = np.array(pts[:N])
    # Center and scale to stay safely within [0.025, 0.975]
    pts = pts * scale + 0.5 * (1 - scale)
    
    # Compute strictly feasible initial radii
    rs = np.full(N, 0.05)
    for i in range(N):
        d_bound = min(pts[i,0], 1.0 - pts[i,0], pts[i,1], 1.0 - pts[i,1])
        d_neigh = np.min([np.hypot(pts[i,0]-pts[j,0], pts[i,1]-pts[j,1]) 
                          for j in range(N) if i != j])
        rs[i] = min(d_bound, d_neigh / 2.0) * 0.9
    
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
        ([5, 5, 5, 5, 5, 1], [0,1,0,1,0,1], 42),
        ([6, 5, 6, 5, 4], [0,1,0,1,0], 123),
        ([5, 6, 5, 6, 4], [1,0,1,0,1], 456),
        ([5, 5, 6, 5, 5], [0,1,0,1,0], 789),
        ([4, 6, 6, 6, 4], [0,1,0,1,0], 111),
        ([5, 4, 5, 6, 6], [0,1,0,1,0], 222),
        ([6, 6, 5, 5, 4], [1,0,1,0,1], 333),
    ]
    
    inits = []
    for cfg, shifts, seed in configs:
        inits.append(make_hex_init(cfg, shifts, seed))
        
    # Phase 1: Multi-start optimization
    for x0 in inits:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-8):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    if best_x is None:
        best_x = inits[0]
        
    # Phase 2: Iterative "Inflate & Resolve" to escape local minima
    # Gradually grow radii and re-optimize positions
    for step in range(40):
        # Always restart from the best feasible solution found so far
        curr_x = best_x.copy()
        
        # Inflate radii slightly
        rs = curr_x[2::3].copy()
        rs *= 1.0004
        curr_x[2::3] = rs
        
        # Perturb centers slightly to help resolve new overlaps
        pert = np.random.normal(0, 0.0003, 2 * N)
        curr_x[:2*N] += pert
        curr_x[:2*N] = np.clip(curr_x[:2*N], 0.001, 0.999)
        
        try:
            res = minimize(compute_objective, curr_x, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1500, 'ftol': 1e-13})
            if np.all(compute_constraints(res.x) >= -1e-8):
                val = -res.fun
                if val > best_val:
                    best_val = val
                    best_x = res.x.copy()
        except Exception:
            pass
            
    # Final safety clamp to guarantee validation passes within numerical tolerance
    c_vals = compute_constraints(best_x)
    min_v = np.min(c_vals)
    if min_v < -1e-9:
        best_x[2::3] -= (-min_v) * 0.5
        best_x[2::3] = np.maximum(best_x[2::3], 0.0)
        
    centers = best_x.reshape(N, 3)[:, :2]
    radii = best_x.reshape(N, 3)[:, 2]
    
    return centers, radii, float(np.sum(radii))
