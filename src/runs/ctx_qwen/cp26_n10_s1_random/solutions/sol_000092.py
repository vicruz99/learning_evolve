# sol_000092 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=cbd41491 sum of radii=2.339977 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(vars_flat):
    """Objective: minimize negative common radius t."""
    return -vars_flat[-1]

def compute_constraints(vars_flat, n):
    """
    Computes inequality constraints >= 0 for valid equal-radius packing.
    Variables: [x0, y0, x1, y1, ..., x25, y25, t]
    """
    t = vars_flat[-1]
    xs = vars_flat[0::2]
    ys = vars_flat[1::2]
    
    # Boundary constraints: t <= x <= 1-t, t <= y <= 1-t
    c_bound = np.concatenate([
        xs - t,
        1.0 - xs - t,
        ys - t,
        1.0 - ys - t
    ])
    
    # Pairwise non-overlap constraints: dist(i,j) >= 2t
    # Vectorized distance matrix
    diff_x = xs[:, np.newaxis] - xs[np.newaxis, :]
    diff_y = ys[:, np.newaxis] - ys[np.newaxis, :]
    dists = np.sqrt(diff_x**2 + diff_y**2)
    np.fill_diagonal(dists, np.inf)
    
    # Extract upper triangular part to avoid duplicates
    triu_idx = np.triu_indices(n, k=1)
    c_pair = dists[triu_idx] - 2.0 * t
    
    return np.concatenate([c_bound, c_pair])

def generate_hex_init(r_guess, n):
    """Generates an initial hexagonal grid of n circles."""
    pts = []
    y = r_guess
    row = 0
    while len(pts) < n:
        shift = r_guess if row % 2 == 1 else 0.0
        x = r_guess + shift
        while x + r_guess <= 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2 * r_guess
        y += r_guess * np.sqrt(3)
        row += 1
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_t = 0.0
    best_centers = None
    np.random.seed(42)
    
    configs = []
    
    # 1. Base hexagonal lattice
    configs.append(generate_hex_init(0.10, n))
    
    # 2. Perturbed hexagonal lattices to escape symmetry traps
    for _ in range(15):
        cfg = generate_hex_init(0.10, n) + np.random.uniform(-0.02, 0.02, (n, 2))
        configs.append(np.clip(cfg, 0.05, 0.95))
        
    # 3. Random uniform starts
    for _ in range(10):
        configs.append(np.random.uniform(0.1, 0.9, (n, 2)))
        
    # Bounds for [x0, y0, ..., x25, y25, t]
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.08, 0.12)]
    cons = {'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}
    
    # Optimize from each configuration
    for cfg in configs:
        x0 = np.concatenate([cfg.flatten(), [0.09]])
        try:
            res = minimize(
                objective, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 5000, 'ftol': 1e-10, 'disp': False}
            )
            # Accept if we found a larger feasible radius
            if res.x[-1] > best_t:
                best_t = res.x[-1]
                best_centers = res.x[:2*n].reshape(n, 2)
        except Exception:
            continue
            
    # Fallback if optimization unexpectedly fails
    if best_centers is None:
        best_centers = generate_hex_init(0.09, n)
        best_t = 0.09
        
    # Recompute exact maximum feasible radius from the optimized centers.
    # This guarantees we capture the true geometric limit and avoids SLSQP numerical conservatism.
    min_wall = min(
        np.min(best_centers[:, 0]), 
        np.min(1.0 - best_centers[:, 0]), 
        np.min(best_centers[:, 1]), 
        np.min(1.0 - best_centers[:, 1])
    )
    
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    exact_t = min(min_wall, min_pair)
    
    # Apply safety margin to strictly satisfy 1e-12 tolerance in validator
    final_t = exact_t * 0.99999
    radii = np.full(n, final_t)
    sum_r = float(np.sum(radii))
    
    return best_centers, radii, sum_r
