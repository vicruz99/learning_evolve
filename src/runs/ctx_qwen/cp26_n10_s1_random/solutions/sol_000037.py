# sol_000037 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000021 (state e14e8c08) state=c6261c32 sum of radii=1.920288 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_constraints_equal(vars_flat, n):
    """
    Computes inequality constraints for equal-radius packing.
    Constraints: boundary distances >= t, pairwise squared distances >= 4*t^2
    Returns an array of constraint values (all must be >= 0).
    """
    t = vars_flat[-1]
    xs = vars_flat[0::2]
    ys = vars_flat[1::2]
    
    c = []
    # Boundary constraints: x >= t, 1-x >= t, y >= t, 1-y >= t
    c.extend(xs - t)
    c.extend(1.0 - xs - t)
    c.extend(ys - t)
    c.extend(1.0 - ys - t)
    
    # Pairwise squared distance constraints: ||c_i - c_j||^2 >= 4*t^2
    # Vectorized calculation for efficiency
    xs_col = xs[:, np.newaxis]
    ys_col = ys[:, np.newaxis]
    dx = xs_col - xs[np.newaxis, :]
    dy = ys_col - ys[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    # Ignore self-distances
    np.fill_diagonal(dist_sq, np.inf)
    # Flatten and filter out infinity
    c.extend(dist_sq[dist_sq < np.inf] - 4.0 * t * t)
    
    return np.array(c)

def objective_equal(vars_flat):
    """Objective: maximize t => minimize -t"""
    return -vars_flat[-1]

def get_hex_init(r_guess, n):
    """Generates an initial hexagonal grid of n circles with estimated radius r_guess."""
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

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    np.random.seed(42)
    
    # Run multiple optimization trials with perturbed initializations
    for trial in range(5):
        init_centers = get_hex_init(0.09, n)
        if trial > 0:
            init_centers += np.random.uniform(-0.015, 0.015, init_centers.shape)
            init_centers = np.clip(init_centers, 0.05, 0.95)
            
        # Variables: [x0, y0, x1, y1, ..., x25, y25, t]
        x0 = np.concatenate([init_centers.flatten(), [0.08]])
        
        # Bounds: centers in [0, 1], radius t in [0.05, 0.12]
        bounds = [(0.0, 1.0)] * (2 * n) + [(0.05, 0.12)]
        
        cons = {'type': 'ineq', 'fun': compute_constraints_equal, 'args': (n,)}
        
        try:
            res = minimize(
                objective_equal, 
                x0, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 1500, 'ftol': 1e-9}
            )
            # Check if we found a configuration with radius > 0.101 (target ~0.10139)
            if res.x[-1] > 0.101:
                best_centers = res.x[:2 * n].reshape(n, 2)
                best_radii = np.full(n, res.x[-1])
                best_sum = np.sum(best_radii)
                break  # Found a high-quality packing, proceed to refine
        except Exception:
            continue

    # Fallback if optimization didn't reach the target threshold
    if best_centers is None:
        best_centers = get_hex_init(0.09, n)
        best_radii = np.full(n, 0.09)
        best_sum = 0.09 * n

    # Final geometric refinement: compute exact maximal valid radii for the optimized centers
    # This guarantees strict validity against the checker's tolerance
    radii = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        min_d = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                    best_centers[i, 1], 1.0 - best_centers[i, 1])
        # Distance to other circles
        for j in range(n):
            if i != j:
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                if d < min_d:
                    min_d = d
        # Assign radius with a tiny buffer for numerical safety
        radii[i] = min_d / 2.0 - 1e-8
        
    sum_radii = np.sum(radii)
    return best_centers, radii, sum_radii
