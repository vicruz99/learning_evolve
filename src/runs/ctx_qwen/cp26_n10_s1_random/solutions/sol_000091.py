# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=0af58391 sum of radii=2.124511 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_flat, n):
    """
    Objective function: maximize radius r while penalizing boundary and overlap violations.
    vars_flat: [x0, y0, ..., x25, y25, r]
    """
    x = vars_flat[:2*n].reshape(n, 2)
    r = vars_flat[2*n]
    
    # High penalty weight to enforce strict feasibility near the optimum
    mu = 5e6
    
    # Boundary violations: circle must be within [0, 1]
    v_x1 = np.maximum(0, r - x[:, 0])
    v_x2 = np.maximum(0, x[:, 0] + r - 1.0)
    v_y1 = np.maximum(0, r - x[:, 1])
    v_y2 = np.maximum(0, x[:, 1] + r - 1.0)
    bdry_pen = np.sum(v_x1**2 + v_x2**2 + v_y1**2 + v_y2**2)
    
    # Pairwise overlap violations: distance must be >= 2r
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    pair_pen = np.sum(np.maximum(0, 2.0 * r - dists)**2)
    
    # We minimize this function
    return -r + mu * (bdry_pen + pair_pen)

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Bounds for variables: x,y in [0,1], r in [0.05, 0.15]
    bounds = [(0.0, 1.0)] * (2*n) + [(0.05, 0.15)]
    
    best_val = np.inf
    best_vars = None
    
    starts = []
    # Hexagonal row distributions that sum to 26
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5], 
        [4, 6, 6, 6, 4], [6, 6, 5, 5, 4], [5, 6, 4, 6, 5]
    ]
    
    # Generate hexagonal lattice starts
    for dist in row_dists:
        r_init = 0.10
        pts = []
        y = r_init
        for ri, cnt in enumerate(dist):
            shift = r_init if ri % 2 == 1 else 0.0
            x = r_init + shift
            for _ in range(cnt):
                if len(pts) < n:
                    pts.append([x, y])
                x += 2 * r_init
            y += np.sqrt(3) * r_init
            
        pts = np.array(pts[:n])
        # Normalize to fit comfortably inside [0.1, 0.9]
        pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0))
        pts = pts * 0.8 + 0.1
        
        # Create perturbed variants to escape local minima
        for _ in range(3):
            p = pts + np.random.uniform(-0.02, 0.02, pts.shape)
            p = np.clip(p, 0.05, 0.95)
            v = np.zeros(2*n + 1)
            v[:2*n] = p.flatten()
            v[2*n] = 0.095  # Initial feasible radius
            starts.append(v)
            
    # Add fully random starts for diversity
    for _ in range(5):
        v = np.zeros(2*n + 1)
        v[:2*n] = np.random.uniform(0.1, 0.9, 2*n)
        v[2*n] = 0.09
        starts.append(v)
        
    # Optimize from each starting configuration
    for cfg in starts:
        res = minimize(compute_objective, cfg, args=(n,), method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 50000, 'ftol': 1e-14})
        if res.fun < best_val:
            best_val = res.fun
            best_vars = res.x.copy()
            
    best_centers = best_vars[:2*n].reshape(n, 2)
    
    # Post-processing: Compute exact maximum feasible radius for EACH circle individually.
    # This exploits local slack in the packing to maximize the total sum.
    radii = np.zeros(n)
    for i in range(n):
        # Start with boundary constraints
        r = min(best_centers[i, 0], 1.0 - best_centers[i, 0], 
                best_centers[i, 1], 1.0 - best_centers[i, 1])
        
        # Constrain by pairwise distances
        for j in range(n):
            if i != j:
                d = np.linalg.norm(best_centers[i] - best_centers[j])
                r = min(r, d / 2.0)
                
        # Apply tiny safety margin to strictly satisfy 1e-12 validation tolerance
        radii[i] = r * 0.99999
        
    sum_r = float(np.sum(radii))
    
    return best_centers, radii, sum_r
