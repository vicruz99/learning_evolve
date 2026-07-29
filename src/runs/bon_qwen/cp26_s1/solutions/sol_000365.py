# sol_000365 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=b4cd445a sum of radii=2.144106 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def calculate_objective(vars):
    """
    Calculates the objective function value.
    Objective: Maximize sum of radii -> Minimize -sum(radii) + penalties.
    vars is a flattened array of [x1, y1, r1, x2, y2, r2, ...]
    """
    xs = vars[0::3]
    ys = vars[1::3]
    rs = vars[2::3]
    
    sum_r = np.sum(rs)
    
    # Boundary penalties
    # Constraints: r <= x <= 1-r  =>  x-r >= 0, 1-r-x >= 0
    # Violation 1: x < r  =>  r - x > 0
    # Violation 2: x > 1-r =>  x + r - 1 > 0
    # Same for y
    
    v1 = rs - xs # Should be <= 0
    v2 = xs + rs - 1.0 # Should be <= 0
    v3 = rs - ys
    v4 = ys + rs - 1.0
    
    penalty_boundary = 0.0
    if np.any(v1 > 0): penalty_boundary += np.sum(v1[v1 > 0]**2)
    if np.any(v2 > 0): penalty_boundary += np.sum(v2[v2 > 0]**2)
    if np.any(v3 > 0): penalty_boundary += np.sum(v3[v3 > 0]**2)
    if np.any(v4 > 0): penalty_boundary += np.sum(v4[v4 > 0]**2)
    
    # Overlap penalties
    # Distance matrix calculation
    centers = np.column_stack((xs, ys))
    # diff[i, j] = center[i] - center[j]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    
    radii_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
    
    # Violation: radii_sum > dist
    violations = radii_sum - dist
    
    # Zero out diagonal (self-distance)
    np.fill_diagonal(violations, 0)
    
    # Sum squared positive violations
    # Note: This sums both (i,j) and (j,i), effectively 2 * sum of unique pairs.
    penalty_overlap = np.sum(np.maximum(violations, 0)**2)
    
    w_b = 10000.0
    w_o = 10000.0
    
    return -sum_r + w_b * penalty_boundary + w_o * penalty_overlap

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    np.random.seed(42)
    
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, 0.05)
    
    # Grid initialization: 5x6 grid = 30 points. Take first 26.
    xs = np.linspace(0.1, 0.9, 5)
    ys = np.linspace(0.1, 0.9, 6)
    
    idx = 0
    for y in ys:
        for x in xs:
            if idx < N_CIRCLES:
                centers[idx] = [x, y]
                idx += 1
            else:
                break
        if idx >= N_CIRCLES:
            break
            
    # Add small random noise to break symmetry
    noise = np.random.uniform(-0.02, 0.02, centers.shape)
    centers += noise
    centers = np.clip(centers, 0, 1)
    
    # Flatten variables
    x0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = [(0, 1)] * (2 * N_CIRCLES) + [(0, 1)] * N_CIRCLES

    # Optimization
    res = minimize(calculate_objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 20000, 'ftol': 1e-15, 'gtol': 1e-15})
    
    opt_xs = res.x[0::3]
    opt_ys = res.x[1::3]
    opt_rs = res.x[2::3]
    
    centers_opt = np.column_stack((opt_xs, opt_ys))
    radii_opt = opt_rs
    
    # Post-processing to ensure strict validity
    # 1. Non-negative radii
    radii_opt = np.maximum(radii_opt, 0)
    
    # 2. Clamp centers to fit radii
    for i in range(N_CIRCLES):
        r = radii_opt[i]
        x = centers_opt[i, 0]
        y = centers_opt[i, 1]
        
        # Cap radius if it exceeds half square size
        if r > 0.5:
            r = 0.5
            radii_opt[i] = r
            x = 0.5
            y = 0.5
            
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        centers_opt[i] = [x, y]
        
    # 3. Resolve overlaps by shrinking radii
    for _ in range(100):
        any_overlap = False
        for i in range(N_CIRCLES):
            for j in range(i + 1, N_CIRCLES):
                dx = centers_opt[i, 0] - centers_opt[j, 0]
                dy = centers_opt[i, 1] - centers_opt[j, 1]
                dist = np.hypot(dx, dy)
                req = radii_opt[i] + radii_opt[j]
                
                if dist < req - 1e-9:
                    overlap = req - dist
                    # Shrink both equally
                    radii_opt[i] -= overlap / 2
                    radii_opt[j] -= overlap / 2
                    if radii_opt[i] < 0: radii_opt[i] = 0
                    if radii_opt[j] < 0: radii_opt[j] = 0
                    any_overlap = True
        if not any_overlap:
            break
            
    # 4. Re-clamp centers
    for i in range(N_CIRCLES):
        r = radii_opt[i]
        x = centers_opt[i, 0]
        y = centers_opt[i, 1]
        centers_opt[i, 0] = np.clip(x, r, 1.0 - r)
        centers_opt[i, 1] = np.clip(y, r, 1.0 - r)

    sum_radii = np.sum(radii_opt)
    
    return centers_opt, radii_opt, float(sum_radii)
