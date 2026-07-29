# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 64b41a5f) state=086cf9c9 sum of radii=2.605119 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def obj_26(x):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(x[52:])

def con_26(x):
    """Constraint function: boundaries and non-overlap (squared distance)."""
    c = x[:52].reshape(N_CIRCLES, 2)
    r = x[52:]
    con = []
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    # Formulated as fun(x) >= 0 for SLSQP
    con.extend(c[:, 0] - r)
    con.extend(1.0 - c[:, 0] - r)
    con.extend(c[:, 1] - r)
    con.extend(1.0 - c[:, 1] - r)
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    # Vectorized computation for all pairs
    diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
    dist2 = np.sum(diff**2, axis=2)
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    violation = dist2 - r_sum**2
    # Only upper triangle needed to avoid duplicate constraints
    con.extend(np.triu(violation, k=1).ravel())
    
    return np.array(con)

def run_packing():
    n = N_CIRCLES
    np.random.seed(42)
    
    # 1. Initial placement
    centers = np.random.rand(n, 2)
    radii = np.ones(n) * 0.02
    
    # 2. Expansion and repulsion simulation
    # This phase quickly finds a dense, feasible configuration
    for step in range(4000):
        radii *= 1.0006
        
        # Enforce boundaries
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)
        
        # Resolve overlaps via pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d2 = dx*dx + dy*dy
                min_d = radii[i] + radii[j]
                
                if d2 < min_d * min_d and d2 > 1e-12:
                    dist = np.sqrt(d2)
                    push = (min_d - dist) * 0.5
                    inv_dist = 1.0 / dist
                    centers[i, 0] += dx * inv_dist * push
                    centers[i, 1] += dy * inv_dist * push
                    centers[j, 0] -= dx * inv_dist * push
                    centers[j, 1] -= dy * inv_dist * push
                    
        # Re-clip after pushes
        centers[:, 0] = np.clip(centers[:, 0], radii, 1.0 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1.0 - radii)

    # 3. Gradient-based optimization
    x0 = np.concatenate([centers.ravel(), radii])
    bnds = [(0.0, 1.0)] * (2 * n) + [(0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': con_26}
    
    try:
        res = minimize(obj_26, x0, method='SLSQP', 
                       constraints=cons, bounds=bnds, 
                       options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
        if res.success:
            centers = res.x[:52].reshape(n, 2)
            radii = res.x[52:]
    except Exception:
        pass
        
    # 4. Post-processing to guarantee strict feasibility
    min_slack = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            slack = d - (radii[i] + radii[j])
            if slack < min_slack:
                min_slack = slack
                
    for i in range(n):
        min_slack = min(min_slack, centers[i, 0] - radii[i])
        min_slack = min(min_slack, 1.0 - centers[i, 0] - radii[i])
        min_slack = min(min_slack, centers[i, 1] - radii[i])
        min_slack = min(min_slack, 1.0 - centers[i, 1] - radii[i])
        
    if min_slack < 0:
        shrink = -min_slack + 1e-7
        radii -= shrink
        
    radii = np.maximum(radii, 0.0)
    for i in range(n):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
    return centers, radii, np.sum(radii)
