# sol_000034 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000006 (state 1103014d) state=a330a8f3 sum of radii=2.618068 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def constraints(vars_vec):
    """
    Inequality constraints g(vars) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c_boundary = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    i, j = np.triu_indices(N_CIRCLES, k=1)
    c_overlap = dist_sq[i, j] - r_sum_sq[i, j]
    
    return np.concatenate([c_boundary, c_overlap])

def generate_hex_init():
    """Generate a base hexagonal lattice for initialization."""
    pts = []
    r_est = 0.09
    dx = 2.0 * r_est
    dy = np.sqrt(3.0) * r_est
    y = r_est
    row = 0
    while len(pts) < N_CIRCLES + 10:
        x = r_est
        if row % 2 == 1:
            x += dx / 2.0
        while x <= 1.0 - r_est and len(pts) < N_CIRCLES + 10:
            pts.append([x, y])
            x += dx
        y += dy
        row += 1
    return np.array(pts[:N_CIRCLES])

def get_safe_radius(pts):
    """Compute a strictly feasible initial radius for given centers."""
    r_min = 1.0
    n = len(pts)
    for i in range(n):
        r_min = min(r_min, pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        for j in range(i+1, n):
            d = np.sqrt(np.sum((pts[i]-pts[j])**2))
            if d/2 < r_min:
                r_min = d/2
    return r_min * 0.6

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    best_sum = -np.inf
    best_vars = None
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    base_pts = generate_hex_init()
    
    for seed in range(12):
        np.random.seed(seed)
        
        # Perturb base points to break symmetry
        pts = base_pts + np.random.uniform(-0.01, 0.01, base_pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        
        r_safe = get_safe_radius(pts)
        
        x0 = np.zeros(3 * N_CIRCLES)
        x0[0::3] = pts[:, 0]
        x0[1::3] = pts[:, 1]
        x0[2::3] = r_safe
        
        try:
            # Stage 1: Coarse optimization
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 600, 'ftol': 1e-7})
            if res.success:
                # Stage 2: Fine optimization
                res2 = minimize(objective, res.x, method='SLSQP', bounds=bounds, constraints=cons,
                                options={'maxiter': 1200, 'ftol': 1e-12})
                if res2.success:
                    curr_sum = -res2.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res2.x
        except Exception:
            continue
            
    if best_vars is None:
        pts = base_pts
        r = get_safe_radius(pts)
        return pts, np.full(N_CIRCLES, r), N_CIRCLES * r
        
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    return centers, radii, float(np.sum(radii))
