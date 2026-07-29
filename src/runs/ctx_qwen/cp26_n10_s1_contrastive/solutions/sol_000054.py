# sol_000054 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000043 (state 8d6d3048) state=94cc489d sum of radii=2.630369 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraints(x):
    """
    Computes inequality constraints g(x) >= 0.
    Includes boundary containment and pairwise non-overlap.
    """
    n = len(x) // 3
    xc = x[0::3]
    yc = x[1::3]
    r = x[2::3]
    
    c = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c.append(xc - r)
    c.append(1.0 - xc - r)
    c.append(yc - r)
    c.append(1.0 - yc - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    d2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    
    # Only upper triangular pairs (i < j) to avoid duplicates
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    c.append(d2[mask] - rs[mask]**2)
    
    return np.concatenate(c)

def make_init(n, seed, style):
    """Generates a strictly feasible initial configuration with rotation to break symmetry."""
    np.random.seed(seed)
    pts = np.zeros((n, 2))
    
    if style == 0:
        # Hexagonal lattice
        r_e = 0.1
        y = r_e
        row = 0
        idx = 0
        while y < 1.0 and idx < n:
            x_start = r_e if row % 2 == 0 else 2.0 * r_e
            x = x_start
            while x < 1.0 and idx < n:
                pts[idx] = [x, y]
                idx += 1
                x += 2.0 * r_e
            y += np.sqrt(3.0) * r_e
            row += 1
    else:
        # Grid layout
        idx = 0
        for i in range(5):
            for j in range(5):
                pts[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        if n > 25:
            pts[25] = [0.5, 0.5]
            
    # Random perturbation
    pts += np.random.uniform(-0.025, 0.025, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    
    # Random rotation to break geometric symmetry
    angle = np.random.uniform(-0.25, 0.25)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    pts = pts @ rot.T
    pts = np.clip(pts, 0.05, 0.95)
    
    # Compute safe initial radii based on geometry
    min_d = 1.0
    for i in range(n):
        db = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        min_d = min(min_d, db)
        for j in range(i+1, n):
            dp = np.linalg.norm(pts[i] - pts[j])
            min_d = min(min_d, dp)
            
    r0 = min_d * 0.35
    x0 = np.zeros(3 * n)
    x0[0::3] = pts[:, 0]
    x0[1::3] = pts[:, 1]
    x0[2::3] = r0
    return x0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_x = None
    best_sum = -np.inf
    
    # Phase 1: Broad search from structured initializations
    for s in range(40):
        x0 = make_init(n, s, s % 2)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
            # Verify constraints are satisfied within numerical tolerance
            if np.min(constraints(res.x)) >= -1e-7:
                s_val = np.sum(res.x[2::3])
                if s_val > best_sum:
                    best_sum = s_val
                    best_x = res.x.copy()
        except Exception:
            continue

    # Phase 2: Local refinement via perturbation
    if best_x is not None:
        for k in range(25):
            x0 = best_x + np.random.uniform(-0.004, 0.004, 3 * n)
            x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
            x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
            x0[2::3] = np.clip(x0[2::3], 1e-5, 0.49)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12})
                if np.min(constraints(res.x)) >= -1e-7:
                    s_val = np.sum(res.x[2::3])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = res.x.copy()
            except Exception:
                continue
                
        # Phase 3: Variable permutation to break solver stagnation on symmetric configurations
        for k in range(10):
            perm = np.random.permutation(n)
            x0 = np.zeros(3 * n)
            for i in range(n):
                x0[3*i : 3*i+3] = best_x[3*perm[i] : 3*perm[i]+3]
                
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12})
                if np.min(constraints(res.x)) >= -1e-7:
                    s_val = np.sum(res.x[2::3])
                    if s_val > best_sum:
                        best_sum = s_val
                        best_x = res.x.copy()
            except Exception:
                continue

    # Fallback valid configuration
    if best_x is None:
        centers = np.zeros((n, 2))
        idx = 0
        for i in range(5):
            for j in range(5):
                centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        centers[25] = [0.5, 0.5]
        radii = np.full(n, 0.09)
        return centers, radii, float(np.sum(radii))
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    return centers, radii, float(np.sum(radii))
