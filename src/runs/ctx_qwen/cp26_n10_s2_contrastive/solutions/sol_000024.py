# sol_000024 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000020 (state c2ddf6ac) state=7d29769f sum of radii=2.628407 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def objective_func(x):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Returns array of inequality constraints g(x) >= 0."""
    c = []
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(N):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        c.append(xi - ri)
        c.append(1.0 - xi - ri)
        c.append(yi - ri)
        c.append(1.0 - yi - ri)
        
    # Non-overlap constraints: dist(i,j) >= ri + rj
    for i in range(N):
        xi, yi, ri = x[3*i], x[3*i+1], x[3*i+2]
        for j in range(i + 1, N):
            xj, yj, rj = x[3*j], x[3*j+1], x[3*j+2]
            dist = np.hypot(xi - xj, yi - yj)
            c.append(dist - ri - rj)
            
    return np.array(c)

def get_optimal_radii_lp(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    
    # Pairwise constraints: r_i + r_j <= dist_ij
    n_pairs = n * (n - 1) // 2
    A_ub = np.zeros((n_pairs, n))
    b_ub = np.zeros(n_pairs)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = d
            idx += 1
            
    # Variable bounds: 0 <= r_i <= min(x, 1-x, y, 1-y)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        bounds.append((0.0, max(0.0, max_r)))
        
    # Try HiGHS first (fast and reliable), fallback to interior-point
    for method in ['highs', 'interior-point', 'revised simplex']:
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method=method)
            if res.success and np.all(res.x >= -1e-9):
                return res.x
        except Exception:
            continue
            
    # Fallback: return small safe radii if LP fails
    return np.full(n, 0.01)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n = N
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # SLSQP bounds and constraints
    bounds_opt = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons_opt = {'type': 'ineq', 'fun': constraint_func}
    
    # Generate diverse initial configurations
    inits = []
    # Hexagonal row patterns summing to 26
    hex_patterns = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [7, 6, 5, 4, 4],
        [4, 5, 6, 5, 6],
        [5, 5, 5, 5, 6],
        [8, 6, 5, 4, 3]
    ]
    
    for pattern in hex_patterns:
        pts = []
        y = 0.06
        dy = 0.155  # Approx hex row spacing
        for r_idx, cnt in enumerate(pattern):
            shift = 0.0 if r_idx % 2 == 0 else 0.09
            x = 0.06 + shift
            for k in range(cnt):
                if len(pts) < n:
                    pts.append([x, y])
                x += 0.17
            y += dy
        # Pad if needed
        while len(pts) < n:
            pts.append([0.5, 0.5])
        inits.append(np.array(pts[:n]))
        
    # Add random initializations
    for seed in range(5):
        rng = np.random.default_rng(seed * 42 + 7)
        inits.append(rng.uniform(0.1, 0.9, (n, 2)))

    trial = 0
    max_trials = 50
    
    for base in inits:
        for nl in [0.0, 0.02, 0.05]:
            if trial >= max_trials:
                break
                
            rng = np.random.default_rng(trial * 31 + 13)
            c_init = base.copy()
            c_init += rng.normal(0, nl, c_init.shape)
            c_init = np.clip(c_init, 0.02, 0.98)
            
            r_init = np.full(n, 0.03)
            x0 = np.zeros(3 * n)
            for i in range(n):
                x0[3*i] = c_init[i, 0]
                x0[3*i+1] = c_init[i, 1]
                x0[3*i+2] = r_init[i]
                
            try:
                res = minimize(
                    objective_func, x0,
                    method='SLSQP',
                    bounds=bounds_opt,
                    constraints=cons_opt,
                    options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False}
                )
                
                if res.success:
                    curr_c = np.column_stack((res.x[0::3], res.x[1::3]))
                    # LP polishing guarantees optimal radii for these centers
                    curr_r = get_optimal_radii_lp(curr_c)
                    curr_s = np.sum(curr_r)
                    
                    if curr_s > best_sum:
                        best_sum = curr_s
                        best_centers = curr_c.copy()
                        best_radii = curr_r.copy()
            except Exception:
                pass
                
            trial += 1

    # Fallback if all optimizations fail (highly unlikely)
    if best_centers is None:
        best_centers = np.random.rand(n, 2) * 0.6 + 0.2
        best_radii = np.full(n, 0.02)
        best_sum = np.sum(best_radii)

    # Final validation and numerical safety
    c_final = best_centers.copy()
    r_final = best_radii.copy()
    
    # Ensure radii respect boundaries strictly
    for i in range(n):
        x, y = c_final[i]
        r_final[i] = min(r_final[i], x, 1.0 - x, y, 1.0 - y, 0.5)
        r_final[i] = max(0.0, r_final[i])
        
    # Iteratively resolve any microscopic overlaps from numerical drift
    for _ in range(10):
        changed = False
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(c_final[i, 0] - c_final[j, 0], c_final[i, 1] - c_final[j, 1])
                if d < r_final[i] + r_final[j] - 1e-12:
                    exc = r_final[i] + r_final[j] - d
                    r_final[i] -= exc * 0.5
                    r_final[j] -= exc * 0.5
                    r_final[i] = max(0.0, r_final[i])
                    r_final[j] = max(0.0, r_final[j])
                    changed = True
        if not changed:
            break
            
    return c_final, r_final, float(np.sum(r_final))
