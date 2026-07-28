# sol_000081 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000032 (state ac51bd1a) state=6da8454c sum of radii=2.617832 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars_arr):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_arr[2::3])

def get_constraints(vars_arr, n):
    """Compute all inequality constraints >= 0 for valid packing."""
    xs = vars_arr[0::3]
    ys = vars_arr[1::3]
    rs = vars_arr[2::3]
    
    con = []
    # Boundary constraints: circles must stay inside [0,1]x[0,1]
    con.extend(xs - rs)
    con.extend(1.0 - xs - rs)
    con.extend(ys - rs)
    con.extend(1.0 - ys - rs)
    
    # Pairwise non-overlap constraints: squared distance >= squared sum of radii
    xs_m = xs[:, None] - xs[None, :]
    ys_m = ys[:, None] - ys[None, :]
    rs_m = rs[:, None] + rs[None, :]
    
    dist_sq = xs_m**2 + ys_m**2
    r_sum_sq = rs_m**2
    
    # Only need to check each pair once
    idx = np.triu_indices(n, k=1)
    con.extend(dist_sq[idx] - r_sum_sq[idx])
    
    return np.array(con)

def generate_hex_init(row_counts, r0=0.095, n=26):
    """Generate initial positions on a hexagonal lattice with specified row counts."""
    pts = []
    y = r0
    for idx, cnt in enumerate(row_counts):
        shift = r0 if idx % 2 == 1 else 0.0
        row_width = (cnt - 1) * 2 * r0
        x_start = 0.5 - row_width / 2.0 + shift
        for c in range(cnt):
            x = x_start + c * 2 * r0
            pts.append([x, y])
        y += np.sqrt(3) * r0
    return np.array(pts[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Diverse row configurations that sum to >= 26, tailored for hexagonal packing
    row_configs = [
        [6, 5, 5, 5, 5], [5, 6, 5, 5, 5], [5, 5, 6, 5, 5],
        [5, 5, 5, 6, 5], [5, 5, 5, 5, 6], [6, 6, 5, 5, 4],
        [5, 5, 5, 5, 5, 1], [4, 6, 6, 6, 4], [5, 6, 6, 5, 4],
        [5, 5, 5, 5, 5, 5]
    ]
    
    inits = []
    rng = np.random.default_rng(42)
    
    for rc in row_configs:
        init_pts = generate_hex_init(rc, r0=0.095, n=n)
        inits.append(init_pts)
        
    # Add controlled perturbations to escape symmetry traps and local minima
    for _ in range(5):
        base = inits[0].copy()
        base += rng.uniform(-0.02, 0.02, base.shape)
        base = np.clip(base, 0.05, 0.95)
        inits.append(base)
        
    # Optimization loop over all initial configurations
    for cfg in inits:
        v0 = np.zeros(3 * n)
        v0[0::3] = cfg[:, 0]
        v0[1::3] = cfg[:, 1]
        v0[2::3] = 0.09  # Feasible initial radius guess
        
        try:
            res = minimize(
                objective, 
                v0, 
                method='SLSQP', 
                bounds=bounds,
                constraints={'type': 'ineq', 'fun': get_constraints, 'args': (n,)},
                options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False}
            )
            
            if np.isfinite(res.fun):
                cx = res.x[0::3]
                cy = res.x[1::3]
                r = res.x[2::3]
                
                # Strict validation matching the grader's tolerance
                valid = True
                for i in range(n):
                    if cx[i] < r[i] - 1e-12 or cx[i] > 1 - r[i] + 1e-12 or \
                       cy[i] < r[i] - 1e-12 or cy[i] > 1 - r[i] + 1e-12:
                        valid = False
                        break
                if valid:
                    for i in range(n):
                        for j in range(i + 1, n):
                            d2 = (cx[i] - cx[j])**2 + (cy[i] - cy[j])**2
                            r_sum = r[i] + r[j]
                            if d2 < r_sum**2 - 1e-12:
                                valid = False
                                break
                        if not valid:
                            break
                            
                if valid:
                    s = np.sum(r)
                    if s > best_sum:
                        best_sum = s
                        best_centers = np.column_stack((cx, cy))
                        best_radii = r.copy()
        except Exception:
            continue
            
    # Fallback configuration if optimization unexpectedly fails
    if best_centers is None:
        best_centers = inits[0]
        best_radii = np.full(n, 0.08)
        best_sum = np.sum(best_radii)
        
    # Safety scaling to guarantee strict validity within numerical tolerance
    scale = 1.0
    for i in range(n):
        x, y, r = best_centers[i, 0], best_centers[i, 1], best_radii[i]
        if r > 1e-12:
            scale = min(scale, x/r, (1-x)/r, y/r, (1-y)/r)
            
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(best_centers[i] - best_centers[j])
            rs = best_radii[i] + best_radii[j]
            if rs > 1e-12:
                scale = min(scale, d/rs)
                
    best_radii *= scale * 0.999999
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, float(best_sum)
