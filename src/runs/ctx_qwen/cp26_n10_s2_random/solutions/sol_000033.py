# sol_000033 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state 58c90071) state=a450ee13 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_objective(vars_flat):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars_flat[2::3])

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints for the packing."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]
    
    c = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.append(xs - rs)
    c.append(1.0 - xs - rs)
    c.append(ys - rs)
    c.append(1.0 - ys - rs)
    
    # Non-overlap constraints: dist^2 >= (r_i + r_j)^2
    idx_i, idx_j = np.triu_indices(N, k=1)
    dx = xs[idx_i] - xs[idx_j]
    dy = ys[idx_i] - ys[idx_j]
    dr = rs[idx_i] + rs[idx_j]
    c.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(c)

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def generate_hex_init(r_guess, perturb_scale=0.0, seed=0):
    """Generates a feasible hexagonal initial configuration scaled to fit r_guess."""
    rng = np.random.default_rng(seed)
    raw_pts = []
    y_raw = 1.0
    rows = [5, 4, 5, 4, 5, 3]  # 26 circles total
    
    for r_idx, count in enumerate(rows):
        x_start_raw = 1.0 + (r_idx % 2) * 1.0
        for k in range(count):
            raw_pts.append([x_start_raw + k*2.0, y_raw])
        y_raw += np.sqrt(3)
        
    raw_pts = np.array(raw_pts)
    min_pt = raw_pts.min(axis=0)
    max_pt = raw_pts.max(axis=0)
    width_raw = max_pt[0] - min_pt[0]
    height_raw = max_pt[1] - min_pt[1]
    
    avail_w = 1.0 - 2.0 * r_guess
    avail_h = 1.0 - 2.0 * r_guess
    
    # Scale to fit tightly within available space
    scale = min(avail_w / width_raw, avail_h / height_raw)
    pts = (raw_pts - min_pt) * scale + r_guess
    
    if perturb_scale > 0:
        pts += rng.normal(0, perturb_scale, pts.shape)
        pts = np.clip(pts, 1e-4, 1.0 - 1e-4)
        
    init = np.zeros(N * 3)
    for i in range(N):
        init[3*i] = pts[i, 0]
        init[3*i+1] = pts[i, 1]
        init[3*i+2] = r_guess
    return init

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize sum of radii."""
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_obj = -np.inf
    
    # Explore different base radii and perturbation levels to escape local minima
    r_candidates = [0.095, 0.098, 0.100, 0.101]
    perturb_candidates = [0.0, 0.001, 0.005, 0.01]
    
    seed_counter = 0
    for r_guess in r_candidates:
        for p in perturb_candidates:
            x0 = generate_hex_init(r_guess, p, seed=seed_counter)
            seed_counter += 1
            try:
                res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-9):
                    curr_obj = -res.fun
                    if curr_obj > best_obj:
                        best_obj = curr_obj
                        best_vars = res.x.copy()
            except Exception:
                pass

    # Fallback if optimization fails entirely
    if best_vars is None:
        best_vars = generate_hex_init(0.095, 0.0, seed=0)

    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    radii = np.maximum(radii, 0.0)
    
    # Iterative radius growth refinement: push radii up if constraints are slack
    for _ in range(5):
        c_vals = compute_constraints(best_vars)
        min_slack = np.min(c_vals)
        if min_slack > 1e-6:
            radii *= 1.002
            best_vars[2::3] = radii
            res = minimize(compute_objective, best_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
            if np.all(compute_constraints(res.x) >= -1e-9):
                best_vars = res.x.copy()
                radii = best_vars[2::3].copy()
                centers = best_vars.reshape(N, 3)[:, :2]
        else:
            break
            
    radii = best_vars.reshape(N, 3)[:, 2]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
