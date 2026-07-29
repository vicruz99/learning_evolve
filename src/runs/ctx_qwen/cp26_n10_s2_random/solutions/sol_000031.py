# sol_000031 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000017 (state 58c90071) state=1c066cf6 sum of radii=2.617835 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
# Precompute indices for pairwise constraints to avoid repeated allocation
IND_I, IND_J = np.triu_indices(N, k=1)

def get_bounds():
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints for the packing."""
    xs = vars_flat[0::3]
    ys = vars_flat[1::3]
    rs = vars_flat[2::3]
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c_bound = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints: dist(i,j) >= r_i + r_j
    dx = xs[IND_I] - xs[IND_J]
    dy = ys[IND_I] - ys[IND_J]
    dists = np.sqrt(dx**2 + dy**2)
    rs_sum = rs[IND_I] + rs[IND_J]
    c_overlap = dists - rs_sum
    
    return np.concatenate([c_bound, c_overlap])

def obj_func(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def generate_hex_init(seed):
    """Generates a feasible initial configuration based on a hexagonal lattice."""
    rng = np.random.default_rng(seed)
    pts = []
    r_est = 0.1
    dy = r_est * np.sqrt(3)
    
    for row in range(10):
        offset = (row % 2) * r_est
        for col in range(10):
            x = col * 2 * r_est + offset + r_est
            y = row * dy + r_est
            if 0 <= x <= 1 and 0 <= y <= 1:
                pts.append([x, y])
        if len(pts) >= N:
            break
            
    pts = np.array(pts[:N])
    # Normalize and scale to fit safely inside [0,1]
    pts = (pts - pts.min(axis=0)) / (pts.max(axis=0) - pts.min(axis=0) + 1e-9)
    pts = pts * 0.8 + 0.1
    pts += rng.normal(0, 0.005, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    
    v = np.zeros(N * 3)
    v[0::3] = pts[:, 0]
    v[1::3] = pts[:, 1]
    v[2::3] = 0.08
    return v

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_val = -np.inf
    
    # Multiple restarts from diverse initial configurations
    seeds = [0, 1, 2, 3, 4, 10, 20, 30, 40, 50]
    for seed in seeds:
        x0 = generate_hex_init(seed)
        for _ in range(3):
            try:
                res = minimize(obj_func, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-7):
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_vars = res.x.copy()
            except Exception:
                pass
            
            # Perturb best found solution to explore neighborhood
            if best_vars is not None:
                x0 = best_vars + np.random.default_rng(seed + 100).normal(0, 0.002, N*3)
                x0[2::3] = best_vars[2::3] # Keep radii stable during position perturbation
                
    # Fallback initialization if optimization fails
    if best_vars is None:
        best_vars = generate_hex_init(0)
        
    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    
    # Iterative Growth Phase: progressively scale radii up and resolve conflicts
    for _ in range(15):
        radii *= 1.003
        cur_vars = np.zeros(N * 3)
        cur_vars[0::3] = centers[:, 0]
        cur_vars[1::3] = centers[:, 1]
        cur_vars[2::3] = radii
        try:
            res = minimize(obj_func, cur_vars, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
            if np.all(compute_constraints(res.x) >= -1e-7):
                centers = res.x.reshape(N, 3)[:, :2]
                radii = res.x.reshape(N, 3)[:, 2]
            else:
                # If growth causes infeasibility that optimizer can't fix, revert and stop growing
                radii /= 1.003
                break
        except Exception:
            break
            
    # Final safety check: ensure strict validity within validator tolerance
    best_vars = np.zeros(N * 3)
    best_vars[0::3] = centers[:, 0]
    best_vars[1::3] = centers[:, 1]
    best_vars[2::3] = radii
    
    for _ in range(50):
        c_vals = compute_constraints(best_vars)
        if np.all(c_vals >= -1e-9):
            break
        # Shrink radii slightly to resolve any remaining numerical violations
        best_vars[2::3] *= 0.99995
        
    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    radii = np.maximum(radii, 0.0)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
