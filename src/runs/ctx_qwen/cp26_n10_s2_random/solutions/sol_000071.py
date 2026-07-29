# sol_000071 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000056 (state 0fa800b4) state=95b12ad0 sum of radii=2.624544 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(vars_flat):
    """Computes boundary and non-overlap constraints. Must return >= 0."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]

    # Boundary constraints
    c1 = xs - rs
    c2 = 1.0 - xs - rs
    c3 = ys - rs
    c4 = 1.0 - ys - rs

    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    idx = np.triu_indices(N, k=1)
    dx = xs[idx[0]] - xs[idx[1]]
    dy = ys[idx[0]] - ys[idx[1]]
    dr = rs[idx[0]] + rs[idx[1]]
    c5 = dx**2 + dy**2 - dr**2

    return np.concatenate([c1, c2, c3, c4, c5])

def compute_objective(vars_flat):
    """Objective: maximize sum of radii -> minimize negative sum."""
    return -np.sum(vars_flat[2::3])

def get_bounds():
    """Variable bounds for x, y in [0,1] and r in [1e-7, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def get_safe_radii(centers):
    """Computes strictly feasible initial radii for given centers."""
    n = centers.shape[0]
    # Distance to boundaries
    rb = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                    np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    # Distance to nearest neighbor
    dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    rp = 0.5 * np.min(dists, axis=1)
    # Scale down slightly to ensure strict initial feasibility
    return np.minimum(rb, rp) * 0.94

def hex_init(n, r, rows):
    """Generates hexagonal lattice positions based on row counts."""
    pos = []
    y = r
    for r_idx, count in enumerate(rows):
        x_start = r + (r_idx % 2) * r
        for c in range(count):
            x = x_start + c * (2.0 * r)
            pos.append([x, y])
        y += r * np.sqrt(3.0)
    return np.array(pos[:n])

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_vars = None
    best_obj = -np.inf
    
    candidates = []
    np.random.seed(42)
    
    # 1. Hexagonal lattice patterns with varying densities
    patterns = [
        [6, 5, 6, 5, 4], [5, 6, 5, 6, 4], [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4], [4, 6, 6, 6, 4]
    ]
    for r_init in [0.092, 0.098, 0.105, 0.112]:
        for pat in patterns:
            c = hex_init(N, r_init, pat)
            c += np.random.normal(0, 0.004, c.shape)
            c = np.clip(c, 0.05, 0.95)
            r_safe = get_safe_radii(c)
            v = np.zeros(N * 3)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            v[2::3] = r_safe
            candidates.append(v)

    # 2. Perturbed grid with center circle
    for r_init in [0.095, 0.100]:
        grid_c = []
        for i in range(5):
            for j in range(5):
                grid_c.append([0.1 + i * 0.2, 0.1 + j * 0.2])
        grid_c.append([0.4, 0.4])  # 26th circle in gap
        grid_c = np.array(grid_c[:N])
        grid_c += np.random.normal(0, 0.005, grid_c.shape)
        grid_c = np.clip(grid_c, 0.05, 0.95)
        r_safe = get_safe_radii(grid_c)
        v = np.zeros(N * 3)
        v[0::3] = grid_c[:, 0]
        v[1::3] = grid_c[:, 1]
        v[2::3] = r_safe
        candidates.append(v)

    # 3. Diverse random starts
    for seed in range(8):
        rng = np.random.default_rng(seed)
        c_rand = rng.uniform(0.15, 0.85, (N, 2))
        r_safe = get_safe_radii(c_rand)
        v = np.zeros(N * 3)
        v[0::3] = c_rand[:, 0]
        v[1::3] = c_rand[:, 1]
        v[2::3] = r_safe
        candidates.append(v)
        
    # Primary optimization phase
    for x0 in candidates:
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
            
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-8):
                curr_obj = -res.fun
                if curr_obj > best_obj:
                    best_obj = curr_obj
                    best_vars = res.x.copy()
        except Exception:
            continue
        
    if best_vars is None:
        best_vars = candidates[0]
    
    # Iterative refinement to escape local minima
    rng_ref = np.random.default_rng(123)
    for _ in range(25):
        x_curr = best_vars.copy()
        # Perturb centers, keep radii stable to maintain feasibility
        x_curr[:2 * N] += rng_ref.normal(0, 0.0008, 2 * N)
        x_curr = np.clip(x_curr, 0.0, 1.0)
        x_curr[2::3] = np.clip(x_curr[2::3], 1e-7, 0.5)
        
        try:
            res = minimize(compute_objective, x_curr, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
            
            c_vals = compute_constraints(res.x)
            if np.all(c_vals >= -1e-8) and -res.fun > best_obj:
                best_obj = -res.fun
                best_vars = res.x.copy()
        except Exception:
            continue
        
    # Final strict repair to guarantee validator tolerance
    for _ in range(30):
        c_check = compute_constraints(best_vars)
        if np.min(c_check) >= -1e-9:
            break
        best_vars[2::3] *= 0.9997
        
    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    radii = np.maximum(radii, 0.0)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
