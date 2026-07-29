# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000036 (state ae916370) state=0fa800b4 sum of radii=2.624513 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_constraints(vars_flat):
    """Computes all boundary and non-overlap constraints for the packing."""
    X = vars_flat.reshape(N, 3)
    xs = X[:, 0]
    ys = X[:, 1]
    rs = X[:, 2]

    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c1 = xs - rs
    c2 = 1.0 - xs - rs
    c3 = ys - rs
    c4 = 1.0 - ys - rs

    # Overlap constraints: dist^2 >= (r_i + r_j)^2
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
    """Returns variable bounds for x, y, r."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)])
    return b

def force_relax_init(n, seed, steps=1200):
    """Generates a densely packed configuration via force-directed relaxation."""
    rng = np.random.default_rng(seed)
    centers = rng.uniform(0.15, 0.85, (n, 2))
    radii = np.full(n, 0.025)
    
    for _ in range(steps):
        radii *= 1.0006
        forces = np.zeros_like(centers)
        
        # Circle-circle repulsion
        for i in range(n):
            for j in range(i + 1, n):
                d = np.hypot(centers[i,0]-centers[j,0], centers[i,1]-centers[j,1])
                req = radii[i] + radii[j]
                if d < req:
                    f = (req - d) * 12.0
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    if d > 1e-9:
                        forces[i,0] += dx/d * f
                        forces[i,1] += dy/d * f
                        forces[j,0] -= dx/d * f
                        forces[j,1] -= dy/d * f
                        
        # Boundary repulsion
        for i in range(n):
            if centers[i,0] < radii[i]: forces[i,0] += 80.0 * (radii[i] - centers[i,0])
            if centers[i,0] > 1.0 - radii[i]: forces[i,0] -= 80.0 * (centers[i,0] - (1.0 - radii[i]))
            if centers[i,1] < radii[i]: forces[i,1] += 80.0 * (radii[i] - centers[i,1])
            if centers[i,1] > 1.0 - radii[i]: forces[i,1] -= 80.0 * (centers[i,1] - (1.0 - radii[i]))
            
        centers += forces * 0.004
        centers = np.clip(centers, 0.005, 0.995)
        
    return centers, radii

def hex_init(n, r_base, rows, seed):
    """Generates a hexagonal lattice initialization with specified row counts."""
    rng = np.random.default_rng(seed)
    pos = []
    y = r_base
    for r_idx, count in enumerate(rows):
        x_start = r_base + (r_idx % 2) * r_base
        for c in range(count):
            x = x_start + c * (2 * r_base)
            pos.append([x, y])
        y += r_base * np.sqrt(3)
    pos = np.array(pos[:n]) + rng.normal(0, 0.003, (n, 2))
    pos = np.clip(pos, 0.05, 0.95)
    return pos, np.full(n, r_base)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': compute_constraints}

    best_vars = None
    best_obj = -np.inf

    candidates = []
    
    # Hexagonal lattice initializations with various densities and row patterns
    row_patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 5, 5, 5],
        [5, 5, 6, 5, 5],
        [6, 6, 5, 5, 4]
    ]
    for r_b in [0.092, 0.098, 0.105, 0.112]:
        for rows in row_patterns:
            for s in range(2):
                try:
                    c, r = hex_init(N, r_b, rows, s)
                    v = np.zeros(N * 3)
                    v[0::3] = c[:, 0]
                    v[1::3] = c[:, 1]
                    v[2::3] = r * 0.96  # Shrink slightly to ensure initial feasibility
                    candidates.append(v)
                except Exception:
                    pass

    # Force relaxation initializations
    for s in range(8):
        try:
            c, r = force_relax_init(N, s)
            v = np.zeros(N * 3)
            v[0::3] = c[:, 0]
            v[1::3] = c[:, 1]
            v[2::3] = r * 0.97
            candidates.append(v)
        except Exception:
            pass

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

    # Refinement phase: perturb best solution and re-optimize
    if best_vars is not None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            x_curr = best_vars.copy()
            # Random perturbation to escape local minima
            x_curr += rng.normal(0, 0.0008, x_curr.shape)
            x_curr = np.clip(x_curr, 0.0, 1.0)
            x_curr[2::3] = np.clip(x_curr[2::3], 1e-7, 0.5)
            
            try:
                res = minimize(compute_objective, x_curr, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                
                c_vals = compute_constraints(res.x)
                if np.all(c_vals >= -1e-8):
                    curr_obj = -res.fun
                    if curr_obj > best_obj:
                        best_obj = curr_obj
                        best_vars = res.x.copy()
            except Exception:
                continue

    # Fallback if optimization fails entirely
    if best_vars is None:
        best_vars = candidates[0]

    # Final safety adjustment to strictly satisfy validator tolerance
    for _ in range(15):
        c_check = compute_constraints(best_vars)
        if np.min(c_check) >= -1e-9:
            break
        # Proportionally shrink radii to resolve constraint violations
        best_vars[2::3] *= 0.9998
        
    centers = best_vars.reshape(N, 3)[:, :2]
    radii = best_vars.reshape(N, 3)[:, 2]
    radii = np.maximum(radii, 0.0)
    sum_r = float(np.sum(radii))
    
    return centers, radii, sum_r
