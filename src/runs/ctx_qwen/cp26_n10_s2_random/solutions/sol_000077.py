# sol_000077 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000054 (state 91c332d2) state=6c7ef571 sum of radii=2.611171 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def compute_radii(centers):
    """
    Computes the maximum valid radius for each circle given fixed centers.
    r_i = min(dist to boundary, 0.5 * dist to nearest neighbor)
    """
    x, y = centers[:, 0], centers[:, 1]
    r_bound = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(dist, np.inf)
    r_pair = 0.5 * np.min(dist, axis=1)
    
    return np.minimum(r_bound, r_pair)

def obj_centers(v):
    """Objective for centers-only optimization: minimize negative sum of radii."""
    return -np.sum(compute_radii(v.reshape(N, 2)))

def obj_full(v):
    """Objective for full optimization: minimize negative sum of radii."""
    return -np.sum(v[2 * N:])

def cons_full(v):
    """Constraints for full optimization: boundary and non-overlap."""
    c = v[:2 * N].reshape(N, 2)
    r = v[2 * N:]
    con = []
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    con.append(c[:, 0] - r)
    con.append(1.0 - c[:, 0] - r)
    con.append(c[:, 1] - r)
    con.append(1.0 - c[:, 1] - r)
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    idx = np.triu_indices(N, k=1)
    dx = c[idx[0], 0] - c[idx[1], 0]
    dy = c[idx[0], 1] - c[idx[1], 1]
    dr = r[idx[0]] + r[idx[1]]
    con.append(dx**2 + dy**2 - dr**2)
    
    return np.concatenate(con)

def generate_starts():
    """Generates diverse initial configurations."""
    starts = []
    np.random.seed(42)
    
    # Hexagonal lattices with varying densities
    for r0 in [0.09, 0.095, 0.10, 0.105]:
        c = []
        y = r0
        row = 0
        while len(c) < N:
            x = r0 + (row % 2) * r0
            while x <= 1.0 - r0 and len(c) < N:
                c.append([x, y])
                x += 2 * r0
            y += r0 * np.sqrt(3)
            row += 1
        starts.append(np.array(c[:N]))
        
    # Random dense configurations
    for _ in range(20):
        starts.append(np.random.rand(N, 2) * 0.8 + 0.1)
        
    return starts

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    bounds_c = [(1e-5, 1.0 - 1e-5)] * (2 * N)
    starts = generate_starts()
    
    best_centers = None
    best_sum = -1.0
    
    # Phase 1: Multi-start Powell optimization on centers
    for c0 in starts:
        try:
            res = minimize(obj_centers, c0.flatten(), method='Powell', bounds=bounds_c,
                           options={'maxiter': 5000, 'ftol': 1e-12, 'xtol': 1e-12})
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(N, 2)
        except Exception:
            pass
            
    # Phase 2: Iterative perturbation refinement
    if best_centers is not None:
        for _ in range(40):
            pert = best_centers + np.random.normal(0, 0.0015, best_centers.shape)
            pert = np.clip(pert, 1e-5, 1.0 - 1e-5)
            try:
                res = minimize(obj_centers, pert.flatten(), method='Powell', bounds=bounds_c,
                               options={'maxiter': 2000, 'ftol': 1e-13})
                if -res.fun > best_sum:
                    best_sum = -res.fun
                    best_centers = res.x.reshape(N, 2)
            except Exception:
                pass

    # Phase 3: SLSQP polish on centers + radii
    bounds_full = [(0.0, 1.0)] * (2 * N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': cons_full}
    
    c_polish = best_centers.copy()
    r_polish = compute_radii(c_polish)
    v0 = np.concatenate([c_polish.flatten(), r_polish])
    
    try:
        res = minimize(obj_full, v0, method='SLSQP', bounds=bounds_full,
                       constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13})
        if np.min(cons_full(res.x)) > -1e-7:
            best_centers = res.x[:2 * N].reshape(N, 2)
            radii = res.x[2 * N:]
        else:
            radii = compute_radii(best_centers)
    except Exception:
        radii = compute_radii(best_centers)
        
    # Phase 4: Strict validation repair (handles floating point drift)
    for _ in range(20):
        valid = True
        for i in range(N):
            x, y, r = best_centers[i, 0], best_centers[i, 1], radii[i]
            if x - r < -1e-12 or x + r > 1.0 + 1e-12 or y - r < -1e-12 or y + r > 1.0 + 1e-12:
                valid = False
                break
        if not valid:
            radii *= 0.9995
            continue
            
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(best_centers[i, 0] - best_centers[j, 0], best_centers[i, 1] - best_centers[j, 1])
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
                    break
            if not valid:
                break
        if valid:
            break
        radii *= 0.9995
        
    return best_centers, radii, float(np.sum(radii))
