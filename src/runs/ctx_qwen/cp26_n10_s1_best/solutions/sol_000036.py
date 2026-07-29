# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000018 (state cd1c4815) state=025191a3 sum of radii=2.631730 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective(vars):
    """Objective to minimize: negative sum of radii."""
    return -np.sum(vars[2::3])

def constraint_func(vars):
    """Vectorized inequality constraints: g(vars) >= 0."""
    n = N_CIRCLES
    cs = vars.reshape(n, 3)
    x, y, r = cs[:, 0], cs[:, 1], cs[:, 2]
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    bcons = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r
    ])
    
    # Overlap constraints: dist^2 - (r1+r2)^2 >= 0
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    # Extract strictly lower triangle indices (avoids self and duplicates)
    i, j = np.tril_indices(n, -1)
    ocons = dx[i, j]**2 + dy[i, j]**2 - dr[i, j]**2
    
    return np.concatenate([bcons, ocons])

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    bounds = []
    for _ in range(N_CIRCLES):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return bounds

def is_valid(centers, radii):
    """Check strict feasibility of a packing configuration."""
    n = centers.shape[0]
    for i in range(n):
        if radii[i] < 0 or centers[i, 0] < radii[i] - 1e-10 or centers[i, 0] > 1 - radii[i] + 1e-10 or \
           centers[i, 1] < radii[i] - 1e-10 or centers[i, 1] > 1 - radii[i] + 1e-10:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
            if dist < radii[i] + radii[j] - 1e-10:
                return False
    return True

def force_directed_init(n, seed):
    """Generate initial configuration using repulsive force simulation."""
    np.random.seed(seed)
    centers = np.random.uniform(0.15, 0.85, (n, 2))
    
    for _ in range(2000):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = np.hypot(dx, dy)
                if dist < 0.25 and dist > 1e-6:
                    f = 0.005 / (dist**2)
                    fx, fy = f * dx, f * dy
                    forces[i] -= [fx, fy]
                    forces[j] += [fx, fy]
        for i in range(n):
            if centers[i, 0] < 0.1: forces[i, 0] += 0.01
            elif centers[i, 0] > 0.9: forces[i, 0] -= 0.01
            if centers[i, 1] < 0.1: forces[i, 1] += 0.01
            elif centers[i, 1] > 0.9: forces[i, 1] -= 0.01
            
        centers += forces * 0.05
        centers = np.clip(centers, 0.05, 0.95)
        
    # Estimate safe initial radii based on nearest neighbors/walls
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
        for j in range(n):
            if i != j:
                d = np.hypot(centers[i, 0] - centers[j, 0], 
                             centers[i, 1] - centers[j, 1])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0 * 0.9
    return centers, radii

def hex_grid_init(n):
    """Generate a dense hexagonal lattice initialization."""
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)
    idx = 0
    r = 0.09
    y = r
    row = 0
    while idx < n:
        x = r if row % 2 == 0 else 2 * r
        while idx < n:
            if x + r > 1.0:
                break
            centers[idx, 0] = x
            centers[idx, 1] = y
            idx += 1
            x += 2 * r
        y += np.sqrt(3) * r
        row += 1
        if y + r > 1.0:
            break
    while idx < n:
        centers[idx] = np.random.uniform(0.1, 0.9, 2)
        radii[idx] = 0.05
        idx += 1
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """Pack 26 circles in a unit square to maximize the sum of radii."""
    n = N_CIRCLES
    bounds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Prepare diverse initial configurations
    inits = [hex_grid_init(n)]
    for s in range(6):
        inits.append(force_directed_init(n, seed=s))
        
    for c_init, r_init in inits:
        x0 = np.zeros(3 * n)
        x0[0::3] = c_init[:, 0]
        x0[1::3] = c_init[:, 1]
        x0[2::3] = r_init
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-11})
            
            if not np.isnan(res.fun):
                current_sum = -res.fun
                if current_sum > best_sum:
                    centers = np.zeros((n, 2))
                    radii = np.zeros(n)
                    for i in range(n):
                        centers[i, 0] = res.x[3 * i]
                        centers[i, 1] = res.x[3 * i + 1]
                        radii[i] = res.x[3 * i + 2]
                        
                    if is_valid(centers, radii):
                        best_sum = current_sum
                        best_centers = centers.copy()
                        best_radii = radii.copy()
        except Exception:
            continue

    # Refinement phase: perturb best solution and re-optimize to escape local minima
    if best_centers is not None:
        for _ in range(3):
            perturbed_c = best_centers + np.random.normal(0, 0.0015, best_centers.shape)
            perturbed_c = np.clip(perturbed_c, 0.01, 0.99)
            perturbed_r = best_radii * 0.985
            x0 = np.zeros(3 * n)
            x0[0::3] = perturbed_c[:, 0]
            x0[1::3] = perturbed_c[:, 1]
            x0[2::3] = perturbed_r
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 2000, 'ftol': 1e-11})
                if not np.isnan(res.fun):
                    current_sum = -res.fun
                    if current_sum > best_sum:
                        centers = np.zeros((n, 2))
                        radii = np.zeros(n)
                        for i in range(n):
                            centers[i, 0] = res.x[3 * i]
                            centers[i, 1] = res.x[3 * i + 1]
                            radii[i] = res.x[3 * i + 2]
                        if is_valid(centers, radii):
                            best_sum = current_sum
                            best_centers = centers.copy()
                            best_radii = radii.copy()
            except Exception:
                pass

    # Fallback (should not be reached with valid inits)
    if best_centers is None:
        best_centers, best_radii = hex_grid_init(n)
        best_sum = np.sum(best_radii)
        
    return best_centers, best_radii, float(best_sum)
