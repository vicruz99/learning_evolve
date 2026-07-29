# sol_000059 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000030 (state 57c93ce5) state=d09a092a sum of radii=2.626678 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
I_IDX, J_IDX = np.tril_indices(N_CIRCLES, -1)

def objective(vars):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars[2::3])

def constraints(vars):
    """Returns all inequality constraints g(vars) >= 0."""
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints
    w1 = x - r
    w2 = 1.0 - x - r
    w3 = y - r
    w4 = 1.0 - y - r
    
    # Overlap constraints: dist^2 >= (r_i + r_j)^2
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    o = dx[I_IDX, J_IDX]**2 + dy[I_IDX, J_IDX]**2 - dr[I_IDX, J_IDX]**2
    
    return np.concatenate([w1, w2, w3, w4, o])

def resolve_overlaps(vars, n, iters=500):
    """Iteratively pushes circles apart and clips to boundaries until feasible."""
    x = vars[0::3].copy()
    y = vars[1::3].copy()
    r = vars[2::3].copy()
    
    for it in range(iters):
        # Clip to walls
        x = np.clip(x, r, 1.0 - r)
        y = np.clip(y, r, 1.0 - r)
        
        max_v = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dist = np.hypot(dx, dy)
                min_d = r[i] + r[j]
                
                if dist < min_d and dist > 1e-9:
                    shift = (min_d - dist) * 0.5
                    nx, ny = dx / dist, dy / dist
                    x[i] -= shift * nx
                    y[i] -= shift * ny
                    x[j] += shift * nx
                    y[j] += shift * ny
                    max_v = max(max_v, min_d - dist)
                    
        if max_v < 1e-7:
            break
            
        # If stuck, slightly shrink radii to allow optimization to proceed
        if it % 50 == 49 and max_v > 1e-5:
            r *= 0.99
            vars[2::3] = r
            
    vars[0::3] = x
    vars[1::3] = y
    vars[2::3] = r
    return vars

def create_hex_init(seed, r_start):
    """Generates a perturbed hexagonal lattice initialization."""
    np.random.seed(seed)
    vars = np.zeros(3 * N_CIRCLES)
    count = 0
    y = r_start
    
    while count < N_CIRCLES:
        x = r_start
        row = int(y / r_start)
        shift = r_start if row % 2 == 1 else 0.0
        
        while count < N_CIRCLES:
            if x + shift + r_start > 1.0:
                break
                
            cx = x + shift + np.random.uniform(-0.005, 0.005)
            cy = y + np.random.uniform(-0.005, 0.005)
            cx = np.clip(cx, r_start, 1.0 - r_start)
            cy = np.clip(cy, r_start, 1.0 - r_start)
            
            vars[3*count] = cx
            vars[3*count+1] = cy
            vars[3*count+2] = r_start
            count += 1
            x += 2 * r_start
            
        y += np.sqrt(3) * r_start
        if y + r_start > 1.0:
            break
            
    # Fill remaining slots if any
    while count < N_CIRCLES:
        vars[3*count] = np.random.uniform(0.15, 0.85)
        vars[3*count+1] = np.random.uniform(0.15, 0.85)
        vars[3*count+2] = 0.06
        count += 1
        
    return vars

def run_packing():
    n = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_sum = -1.0
    best_vars = None
    
    # Target radii near the theoretical optimum for N=26 (~0.10192)
    r_starts = [0.098, 0.099, 0.100, 0.101, 0.1015]
    
    # Phase 1: Multi-start optimization from feasible hexagonal grids
    for idx, r_start in enumerate(r_starts):
        for seed in range(5):
            x0 = create_hex_init(idx * 5 + seed, r_start)
            x0 = resolve_overlaps(x0, n, iters=300)
            
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
                s = -res.fun
                if s > best_sum:
                    cvals = constraints(res.x)
                    if np.min(cvals) >= -1e-7:
                        best_sum = s
                        best_vars = res.x.copy()
            except Exception:
                pass
                
    # Phase 2: Local refinement to escape shallow minima
    if best_vars is not None:
        for _ in range(4):
            x0 = best_vars + np.random.normal(0, 1e-4, 3 * n)
            for i in range(n):
                r = max(0.0, x0[3*i+2])
                x0[3*i] = np.clip(x0[3*i], r, 1.0-r)
                x0[3*i+1] = np.clip(x0[3*i+1], r, 1.0-r)
                x0[3*i+2] = r
            try:
                res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False})
                s = -res.fun
                if s > best_sum:
                    cvals = constraints(res.x)
                    if np.min(cvals) >= -1e-7:
                        best_sum = s
                        best_vars = res.x.copy()
            except Exception:
                break

    # Extract results
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_vars[3*i]
        centers[i, 1] = best_vars[3*i+1]
        radii[i] = best_vars[3*i+2]
        
    # Phase 3: Strict validation and minimal repair
    valid = True
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            valid = False
            break
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                if np.hypot(dx, dy) < radii[i] + radii[j] - 1e-9:
                    valid = False
                    break
            if not valid:
                break
                
    if not valid:
        factor = 0.995
        for _ in range(50):
            radii *= factor
            centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
            centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
            flat_vars = np.concatenate([centers[:, 0], centers[:, 1], radii])
            if np.all(constraints(flat_vars) >= -1e-9):
                break
                
    return centers, radii, float(np.sum(radii))
