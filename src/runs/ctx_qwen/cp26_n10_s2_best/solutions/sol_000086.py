# sol_000086 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000049 (state 0aad4082) state=e307a773 sum of radii=2.631094 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(vars):
    """Objective: minimize negative sum of radii."""
    return -np.sum(vars[2*N:])

def constraints(vars, pair_i, pair_j):
    """Inequality constraints: boundaries and non-overlap (squared distances)."""
    x = vars[:N]
    y = vars[N:2*N]
    r = vars[2*N:]
    
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    # Pairwise constraints: dist^2 >= (r_i + r_j)^2
    cons = np.concatenate([
        x - r,
        1.0 - x - r,
        y - r,
        1.0 - y - r,
        (x[pair_i] - x[pair_j])**2 + (y[pair_i] - y[pair_j])**2 - (r[pair_i] + r[pair_j])**2
    ])
    return cons

def ensure_feasible(vars, pair_i, pair_j):
    """Adjusts radii to guarantee the configuration satisfies all constraints strictly."""
    x = vars[:N].copy()
    y = vars[N:2*N].copy()
    r = vars[2*N:].copy()
    tol = 1e-7
    
    # Enforce boundary constraints
    r = np.minimum(r, np.minimum(x, 1.0 - x))
    r = np.minimum(r, np.minimum(y, 1.0 - y))
    
    # Enforce non-overlap constraints iteratively
    for _ in range(5):
        changed = False
        for idx in range(len(pair_i)):
            i, j = pair_i[idx], pair_j[idx]
            d = np.hypot(x[i] - x[j], y[i] - y[j])
            if d < r[i] + r[j] - tol:
                shrink = (r[i] + r[j] - d) * 0.5 + tol
                r[i] -= shrink
                r[j] -= shrink
                changed = True
        r = np.maximum(r, 0.0)
        if not changed:
            break
            
    return np.concatenate([x, y, r])

def generate_hex_centers(r_est, shift=0.0):
    """Generates a hexagonal lattice configuration."""
    centers = []
    y = r_est + shift
    row = 0
    while len(centers) < N:
        x_start = r_est if row % 2 == 0 else 2 * r_est
        x = x_start
        while x <= 1.0 - r_est and len(centers) < N:
            centers.append([x, y])
            x += 2 * r_est
        y += np.sqrt(3) * r_est
        row += 1
    return np.array(centers[:N])

def run_packing():
    pair_i, pair_j = np.triu_indices(N, k=1)
    bounds = [(0.0, 1.0)] * (2*N) + [(0.0, 0.5)] * N
    cons_dict = {'type': 'ineq', 'fun': constraints, 'args': (pair_i, pair_j)}
    
    best_sum = -1.0
    best_x = None
    
    # Generate diverse initial configurations
    inits = []
    
    # 1. Hexagonal lattices with varying densities and shifts
    for r_est in [0.085, 0.090, 0.095, 0.100]:
        for shift in [0.0, 0.01, -0.01, 0.02, 0.04]:
            c = generate_hex_centers(r_est, shift)
            c += np.random.uniform(-0.005, 0.005, c.shape)
            c = np.clip(c, 0.02, 0.98)
            inits.append(np.concatenate([c[:, 0], c[:, 1], np.full(N, 0.03)]))
            
    # 2. Grid configurations
    for s in np.linspace(0.0, 0.04, 3):
        pts = []
        for i in range(6):
            for j in range(5):
                pts.append([0.08 + s + i*0.17, 0.08 + s + j*0.20])
        pts = np.array(pts[:N])
        pts += np.random.uniform(-0.005, 0.005, pts.shape)
        pts = np.clip(pts, 0.02, 0.98)
        inits.append(np.concatenate([pts[:, 0], pts[:, 1], np.full(N, 0.03)]))
        
    # 3. Random configurations
    np.random.seed(42)
    for _ in range(5):
        c = np.random.uniform(0.1, 0.9, (N, 2))
        inits.append(np.concatenate([c[:, 0], c[:, 1], np.full(N, 0.02)]))
        
    # Primary optimization pass
    for x0 in inits:
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons_dict,
                           options={'maxiter': 8000, 'ftol': 1e-12, 'disp': False})
            if -res.fun > best_sum:
                if np.all(constraints(res.x, pair_i, pair_j) >= -1e-7):
                    best_sum = -res.fun
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Local search refinement: perturb best solution to escape shallow local minima
    if best_x is not None:
        current_x = best_x.copy()
        for step in range(20):
            pert = current_x.copy()
            # Perturb centers slightly
            pert[:2*N] += np.random.uniform(-0.004, 0.004, 2*N)
            pert[:2*N] = np.clip(pert[:2*N], 0.01, 0.99)
            # Shrink radii to ensure feasibility after perturbation
            pert[2*N:] *= 0.94
            # Guarantee strict feasibility before optimization
            pert = ensure_feasible(pert, pair_i, pair_j)
            
            try:
                res = minimize(objective, pert, method='SLSQP', bounds=bounds,
                               constraints=cons_dict,
                               options={'maxiter': 6000, 'ftol': 1e-12, 'disp': False})
                if -res.fun > best_sum:
                    if np.all(constraints(res.x, pair_i, pair_j) >= -1e-7):
                        best_sum = -res.fun
                        best_x = res.x.copy()
                        current_x = best_x.copy()
            except Exception:
                continue
                
    # Fallback in case optimization fails completely
    if best_x is None:
        best_x = inits[0]
        
    centers = np.column_stack((best_x[:N], best_x[N:2*N]))
    radii = best_x[2*N:].copy()
    
    # Strict post-processing to guarantee validation passes
    for i in range(N):
        radii[i] = min(radii[i], centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        
    for _ in range(10):
        for i in range(N):
            for j in range(i + 1, N):
                d = np.hypot(centers[i, 0] - centers[j, 0], centers[i, 1] - centers[j, 1])
                if d < radii[i] + radii[j] - 1e-9:
                    shrink = (radii[i] + radii[j] - d) / 2.0 + 1e-8
                    radii[i] = max(0.0, radii[i] - shrink)
                    radii[j] = max(0.0, radii[j] - shrink)
                    
    return centers, radii, float(np.sum(radii))
