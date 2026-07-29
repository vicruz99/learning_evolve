# sol_000083 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000025 (state d15e4e7a) state=ab0998a2 sum of radii=2.621143 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints: boundary containment and pairwise separation.
    All constraints are formulated as g(vars_vec) >= 0.
    """
    x = vars_vec[0::3]
    y = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = []
    # Boundary constraints: x >= r, x <= 1-r, y >= r, y <= 1-r
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    # Using precomputed indices for efficiency
    dx = x[I_IDX] - x[J_IDX]
    dy = y[I_IDX] - y[J_IDX]
    dist2 = dx**2 + dy**2
    rs = r[I_IDX] + r[J_IDX]
    c.append(dist2 - rs**2)
    
    return np.concatenate(c)

def compute_safe_radii(centers):
    """Compute strictly feasible initial radii for a given set of centers."""
    n = centers.shape[0]
    r = np.zeros(n)
    for i in range(n):
        # Distance to boundaries
        d_bound = min(centers[i, 0], 1.0 - centers[i, 0], 
                      centers[i, 1], 1.0 - centers[i, 1])
        # Distance to nearest neighbor
        d_min = 1.0
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < d_min:
                    d_min = d
        # Safe radius: fraction of limiting distance
        r[i] = 0.42 * min(d_bound, d_min * 0.5)
    return r

def run_packing():
    rng = np.random.default_rng(42)
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_vars = None
    
    # 1. Generate base layouts
    bases = []
    
    # Hexagonal lattice (rows: 6, 5, 6, 5, 4)
    pts_hex = []
    r_est = 0.095
    y = r_est
    row = 0
    while len(pts_hex) < N:
        x_off = (row % 2) * r_est
        x = r_est + x_off
        while x <= 1.0 - r_est and len(pts_hex) < N:
            pts_hex.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    bases.append(np.array(pts_hex[:N]))
    
    # Square grid 5x5 + center
    pts_grid = []
    for i in range(5):
        for j in range(5):
            pts_grid.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    pts_grid.append([0.5, 0.5])
    bases.append(np.array(pts_grid[:N]))
    
    # Random uniform
    bases.append(rng.uniform(0.15, 0.85, (N, 2)))
    
    # Create perturbed versions for diverse starts
    inits = []
    for base in bases:
        inits.append(base)
        for _ in range(5):
            pert = base + rng.uniform(-0.04, 0.04, (N, 2))
            inits.append(np.clip(pert, 0.05, 0.95))
            
    # 2. Optimization loop over all initializations
    for base_centers in inits:
        r0 = compute_safe_radii(base_centers)
        x0 = np.zeros(3 * N)
        x0[0::3] = base_centers[:, 0]
        x0[1::3] = base_centers[:, 1]
        x0[2::3] = r0
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = res.x.copy()
        except Exception:
            continue
            
    # 3. Local perturbation refinement on the best configuration found
    if best_vars is not None:
        for _ in range(20):
            x_pert = best_vars.copy()
            # Perturb centers slightly
            x_pert[0::3] = np.clip(x_pert[0::3] + rng.uniform(-0.008, 0.008, N), 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3] + rng.uniform(-0.008, 0.008, N), 0.01, 0.99)
            # Slightly perturb radii to break symmetry
            x_pert[2::3] = np.maximum(x_pert[2::3] + rng.uniform(-0.003, 0.003, N), 1e-6)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = constraint_func(res.x)
                    if np.min(c_val) >= -1e-9:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_vars = res.x.copy()
            except Exception:
                continue
                
    # 4. Final high-precision polish
    if best_vars is not None:
        try:
            res_final = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
            if res_final.success:
                c_val = constraint_func(res_final.x)
                if np.min(c_val) >= -1e-9:
                    best_vars = res_final.x
                    best_sum = -res_final.fun
        except Exception:
            pass
            
    # Fallback safety net
    if best_vars is None:
        base_centers = bases[1]
        r0 = compute_safe_radii(base_centers)
        best_vars = np.zeros(3 * N)
        best_vars[0::3] = base_centers[:, 0]
        best_vars[1::3] = base_centers[:, 1]
        best_vars[2::3] = r0
        best_sum = np.sum(r0)
        
    # Extract result
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    radii = np.maximum(radii, 0.0)
    final_sum = float(np.sum(radii))
    
    return centers, radii, final_sum
