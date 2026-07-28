# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000046 (state 0aa7241c) state=79032cee sum of radii=1.685806 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def objective(vars):
    """Minimize negative sum of radii => Maximize sum of radii"""
    return -np.sum(vars[2::3])

def constraints(vars, n):
    """
    Computes inequality constraints >= 0 for valid packing.
    Uses squared distances for numerical stability.
    """
    x = vars[0::3]
    y = vars[1::3]
    r = vars[2::3]
    
    # Boundary constraints: circle must be inside [0,1]x[0,1]
    # x >= r, 1-x >= r, y >= r, 1-y >= r
    c_boundary = np.concatenate([x - r, 1.0 - x - r, y - r, 1.0 - y - r])
    
    # Pairwise non-overlap constraints: ||c_i - c_j||^2 >= (r_i + r_j)^2
    idx = np.triu_indices(n, k=1)
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dr = r[:, None] + r[None, :]
    
    dist_sq = dx**2 + dy**2
    r_sum_sq = dr**2
    
    c_pairwise = dist_sq[idx] - r_sum_sq[idx]
    
    return np.concatenate([c_boundary, c_pairwise])

def generate_hex_config(n, row_counts, r_init):
    """Generates an initial hexagonal grid layout for n circles."""
    centers = []
    y = r_init
    row_idx = 0
    for count in row_counts:
        shift = r_init if row_idx % 2 == 1 else 0.0
        x = r_init + shift
        for _ in range(count):
            if len(centers) < n:
                centers.append([x, y])
            x += 2.0 * r_init
        y += np.sqrt(3) * r_init
        row_idx += 1
        
    # Fallback if layout didn't yield enough points
    while len(centers) < n:
        centers.append([0.5, 0.5])
        
    return np.array(centers[:n])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': constraints, 'args': (n,)}
    
    best_sum = -1.0
    best_vars = None
    
    np.random.seed(42)
    
    # Known dense row distributions for 26 circles in a square
    row_dists = [
        [5, 6, 5, 6, 4], [6, 5, 6, 5, 4], [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4], [6, 6, 6, 4, 4], [5, 6, 6, 5, 4], [6, 5, 5, 6, 4]
    ]
    
    configs = []
    for rd in row_dists:
        pts = generate_hex_config(n, rd, 0.095)
        # Create multiple perturbed versions to break symmetry
        for _ in range(4):
            noise = np.random.uniform(-0.015, 0.015, pts.shape)
            p = np.clip(pts + noise, 0.05, 0.95)
            v = np.zeros(3 * n)
            v[0::3] = p[:, 0]
            v[1::3] = p[:, 1]
            v[2::3] = 0.09 + np.random.uniform(-0.005, 0.005)
            configs.append(v)
            
    # Add purely random starts to explore non-hexagonal minima
    for _ in range(5):
        v = np.zeros(3 * n)
        v[0::3] = np.random.uniform(0.1, 0.9, n)
        v[1::3] = np.random.uniform(0.1, 0.9, n)
        v[2::3] = 0.05
        configs.append(v)
        
    # Phase 1: Optimize from initial configurations
    for v0 in configs:
        try:
            res = opt.minimize(objective, v0, method='SLSQP', bounds=bounds,
                              constraints=cons, options={'maxiter': 8000, 'ftol': 1e-12})
            if res.success or res.fun < -2.5:
                v_opt = res.x
                r_opt = v_opt[2::3]
                if np.all(r_opt > 1e-5):
                    s = np.sum(r_opt)
                    if s > best_sum:
                        best_sum = s
                        best_vars = v_opt.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation loop to escape local minima (Basin Hopping style)
    if best_vars is not None:
        for iter in range(15):
            scale_pert = 0.006 * (1.0 - iter/15.0)
            pert = np.random.normal(0, scale_pert, best_vars.shape)
            v_pert = best_vars + pert
            
            # Enforce bounds on perturbed start
            v_pert[0::3] = np.clip(v_pert[0::3], 0.0, 1.0)
            v_pert[1::3] = np.clip(v_pert[1::3], 0.0, 1.0)
            v_pert[2::3] = np.clip(v_pert[2::3], 1e-6, 0.5)
            
            try:
                res = opt.minimize(objective, v_pert, method='SLSQP', bounds=bounds,
                                  constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12})
                if res.success or res.fun < best_sum - 1e-6:
                    v_opt = res.x
                    r_opt = v_opt[2::3]
                    if np.all(r_opt > 1e-5):
                        s = np.sum(r_opt)
                        if s > best_sum:
                            best_sum = s
                            best_vars = v_opt.copy()
            except Exception:
                continue

    # Fallback if optimization fails unexpectedly
    if best_vars is None:
        best_vars = np.zeros(3*n)
        best_vars[0::3] = np.tile(np.linspace(0.15, 0.85, 5), 6)[:n]
        best_vars[1::3] = np.repeat(np.linspace(0.15, 0.85, 6), 5)[:n]
        best_vars[2::3] = 0.08
        best_sum = np.sum(best_vars[2::3])

    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    
    # Phase 3: Radius refinement
    # Fix centers and compute exact maximal feasible radius for each circle independently.
    # This often increases the sum because the joint optimizer may leave circles 
    # smaller than necessary due to constraint coupling or tolerance limits.
    radii = np.zeros(n)
    for i in range(n):
        min_d = min(centers[i,0], 1.0 - centers[i,0], centers[i,1], 1.0 - centers[i,1])
        for j in range(n):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < min_d:
                    min_d = d
        radii[i] = min_d / 2.0
        
    # Safety scaling to strictly satisfy the validator's 1e-12 tolerance
    radii *= 0.99999
    centers = np.clip(centers, 1e-9, 1.0 - 1e-9)
    final_sum = np.sum(radii)
    
    return centers, radii, float(final_sum)
