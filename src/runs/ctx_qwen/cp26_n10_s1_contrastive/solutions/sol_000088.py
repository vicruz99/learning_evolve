# sol_000088 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000042 (state 26164787) state=2036d242 sum of radii=2.621824 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
import math

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(vars_vec[2*N:])

def constraints(vars_vec):
    """
    Inequality constraints g(vars) >= 0.
    Ensures circles are inside [0,1]^2 and do not overlap.
    """
    centers = vars_vec[:2*N].reshape(N, 2)
    radii = vars_vec[2*N:]
    
    c = []
    # Boundary constraints
    c.append(centers[:, 0] - radii)
    c.append(1.0 - centers[:, 0] - radii)
    c.append(centers[:, 1] - radii)
    c.append(1.0 - centers[:, 1] - radii)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = centers[:, 0, np.newaxis] - centers[:, 0]
    dy = centers[:, 1, np.newaxis] - centers[:, 1]
    dist_sq = dx**2 + dy**2
    
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    c.append(dist_sq[I_IDX, J_IDX] - r_sum[I_IDX, J_IDX]**2)
    
    return np.concatenate(c)

def get_lp_radii(centers):
    """
    Solves LP to maximize sum(r_i) subject to boundary and pairwise constraints.
    Returns optimal radii array.
    """
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    for i in range(n):
        x, y = centers[i]
        for b_val in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b_val)
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
        
    return np.full(n, 1e-6)

def run_packing():
    bounds = [(0.0, 1.0)] * (2*N) + [(1e-7, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_vars = None
    best_sum = -1.0
    
    # Generate initial center configurations
    inits_centers = []
    
    # 1. Hexagonal lattice
    r_est = 0.095
    pts = []
    y = r_est
    row = 0
    while len(pts) < N:
        x_off = (row % 2) * r_est
        x = r_est + x_off
        while x <= 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += math.sqrt(3.0) * r_est
        row += 1
    inits_centers.append(np.array(pts[:N]))
    
    # 2. 5x5 Grid + 1 center
    g = []
    for i in range(5):
        for j in range(5):
            g.append([0.1 + i*0.2, 0.1 + j*0.2])
    g.append([0.5, 0.5])
    inits_centers.append(np.array(g))
    
    # 3. Perturbed variants to explore topology space
    rng = np.random.RandomState(42)
    for seed in range(20):
        base = inits_centers[seed % 2].copy()
        pert = base + rng.randn(N, 2) * (0.02 + 0.01 * seed)
        pert = np.clip(pert, 0.05, 0.95)
        inits_centers.append(pert)
        
    # Phase 1: Optimization from diverse starts
    for ic in inits_centers:
        # Initialize radii using LP for exact boundary alignment
        r_lp = get_lp_radii(ic)
        x0 = np.concatenate([ic.flatten(), r_lp])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-13})
            if np.min(constraints(res.x)) >= -1e-8:
                # LP Projection step: squeeze radii to true feasible limit for optimized centers
                centers_opt = res.x[:2*N].reshape(N, 2)
                radii_proj = get_lp_radii(centers_opt)
                curr_sum = np.sum(radii_proj)
                
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_vars = np.concatenate([centers_opt.flatten(), radii_proj])
        except Exception:
            continue
            
    # Phase 2: Local refinement via perturbation & LP projection
    if best_vars is not None:
        for _ in range(25):
            x_pert = best_vars.copy()
            x_pert[:2*N] += rng.randn(2*N) * 0.002
            x_pert[:2*N] = np.clip(x_pert[:2*N], 0.001, 0.999)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
                if np.min(constraints(res.x)) >= -1e-8:
                    centers_opt = res.x[:2*N].reshape(N, 2)
                    radii_proj = get_lp_radii(centers_opt)
                    curr_sum = np.sum(radii_proj)
                    
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_vars = np.concatenate([centers_opt.flatten(), radii_proj])
            except Exception:
                continue
                
    # Phase 3: High-precision final polish
    if best_vars is not None:
        try:
            res_f = minimize(objective, best_vars, method='SLSQP', bounds=bounds,
                             constraints=cons, options={'maxiter': 5000, 'ftol': 1e-14})
            if np.min(constraints(res_f.x)) >= -1e-8:
                centers_opt = res_f.x[:2*N].reshape(N, 2)
                radii_proj = get_lp_radii(centers_opt)
                best_vars = np.concatenate([centers_opt.flatten(), radii_proj])
        except Exception:
            pass
            
    centers = best_vars[:2*N].reshape(N, 2)
    radii = best_vars[2*N:]
    return centers, radii, float(np.sum(radii))
