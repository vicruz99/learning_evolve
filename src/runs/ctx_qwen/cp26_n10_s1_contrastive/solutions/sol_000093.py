# sol_000093 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000079 (state c990a719) state=9070bd7e sum of radii=2.608631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 26

def compute_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    n_pair = n * (n - 1) // 2
    n_constr = 4 * n + n_pair
    A_ub = np.zeros((n_constr, n))
    b_ub = np.zeros(n_constr)
    
    x, y = centers[:, 0], centers[:, 1]
    idx = np.arange(n)
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    A_ub[0:n, idx] = 1.0; b_ub[0:n] = x
    A_ub[n:2*n, idx] = 1.0; b_ub[n:2*n] = 1.0 - x
    A_ub[2*n:3*n, idx] = 1.0; b_ub[2*n:3*n] = y
    A_ub[3*n:4*n, idx] = 1.0; b_ub[3*n:4*n] = 1.0 - y
    
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    i_idx, j_idx = np.triu_indices(n, k=1)
    base = 4 * n
    A_ub[base:base+n_pair, i_idx] = 1.0
    A_ub[base:base+n_pair, j_idx] = 1.0
    b_ub[base:base+n_pair] = dists[i_idx, j_idx]
    
    bounds = [(0.0, None)] * n
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def fitness_de(centers_flat):
    """Objective for DE: maximize sum of radii via LP."""
    centers = centers_flat.reshape(N, 2)
    s, _ = compute_lp_radii(centers)
    return -s  # DE minimizes

def obj_slq(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def cons_slq(v):
    """Inequality constraints for SLSQP: boundary and non-overlap."""
    xc = v[0::3]
    yc = v[1::3]
    r = v[2::3]
    c = []
    c.append(xc - r)
    c.append(1.0 - xc - r)
    c.append(yc - r)
    c.append(1.0 - yc - r)
    dx = xc[:, None] - xc[None, :]
    dy = yc[:, None] - yc[None, :]
    d2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    i_idx, j_idx = np.triu_indices(N, k=1)
    c.append(d2[i_idx, j_idx] - rs[i_idx, j_idx]**2)
    return np.concatenate(c)

def run_packing():
    bounds_c = [(0.0, 1.0)] * (2 * N)
    rng = np.random.RandomState(123)
    
    # Generate structured initial population for DE
    init_pop = []
    pts_hex = []
    r_est = 0.098
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
    base_pop = np.array(pts_hex[:N]).flatten()
    init_pop.append(base_pop)
    
    for _ in range(9):
        pert = base_pop.copy()
        pert += rng.uniform(-0.06, 0.06, 2 * N)
        pert = np.clip(pert, 0.01, 0.99)
        init_pop.append(pert)
        
    init_pop = np.array(init_pop)
    
    # Phase 1: Global search on centers using Differential Evolution
    try:
        de_res = differential_evolution(fitness_de, bounds_c, seed=42, maxiter=300, 
                                        popsize=10, tol=1e-7, mutation=(0.5, 1.0), 
                                        recombination=0.9, init=init_pop, polishing=False)
        best_c = de_res.x.reshape(N, 2)
        best_s, best_r = compute_lp_radii(best_c)
    except Exception:
        best_c = init_pop[0].reshape(N, 2)
        best_s, best_r = compute_lp_radii(best_c)

    # Phase 2: High-precision joint refinement with SLSQP
    bounds_slq = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
    cons_slq_dict = {'type': 'ineq', 'fun': cons_slq}
    
    x0 = np.zeros(3 * N)
    x0[0::3] = best_c[:, 0]
    x0[1::3] = best_c[:, 1]
    x0[2::3] = best_r * 0.98  # Shrink slightly to ensure strict initial feasibility
    
    res_slq = minimize(obj_slq, x0, method='SLSQP', bounds=bounds_slq,
                       constraints=cons_slq_dict, options={'maxiter': 4000, 'ftol': 1e-13})
    
    if res_slq.success:
        c_val = cons_slq(res_slq.x)
        if np.min(c_val) >= -1e-8:
            best_c = np.column_stack((res_slq.x[0::3], res_slq.x[1::3]))
            best_r = res_slq.x[2::3]
            best_s = -res_slq.fun
            
    # Phase 3: Local perturbation search to escape shallow local optima
    for k in range(20):
        x_pert = np.concatenate([best_c.flatten(), best_r])
        x_pert += rng.normal(0, 0.003, 3 * N)
        x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
        x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
        x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.49)
        
        try:
            res_p = minimize(obj_slq, x_pert, method='SLSQP', bounds=bounds_slq,
                             constraints=cons_slq_dict, options={'maxiter': 2000, 'ftol': 1e-12})
            if res_p.success:
                c_val = cons_slq(res_p.x)
                if np.min(c_val) >= -1e-8:
                    s_new = -res_p.fun
                    if s_new > best_s:
                        best_s = s_new
                        best_c = np.column_stack((res_p.x[0::3], res_p.x[1::3]))
                        best_r = res_p.x[2::3]
        except Exception:
            continue

    # Fallback configuration (should rarely be triggered)
    if best_s < 1.0:
        best_c = np.column_stack([np.linspace(0.1, 0.9, 6).repeat(5)[:N], 
                                  np.tile(np.linspace(0.1, 0.9, 5), 6)[:N]])
        best_r = np.full(N, 0.05)
        best_s = np.sum(best_r)

    return best_c, best_r, float(best_s)
