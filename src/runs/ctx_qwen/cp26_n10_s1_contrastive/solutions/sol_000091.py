# sol_000091 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000079 (state c990a719) state=b63eb553 sum of radii=2.613222 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26
I_IDX, J_IDX = np.triu_indices(N, k=1)

def get_lp_radii(centers):
    """Given fixed centers, solve LP to find radii maximizing sum(r_i)."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x, r_i <= 1-x, r_i <= y, r_i <= 1-y
    for i in range(n):
        x, y = centers[i]
        row = np.zeros(n); row[i] = 1.0
        A_ub.extend([row, row, row, row])
        b_ub.extend([x, 1.0-x, y, 1.0-y])
        
    # Pairwise constraints: r_i + r_j <= dist(i,j)
    for i in range(n):
        for j in range(i+1, n):
            d = np.sqrt(np.sum((centers[i] - centers[j])**2))
            if d < 1e-9: d = 1e-9
            row = np.zeros(n); row[i] = 1.0; row[j] = 1.0
            A_ub.append(row)
            b_ub.append(d)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def obj_centers(centers_flat):
    """Objective for center optimization: minimize negative sum of LP-optimal radii."""
    centers = centers_flat.reshape(N, 2)
    s, _ = get_lp_radii(centers)
    return -s

def constraints_slqp(v):
    """Inequality constraints for SLSQP: boundary containment and non-overlap."""
    x = v[0::3]; y = v[1::3]; r = v[2::3]
    c = []
    c.append(x - r)
    c.append(1.0 - x - r)
    c.append(y - r)
    c.append(1.0 - y - r)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist2 = dx**2 + dy**2
    rs = r[:, None] + r[None, :]
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def obj_slqp(v):
    """Objective for SLSQP: minimize negative sum of radii."""
    return -np.sum(v[2::3])

def generate_inits():
    """Generate diverse initial center configurations."""
    inits = []
    # Hexagonal lattice base
    r_est = 0.095
    y = r_est
    row = 0
    pts = []
    while len(pts) < N:
        x = r_est + (row % 2) * r_est
        while x < 1.0 - r_est and len(pts) < N:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    pts = np.array(pts[:N])
    
    rng = np.random.RandomState(42)
    # Perturbed hexagonal starts
    for i in range(4):
        p = pts + rng.uniform(-0.03, 0.03, pts.shape)
        p = np.clip(p, 0.02, 0.98)
        inits.append(p)
        
    # Force-spread random starts
    for i in range(4):
        p = rng.rand(N, 2) * 0.8 + 0.1
        for _ in range(50):
            diff = p[:, None, :] - p[None, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            dists = np.maximum(dists, 1e-5)
            f = np.sum((1.0 / dists**2)[:, :, None] * diff / dists[:, :, None], axis=1)
            p += 0.003 * f
            p = np.clip(p, 0.02, 0.98)
        inits.append(p)
    return inits

def run_packing():
    np.random.seed(123)
    inits = generate_inits()
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Phase 1: Optimize centers using LP for radii evaluation
    for init_pts in inits:
        c0 = np.clip(init_pts, 0.01, 0.99).flatten()
        try:
            res = minimize(obj_centers, c0, method='Nelder-Mead', 
                           options={'maxiter': 120, 'xatol': 1e-5, 'fatol': 1e-7})
            curr_centers = np.clip(res.x.reshape(N, 2), 0.001, 0.999)
            curr_sum, curr_radii = get_lp_radii(curr_centers)
            
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = curr_centers
                best_radii = curr_radii
        except Exception:
            continue
            
    # Phase 2 & 3: SLSQP polish and perturbation refinement on full variables
    if best_centers is not None:
        bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N
        cons = {'type': 'ineq', 'fun': constraints_slqp}
        
        x0 = np.zeros(3 * N)
        x0[0::3] = best_centers[:, 0]
        x0[1::3] = best_centers[:, 1]
        x0[2::3] = best_radii * 0.995  # Shrink slightly for strict feasibility
        
        try:
            res_final = minimize(obj_slqp, x0, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 3000, 'ftol': 1e-13})
            if res_final.success:
                c_val = constraints_slqp(res_final.x)
                if np.min(c_val) >= -1e-8:
                    final_sum = -res_final.fun
                    if final_sum > best_sum:
                        best_sum = final_sum
                        best_centers = np.column_stack((res_final.x[0::3], res_final.x[1::3]))
                        best_radii = res_final.x[2::3]
        except Exception:
            pass

        # Local perturbations to escape remaining local minima
        for _ in range(10):
            rng = np.random.RandomState(None)
            x_pert = np.concatenate([best_centers.flatten(), best_radii])
            x_pert += rng.normal(0, 0.003, x_pert.shape)
            x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.49)
            
            try:
                res_pert = minimize(obj_slqp, x_pert, method='SLSQP', bounds=bounds,
                                    constraints=cons, options={'maxiter': 1500, 'ftol': 1e-12})
                if res_pert.success:
                    c_val = constraints_slqp(res_pert.x)
                    if np.min(c_val) >= -1e-8:
                        p_sum = -res_pert.fun
                        if p_sum > best_sum:
                            best_sum = p_sum
                            best_centers = np.column_stack((res_pert.x[0::3], res_pert.x[1::3]))
                            best_radii = res_pert.x[2::3]
            except Exception:
                continue

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
