# sol_000098 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000064 (state 39c4bccd) state=b011c925 sum of radii=2.367276 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize, differential_evolution

N = 26
I_PAIR, J_PAIR = np.triu_indices(N, k=1)

# Precompute the constant structure of the LP constraint matrix
# Rows 0..4N-1: Wall constraints (x_i, 1-x_i, y_i, 1-y_i)
# Rows 4N..end: Pairwise constraints (r_i + r_j)
A_LP = np.zeros((4 * N + len(I_PAIR), N))
for i in range(N):
    A_LP[i, i] = 1.0
    A_LP[N + i, i] = 1.0
    A_LP[2 * N + i, i] = 1.0
    A_LP[3 * N + i, i] = 1.0
    
for k in range(len(I_PAIR)):
    i, j = I_PAIR[k], J_PAIR[k]
    A_LP[4 * N + k, i] = 1.0
    A_LP[4 * N + k, j] = 1.0

def solve_lp_radii(centers):
    """Solves the LP to find optimal radii for fixed centers."""
    b = np.empty(4 * N + len(I_PAIR))
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Fill wall constraint bounds
    for i in range(N):
        b[i] = x[i]
        b[N + i] = 1.0 - x[i]
        b[2 * N + i] = y[i]
        b[3 * N + i] = 1.0 - y[i]
        
    # Fill pairwise distance bounds
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    b[4 * N:] = dists[I_PAIR, J_PAIR]
    
    res = linprog(-np.ones(N), A_ub=A_LP, b_ub=b, bounds=[(0, None)] * N, method='highs')
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(N)

def center_obj(vars_vec):
    """Objective for center optimization: minimize negative LP-sum-of-radii."""
    centers = vars_vec.reshape(N, 2)
    s, _ = solve_lp_radii(centers)
    return -s

def obj_full(x):
    """Objective for full SLSQP polish: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def full_constraints(x):
    """Inequality constraints g(x) >= 0 for SLSQP."""
    cx = x[0::3]
    cy = x[1::3]
    r = x[2::3]
    
    c = []
    # Boundary containment
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[:, np.newaxis] - cx[np.newaxis, :]
    dy = cy[:, np.newaxis] - cy[np.newaxis, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, np.newaxis] + r[np.newaxis, :]
    c.append(dist2[I_PAIR, J_PAIR] - rs[I_PAIR, J_PAIR]**2)
    
    return np.concatenate(c)

def run_packing():
    # Phase 1: Global search over center positions using Differential Evolution
    bounds_centers = [(0.0, 1.0)] * (2 * N)
    best_centers = None
    
    try:
        de_res = differential_evolution(
            center_obj, bounds_centers, popsize=12, maxiter=200,
            tol=1e-8, mutation=(0.5, 1.0), recombination=0.9,
            seed=42, polishing=False, workers=1
        )
        best_centers = de_res.x.reshape(N, 2)
    except Exception:
        pass
        
    # Fallback initialization if DE fails
    if best_centers is None:
        best_centers = np.zeros((N, 2))
        r_est = 0.095
        y = r_est
        row = 0
        idx = 0
        while y < 1.0 and idx < N:
            x_start = r_est if row % 2 == 0 else 2.0 * r_est
            x = x_start
            while x < 1.0 and idx < N:
                best_centers[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1

    # Phase 2: Get mathematically optimal radii for the best centers found
    best_sum, best_radii = solve_lp_radii(best_centers)
    
    # Phase 3: Local refinement using SLSQP on all variables (centers + radii)
    x0 = np.empty(3 * N)
    x0[0::3] = best_centers[:, 0]
    x0[1::3] = best_centers[:, 1]
    x0[2::3] = best_radii
    
    # Ensure strict interior feasibility for SLSQP's initial steps
    x0[2::3] *= 0.995
    x0[0::3] = np.clip(x0[0::3], 0.01, 0.99)
    x0[1::3] = np.clip(x0[1::3], 0.01, 0.99)
    
    bounds_full = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': full_constraints}
    
    try:
        res_slsqp = minimize(
            obj_full, x0, method='SLSQP', bounds=bounds_full, constraints=cons,
            options={'maxiter': 6000, 'ftol': 1e-14, 'disp': False}
        )
        if res_slsqp.success:
            c_val = full_constraints(res_slsqp.x)
            if np.min(c_val) >= -1e-8:
                best_centers = np.column_stack((res_slsqp.x[0::3], res_slsqp.x[1::3]))
                best_radii = res_slsqp.x[2::3]
                best_sum = np.sum(best_radii)
    except Exception:
        pass
        
    # Final safety & formatting
    best_radii = np.maximum(best_radii, 0.0)
    final_sum = float(np.sum(best_radii))
    
    return best_centers, best_radii, final_sum
