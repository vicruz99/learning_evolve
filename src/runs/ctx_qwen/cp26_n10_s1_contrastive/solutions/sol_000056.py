# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000043 (state 8d6d3048) state=9dc58de0 sum of radii=2.617835 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import cdist

def compute_lp_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii subject to non-overlap and boundary constraints."""
    n = centers.shape[0]
    c_obj = -np.ones(n)  # Minimize negative sum => maximize sum
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        bounds_vals = [x, 1.0 - x, y, 1.0 - y]
        for bv in bounds_vals:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(bv)
            
    # Pairwise constraints: r_i + r_j <= dist(i, j)
    dists = cdist(centers, centers)
    np.fill_diagonal(dists, np.inf)
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_lp = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_lp, method='highs')
        if res.success:
            return res.fun * -1.0, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def joint_objective(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def joint_constraints(x):
    """Inequality constraints: g(x) >= 0."""
    n = len(x) // 3
    xc = x[0::3]
    yc = x[1::3]
    r = x[2::3]
    
    c = []
    # Boundary constraints
    c.append(xc - r)
    c.append(1.0 - xc - r)
    c.append(yc - r)
    c.append(1.0 - yc - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = xc[:, np.newaxis] - xc[np.newaxis, :]
    dy = yc[:, np.newaxis] - yc[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    r_sum = r[:, np.newaxis] + r[np.newaxis, :]
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    c.append(dist_sq[i_idx, j_idx] - r_sum[i_idx, j_idx]**2)
    
    return np.concatenate(c)

def force_repulsion_init(n, seed):
    """Generate well-spread centers using force-directed layout."""
    np.random.seed(seed)
    pts = np.random.rand(n, 2) * 0.8 + 0.1
    for _ in range(400):
        forces = np.zeros_like(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, 1.0)
        
        # Coulomb-like repulsion, masked to short range for efficiency
        rep_strength = 0.005 / (dists**2 + 1e-5)
        rep_strength[dists > 0.4] = 0.0
        forces += np.sum(diff * rep_strength[:, :, np.newaxis], axis=1)
        
        # Wall repulsion
        for i in range(n):
            x, y = pts[i]
            margin = 0.1
            if x < margin: forces[i, 0] += 2.0 * (margin - x)
            elif x > 1.0 - margin: forces[i, 0] -= 2.0 * (x - (1.0 - margin))
            if y < margin: forces[i, 1] += 2.0 * (margin - y)
            elif y > 1.0 - margin: forces[i, 1] -= 2.0 * (y - (1.0 - margin))
            
        pts += forces * 0.15
        pts = np.clip(pts, 0.02, 0.98)
    return pts

def generate_hex_init(n):
    pts = []
    r_est = 0.095
    y = r_est
    row = 0
    while y < 1.0 and len(pts) < n:
        x_start = r_est if row % 2 == 0 else 2.0 * r_est
        x = x_start
        while x < 1.0 and len(pts) < n:
            pts.append([x, y])
            x += 2.0 * r_est
        y += np.sqrt(3.0) * r_est
        row += 1
    return np.array(pts[:n])

def generate_grid_init(n):
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([0.1 + 0.2 * i, 0.1 + 0.2 * j])
    if n > 25:
        pts.append([0.5, 0.5])
    return np.array(pts[:n])

def run_packing():
    n = 26
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * n
    cons = {'type': 'ineq', 'fun': joint_constraints}
    
    best_x = None
    best_sum = -np.inf
    
    # Diverse center initializations
    center_inits = [generate_hex_init(n), generate_grid_init(n)]
    for s in range(6):
        center_inits.append(force_repulsion_init(n, s))
        
    # Phase 1: LP-optimized radii + SLSQP joint refinement
    x0_candidates = []
    for centers in center_inits:
        _, radii = compute_lp_radii(centers)
        x0 = np.zeros(3 * n)
        x0[0::3] = centers[:, 0]
        x0[1::3] = centers[:, 1]
        x0[2::3] = radii
        # Shrink radii slightly to ensure strict interior feasibility for SLSQP start
        x0[2::3] *= 0.92
        x0[0::3] = np.clip(x0[0::3], 0.02, 0.98)
        x0[1::3] = np.clip(x0[1::3], 0.02, 0.98)
        x0_candidates.append(x0)
        
    for x0 in x0_candidates:
        try:
            res = minimize(joint_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if np.min(joint_constraints(res.x)) >= -1e-5:
                curr_sum = np.sum(res.x[2::3])
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Perturbation refinement to escape local minima
    if best_x is not None:
        for _ in range(12):
            x0_pert = best_x.copy()
            x0_pert += np.random.uniform(-0.003, 0.003, x0_pert.shape)
            x0_pert[0::3] = np.clip(x0_pert[0::3], 0.02, 0.98)
            x0_pert[1::3] = np.clip(x0_pert[1::3], 0.02, 0.98)
            x0_pert[2::3] = np.clip(x0_pert[2::3], 1e-6, 0.49)
            
            try:
                res_p = minimize(joint_objective, x0_pert, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                if np.min(joint_constraints(res_p.x)) >= -1e-5:
                    curr_sum = np.sum(res_p.x[2::3])
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res_p.x.copy()
            except Exception:
                continue
                
        # Phase 3: High-precision final polish
        try:
            res_final = minimize(joint_objective, best_x, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            if np.min(joint_constraints(res_final.x)) >= -1e-5:
                best_x = res_final.x
        except Exception:
            pass

    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    return centers, radii, float(np.sum(radii))
