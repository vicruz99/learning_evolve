# sol_000023 | problem=circle_packing_26 entrypoint=run_packing
# generation=1 parent=sol_000011 (state bbbe9bd5) state=09043dd0 sum of radii=2.509105 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N_CIRCLES = 26

def solve_max_radii(centers):
    """Given fixed centers, solve LP to maximize sum of radii."""
    n = centers.shape[0]
    c_obj = -np.ones(n)
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, 1-x_i, y_i, 1-y_i
    for i in range(n):
        x, y = centers[i]
        for b in [x, 1.0 - x, y, 1.0 - y]:
            row = np.zeros(n)
            row[i] = 1.0
            A_ub.append(row)
            b_ub.append(b)
            
    # Pairwise constraints: r_i + r_j <= dist_ij
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds_r = [(0.0, None)] * n
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
        if res.success:
            return -res.fun, res.x
    except Exception:
        pass
    return 0.0, np.zeros(n)

def center_objective(centers_flat):
    """Objective for center optimization: maximize sum of optimal radii."""
    centers = centers_flat.reshape(N_CIRCLES, 2)
    centers = np.clip(centers, 0.0, 1.0)
    s, _ = solve_max_radii(centers)
    return -s

def full_objective(vars):
    """Objective for full SLSQP optimization."""
    return -np.sum(vars[2::3])

def full_constraints(vars):
    """Constraints for full SLSQP optimization."""
    centers = vars[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = vars[2*N_CIRCLES:]
    
    cons = []
    # Boundary
    cons.extend(centers[:, 0] - radii)
    cons.extend(1.0 - centers[:, 0] - radii)
    cons.extend(centers[:, 1] - radii)
    cons.extend(1.0 - centers[:, 1] - radii)
    
    # Pairwise
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    ii, jj = np.triu_indices(N_CIRCLES, k=1)
    cons.extend(dist_sq[ii, jj] - r_sum[ii, jj]**2)
    
    return np.array(cons)

def run_packing():
    np.random.seed(42)
    best_centers = None
    best_radii = None
    best_sum = -1.0
    
    # Generate initial configurations
    inits = []
    
    # 1. Hexagonal lattice
    r_init = 0.09
    pts = []
    y = r_init
    row = 0
    while len(pts) < N_CIRCLES:
        x_off = r_init if row % 2 == 0 else 2 * r_init
        x = x_off
        while x <= 1.0 - r_init and len(pts) < N_CIRCLES:
            pts.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row += 1
    inits.append(np.array(pts[:N_CIRCLES]))
    
    # 2. Perturbed Hexagonal
    inits.append(inits[0] + np.random.randn(*inits[0].shape) * 0.02)
    
    # 3. Square grid + jitter
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    grid_pts.append([0.5, 0.5])
    inits.append(np.array(grid_pts) + np.random.randn(N_CIRCLES, 2) * 0.01)
    
    # Optimize centers using Nelder-Mead
    for init in inits:
        init_flat = np.clip(init, 0.01, 0.99).flatten()
        try:
            res = minimize(center_objective, init_flat, method='Nelder-Mead',
                           options={'xatol': 1e-6, 'fatol': 1e-8, 'maxiter': 3000})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res.x.reshape(N_CIRCLES, 2)
        except Exception:
            pass
            
    # Refine with SLSQP on full variables
    if best_centers is not None:
        # Get corresponding radii for initialization
        _, radii_init = solve_max_radii(best_centers)
        vars0 = np.zeros(3 * N_CIRCLES)
        for i in range(N_CIRCLES):
            vars0[3*i] = best_centers[i, 0]
            vars0[3*i+1] = best_centers[i, 1]
            vars0[3*i+2] = radii_init[i]
            
        bounds = [(0.0, 1.0)] * (2*N_CIRCLES) + [(0.0, 0.5)] * N_CIRCLES
        try:
            res_full = minimize(full_objective, vars0, method='SLSQP', bounds=bounds,
                                constraints={'type': 'ineq', 'fun': full_constraints},
                                options={'maxiter': 8000, 'ftol': 1e-13})
            if res_full.success:
                best_centers = res_full.x[:2*N_CIRCLES].reshape(N_CIRCLES, 2)
                best_radii = res_full.x[2*N_CIRCLES:]
                best_sum = np.sum(best_radii)
            else:
                _, best_radii = solve_max_radii(best_centers)
                best_sum = np.sum(best_radii)
        except Exception:
            _, best_radii = solve_max_radii(best_centers)
            best_sum = np.sum(best_radii)
            
    # Safety clip to ensure strict validity within tolerance
    if best_centers is not None:
        best_centers = np.clip(best_centers, 1e-9, 1.0 - 1e-9)
        best_radii = np.maximum(best_radii, 0.0)
        
    return best_centers, best_radii, float(best_sum)
