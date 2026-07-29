# sol_000103 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000077 (state eb8dc077) state=70e212a5 sum of radii=2.607074 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, differential_evolution, minimize

N_CIRCLES = 26

def solve_radii_lp(centers):
    """
    Given fixed centers, solves the LP to find radii that maximize sum(r_i).
    Constraints:
    1. r_i >= 0
    2. r_i <= distance to boundaries
    3. r_i + r_j <= distance(centers[i], centers[j])
    """
    n = centers.shape[0]
    # Clip centers slightly inside to avoid degenerate LP bases
    centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
    
    num_constraints = 4 * n + n * (n - 1) // 2
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        bounds_val = [x, 1.0 - x, y, 1.0 - y]
        for b in bounds_val:
            A_ub[idx, i] = 1.0
            b_ub[idx] = b
            idx += 1
            
    # Pairwise constraints
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dists[i, j]
            idx += 1
            
    bounds = [(0.0, None)] * n
    
    try:
        res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success and res.fun is not None:
            return np.maximum(res.x, 0.0), -res.fun
    except Exception:
        pass
        
    # Fallback
    return np.full(n, 1e-5), 1e-4

def objective_centers(x):
    """Objective for center optimization: minimize negative sum of radii."""
    centers = x.reshape(N_CIRCLES, 2)
    _, sum_r = solve_radii_lp(centers)
    return -sum_r

def generate_hex_init(seed, rotation=0.0, scale=1.0):
    """Generate a diverse initial center configuration based on a hexagonal lattice."""
    np.random.seed(seed)
    patterns = [
        [5, 6, 5, 6, 4],
        [6, 5, 6, 5, 4],
        [4, 6, 5, 6, 5],
        [5, 5, 5, 5, 6]
    ]
    pat = patterns[seed % len(patterns)]
    pts = []
    y = 0.06
    r_est = 0.10
    dy = r_est * np.sqrt(3.0)
    
    for r_idx, cnt in enumerate(pat):
        shift = (r_idx % 2) * r_est
        x = 0.06 + shift
        for _ in range(cnt):
            if len(pts) < N_CIRCLES:
                pts.append([x, y])
            x += 2.0 * r_est
        y += dy
        
    pts = np.array(pts[:N_CIRCLES])
    
    # Center, scale, and rotate
    pts = (pts - 0.5) * scale + 0.5
    if rotation != 0.0:
        c, s = np.cos(rotation), np.sin(rotation)
        rot_mat = np.array([[c, -s], [s, c]])
        pts = (pts - 0.5) @ rot_mat.T + 0.5
        
    # Add jitter to break symmetry
    jitter = np.random.randn(N_CIRCLES, 2) * 0.02
    pts = np.clip(pts + jitter, 0.02, 0.98)
    return pts.flatten()

def run_packing():
    np.random.seed(42)
    best_sum = -np.inf
    best_centers = None
    best_radii = None
    
    # Phase 1: Global search on centers using Differential Evolution
    bounds_centers = [(0.0, 1.0)] * (2 * N_CIRCLES)
    
    # Generate a diverse set of initial populations to guide DE
    init_population = []
    for i in range(15):
        init_population.append(generate_hex_init(i, rotation=np.random.uniform(-0.15, 0.15), scale=np.random.uniform(0.9, 1.1)))
        
    try:
        res_de = differential_evolution(
            objective_centers, 
            bounds_centers,
            strategy='best1bin',
            maxiter=80,
            popsize=12,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=123,
            initial_guess=init_population,
            polishing=False
        )
        if res_de.success:
            curr_sum = -res_de.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_centers = res_de.x.reshape(N_CIRCLES, 2)
    except Exception:
        pass

    # Fallback if DE fails
    if best_centers is None:
        best_centers = generate_hex_init(0).reshape(N_CIRCLES, 2)

    # Phase 2: Local refinement on centers using Nelder-Mead
    # Nelder-Mead handles the non-smooth LP objective well
    try:
        res_nm = minimize(objective_centers, best_centers.flatten(), method='Nelder-Mead',
                          options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-9})
        curr_sum = -res_nm.fun
        if curr_sum > best_sum:
            best_sum = curr_sum
            best_centers = res_nm.x.reshape(N_CIRCLES, 2)
    except Exception:
        pass

    # Solve LP to get corresponding optimal radii
    best_radii, best_sum = solve_radii_lp(best_centers)

    # Phase 3: Joint optimization with SLSQP for final precision polish
    x0_joint = np.zeros(3 * N_CIRCLES)
    x0_joint[0::3] = best_centers[:, 0]
    x0_joint[1::3] = best_centers[:, 1]
    # Slightly shrink radii to ensure strict feasibility for SLSQP's interior steps
    x0_joint[2::3] = np.maximum(best_radii, 1e-6) * 0.995
    
    def obj_joint(x):
        return -np.sum(x[2::3])
        
    def cons_joint(x):
        C = x.reshape(N_CIRCLES, 3)
        xc, yc, r = C[:, 0], C[:, 1], C[:, 2]
        c_list = []
        # Boundary constraints
        c_list.append(xc - r)
        c_list.append(1.0 - xc - r)
        c_list.append(yc - r)
        c_list.append(1.0 - yc - r)
        
        # Pairwise separation constraints
        i_idx, j_idx = np.triu_indices(N_CIRCLES, k=1)
        dx = xc[i_idx] - xc[j_idx]
        dy = yc[i_idx] - yc[j_idx]
        r_sum = r[i_idx] + r[j_idx]
        c_list.append(dx*dx + dy*dy - r_sum*r_sum)
        return np.concatenate(c_list)
        
    bounds_joint = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': cons_joint}
    
    try:
        res_joint = minimize(obj_joint, x0_joint, method='SLSQP', bounds=bounds_joint,
                             constraints=cons, options={'maxiter': 10000, 'ftol': 1e-13, 'disp': False})
        if res_joint.success:
            cons_vals = cons_joint(res_joint.x)
            if np.min(cons_vals) >= -1e-7:
                curr_sum = -res_joint.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_centers = res_joint.x.reshape(N_CIRCLES, 3)[:, :2]
                    best_radii = res_joint.x[2::3]
    except Exception:
        pass

    # Phase 4: Perturbation search around the best found solution to escape local minima
    for _ in range(20):
        x_pert = np.concatenate([best_centers.flatten(), best_radii])
        x_pert += np.random.randn(3 * N_CIRCLES) * 0.004
        x_pert[0::3] = np.clip(x_pert[0::3], 0.01, 0.99)
        x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
        x_pert[2::3] = np.clip(x_pert[2::3], 1e-5, 0.49)
        
        try:
            res_pert = minimize(obj_joint, x_pert, method='SLSQP', bounds=bounds_joint,
                                constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            if res_pert.success:
                cons_vals = cons_joint(res_pert.x)
                if np.min(cons_vals) >= -1e-7:
                    curr_sum = -res_pert.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_centers = res_pert.x.reshape(N_CIRCLES, 3)[:, :2]
                        best_radii = res_pert.x[2::3]
        except Exception:
            continue

    best_radii = np.maximum(best_radii, 0.0)
    return best_centers, best_radii, float(best_sum)
