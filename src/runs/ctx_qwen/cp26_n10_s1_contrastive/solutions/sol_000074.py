# sol_000074 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=ebc36b4a sum of radii=2.627070 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def objective_func(x):
    """Minimize negative sum of radii."""
    return -np.sum(x[2::3])

def constraint_func(x):
    """Compute inequality constraints: boundary containment and pairwise separation."""
    n = N_CIRCLES
    C = x.reshape(n, 3)
    xc = C[:, 0]
    yc = C[:, 1]
    r = C[:, 2]
    
    c = []
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    c.append(xc - r)
    c.append(1.0 - xc - r)
    c.append(yc - r)
    c.append(1.0 - yc - r)
    
    # Pairwise separation: dist^2 >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(n, k=1)
    dx = xc[i_idx] - xc[j_idx]
    dy = yc[i_idx] - yc[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c)

def get_strictly_feasible_radii(centers):
    """Compute initial radii that guarantee strict feasibility."""
    n = centers.shape[0]
    radii = np.zeros(n)
    for i in range(n):
        d_wall = min(centers[i, 0], 1.0 - centers[i, 0], 
                     centers[i, 1], 1.0 - centers[i, 1])
        d_min = np.inf
        for j in range(n):
            if i != j:
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if d < d_min:
                    d_min = d
        # Use 0.45 factor to give optimizer room to expand radii
        radii[i] = 0.45 * min(d_wall, 0.5 * d_min)
    return np.maximum(radii, 1e-5)

def generate_init_config(seed, config_type):
    """Generate a strictly feasible initial configuration."""
    np.random.seed(seed)
    centers = np.zeros((N_CIRCLES, 2))
    
    if config_type == 0:  # Hexagonal lattice
        r_est = 0.098
        y = r_est
        row = 0
        idx = 0
        while y < 1.0 - r_est + 0.01 and idx < N_CIRCLES:
            x_start = r_est if row % 2 == 0 else 2.0 * r_est
            x = x_start
            while x < 1.0 - r_est + 0.01 and idx < N_CIRCLES:
                centers[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
        centers = centers[:N_CIRCLES]
    elif config_type == 1:  # 5x5 Grid + 1 center
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i*0.2, 0.1 + j*0.2])
        pts.append([0.5, 0.5])
        centers = np.array(pts[:N_CIRCLES])
    else:  # Random spread
        centers = np.random.uniform(0.12, 0.88, (N_CIRCLES, 2))
        
    # Add controlled perturbation
    pert = np.random.normal(0, 0.02, centers.shape)
    centers += pert
    centers = np.clip(centers, 0.02, 0.98)
    
    radii = get_strictly_feasible_radii(centers)
    
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii
    return x0

def run_packing():
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_sum = -np.inf
    best_x = None
    
    # Phase 1: Broad search with diverse initializations
    for seed in range(60):
        config_type = seed % 3
        x0 = generate_init_config(seed, config_type)
        
        try:
            res = minimize(objective_func, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 12000, 'ftol': 1e-13})
            
            if res.success:
                cons_val = constraint_func(res.x)
                if np.min(cons_val) >= -1e-8:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
        except Exception:
            continue

    # Phase 2: Local perturbation refinement on the best found configuration
    if best_x is not None:
        for _ in range(25):
            x_pert = best_x.copy()
            # Perturb centers
            x_pert[0::3] += np.random.normal(0, 0.006, N_CIRCLES)
            x_pert[1::3] += np.random.normal(0, 0.006, N_CIRCLES)
            # Perturb radii slightly
            x_pert[2::3] *= np.random.uniform(0.96, 1.04, N_CIRCLES)
            
            # Clip to bounds to maintain feasibility during optimization
            x_pert[0::3] = np.clip(x_pert[0::3], 0.0, 1.0)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.0, 1.0)
            x_pert[2::3] = np.clip(x_pert[2::3], 1e-6, 0.5)
            
            try:
                res = minimize(objective_func, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 8000, 'ftol': 1e-13})
                if res.success:
                    cons_val = constraint_func(res.x)
                    if np.min(cons_val) >= -1e-8:
                        curr_sum = -res.fun
                        if curr_sum > best_sum:
                            best_sum = curr_sum
                            best_x = res.x.copy()
            except Exception:
                continue

    # Fallback if optimization completely fails
    if best_x is None:
        best_x = generate_init_config(0, 0)
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
