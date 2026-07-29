# sol_000068 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000027 (state bf2de84b) state=8bb0dee5 sum of radii=2.622766 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26

def compute_objective(x):
    """Objective function: minimize negative sum of radii."""
    return -np.sum(x[2::3])

def compute_constraints(x):
    """Compute all inequality constraints: boundary containment and pairwise separation."""
    N = N_CIRCLES
    C = x.reshape(N, 3)
    xc = C[:, 0]
    yc = C[:, 1]
    r = C[:, 2]
    
    c_list = []
    # Boundary constraints: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    c_list.append(xc - r)
    c_list.append(1.0 - xc - r)
    c_list.append(yc - r)
    c_list.append(1.0 - yc - r)
    
    # Pairwise separation constraints: dist_sq >= (r_i + r_j)^2
    i_idx, j_idx = np.triu_indices(N, k=1)
    dx = xc[i_idx] - xc[j_idx]
    dy = yc[i_idx] - yc[j_idx]
    r_sum = r[i_idx] + r[j_idx]
    c_list.append(dx*dx + dy*dy - r_sum*r_sum)
    
    return np.concatenate(c_list)

def generate_init_config(seed, pattern):
    """Generate a strictly feasible initial configuration."""
    np.random.seed(seed)
    pts = np.zeros((N_CIRCLES, 2))
    
    if pattern == 0:  # Hexagonal lattice
        r_est = 0.09
        y_pos = r_est
        row = 0
        idx = 0
        while idx < N_CIRCLES:
            shift = (row % 2) * r_est
            x_pos = r_est + shift
            while x_pos <= 1.0 - r_est and idx < N_CIRCLES:
                pts[idx] = [x_pos, y_pos]
                idx += 1
                x_pos += 2.0 * r_est
            y_pos += np.sqrt(3.0) * r_est
            row += 1
    elif pattern == 1:  # Square grid + center
        idx = 0
        for i in range(5):
            for j in range(5):
                pts[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
        if N_CIRCLES > 25:
            pts[25] = [0.5, 0.5]
    else:  # Random
        pts = np.random.rand(N_CIRCLES, 2)
        
    # Add jitter to break symmetry and help optimization
    pts += np.random.uniform(-0.02, 0.02, pts.shape)
    pts = np.clip(pts, 0.05, 0.95)
    
    # Compute safe initial radii to guarantee strict feasibility
    radii = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        dw = min(pts[i,0], 1.0-pts[i,0], pts[i,1], 1.0-pts[i,1])
        dm = 1.0
        for j in range(N_CIRCLES):
            if i != j:
                d = np.sqrt(np.sum((pts[i]-pts[j])**2))
                if d < dm: 
                    dm = d
        radii[i] = 0.4 * min(dw, 0.5*dm)
        
    x0 = np.zeros(3 * N_CIRCLES)
    for i in range(N_CIRCLES):
        x0[3*i] = pts[i,0]
        x0[3*i+1] = pts[i,1]
        x0[3*i+2] = radii[i]
    return x0

def run_packing():
    """Main optimization loop: global search, perturbation, and precision polish."""
    np.random.seed(42)  # For reproducibility
    N = N_CIRCLES
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N
    cons = {'type': 'ineq', 'fun': compute_constraints}
    
    best_sum = -np.inf
    best_x = None
    
    # Phase 1: Broad search from diverse initializations
    for seed in range(25):
        pattern = seed % 3
        x0 = generate_init_config(seed, pattern)
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
            
            cons_val = compute_constraints(res.x)
            if np.min(cons_val) >= -1e-9:
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_x = res.x.copy()
        except Exception:
            continue
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_x is not None:
        for k in range(35):
            x_pert = best_x + np.random.normal(0, 0.006, 3*N)
            x_pert[::3] = np.clip(x_pert[::3], 0.01, 0.99)
            x_pert[1::3] = np.clip(x_pert[1::3], 0.01, 0.99)
            x_pert[2::3] = np.maximum(x_pert[2::3], 1e-6)
            
            try:
                res = minimize(compute_objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-12, 'disp': False})
                cons_val = compute_constraints(res.x)
                if np.min(cons_val) >= -1e-9:
                    curr_sum = -res.fun
                    if curr_sum > best_sum:
                        best_sum = curr_sum
                        best_x = res.x.copy()
            except Exception:
                continue
                
    # Phase 3: High-precision polish
    if best_x is not None:
        try:
            res_final = minimize(compute_objective, best_x, method='SLSQP', bounds=bounds,
                                 constraints=cons, options={'maxiter': 10000, 'ftol': 1e-14, 'disp': False})
            if np.min(compute_constraints(res_final.x)) >= -1e-9:
                best_x = res_final.x
                best_sum = -res_final.fun
        except Exception:
            pass
            
    # Fallback to a valid grid if optimization somehow fails
    if best_x is None:
        fallback = generate_init_config(0, 1)
        best_x = fallback
        best_sum = np.sum(fallback[2::3])
        
    centers = np.column_stack((best_x[0::3], best_x[1::3]))
    radii = best_x[2::3]
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
