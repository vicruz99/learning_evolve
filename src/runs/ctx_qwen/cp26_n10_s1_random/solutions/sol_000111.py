# sol_000111 | problem=circle_packing_26 entrypoint=run_packing
# generation=3 parent=sol_000034 (state e427cf82) state=5f28d59d sum of radii=1.473829 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

N = 26

def compute_min_clearance(centers_flat):
    """Computes the maximum equal radius feasible for the given centers."""
    centers = centers_flat.reshape(N, 2)
    # Distance to boundaries
    d_b = np.minimum(np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
                     np.minimum(centers[:, 1], 1.0 - centers[:, 1]))
    min_b = np.min(d_b)
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_p = np.min(dists) / 2.0
    
    return min(min_b, min_p)

def neg_clearance(x):
    """Objective for Nelder-Mead: minimize negative clearance."""
    return -compute_min_clearance(x)

def obj_slqp(x_flat):
    """Objective for SLSQP: minimize negative radius."""
    return -x_flat[-1]

def cons_slqp(x_flat):
    """Inequality constraints for SLSQP: all must be >= 0."""
    x = x_flat[:N]
    y = x_flat[N:2*N]
    r = x_flat[-1]
    
    c = []
    # Boundary constraints
    c.extend(x - r)
    c.extend(1.0 - x - r)
    c.extend(y - r)
    c.extend(1.0 - y - r)
    
    # Pairwise separation constraints
    ii, jj = np.triu_indices(N, k=1)
    dx = x[ii] - x[jj]
    dy = y[ii] - y[jj]
    d = np.sqrt(dx**2 + dy**2)
    c.extend(d - 2.0 * r)
    
    return np.array(c)

def run_packing():
    np.random.seed(42)
    best_r = 0.0
    best_centers = None
    
    # 1. Generate diverse initial hexagonal configurations
    configs = []
    r_est = 0.10
    pts = []
    y = r_est
    row = 0
    while len(pts) < N + 5:
        x = r_est
        shift = r_est if row % 2 == 1 else 0.0
        while x + r_est <= 1.0:
            pts.append([x + shift, y])
            x += 2.0 * r_est
        y += np.sqrt(3) * r_est
        row += 1
    base_hex = np.array(pts[:N])
    
    # Normalize to fit comfortably in [0,1]
    min_c = base_hex.min(axis=0)
    max_c = base_hex.max(axis=0)
    scale = 0.90 / (max_c - min_c).max()
    base_hex = (base_hex - min_c) * scale + 0.05
    configs.append(base_hex)
    
    # Add randomized perturbations
    for _ in range(12):
        pert = base_hex + np.random.uniform(-0.025, 0.025, base_hex.shape)
        pert = np.clip(pert, 0.05, 0.95)
        configs.append(pert)
        
    # 2. Optimization Loop
    for cfg in configs:
        # Phase A: Nelder-Mead global search on centers
        res_nm = minimize(neg_clearance, cfg.flatten(),
                          method='Nelder-Mead', 
                          options={'maxiter': 4000, 'xatol': 1e-8, 'fatol': 1e-9})
        centers_nm = res_nm.x.reshape(N, 2)
        
        # Phase B: SLSQP local refinement
        r_init = compute_min_clearance(res_nm.x)
        x0_slqp = np.concatenate([centers_nm.flatten(), [r_init]])
        bounds_slqp = [(0.0, 1.0)] * (2 * N) + [(0.08, 0.12)]
        
        res_slqp = minimize(obj_slqp, x0_slqp, method='SLSQP', bounds=bounds_slqp,
                            constraints={'type': 'ineq', 'fun': cons_slqp},
                            options={'maxiter': 1500, 'ftol': 1e-10})
        
        if res_slqp.x[-1] > best_r:
            best_r = res_slqp.x[-1]
            best_centers = res_slqp.x[:2 * N].reshape(N, 2)
            
    # Fallback if optimization fails
    if best_centers is None:
        best_centers = configs[0]
        
    # 3. LP Phase: Maximize sum of radii for fixed optimal centers
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
            
    num_pairs = len(pairs)
    A_ub = np.zeros((num_pairs + 4 * N, N))
    b_ub = np.zeros(num_pairs + 4 * N)
    idx = 0
    
    for i, j in pairs:
        d = np.linalg.norm(best_centers[i] - best_centers[j])
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        b_ub[idx] = d
        idx += 1
        
    for i in range(N):
        x, y = best_centers[i]
        A_ub[idx, i] = 1.0; b_ub[idx] = x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - x; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = y; idx += 1
        A_ub[idx, i] = 1.0; b_ub[idx] = 1.0 - y; idx += 1
        
    try:
        lp_res = linprog(-np.ones(N), A_ub=A_ub, b_ub=b_ub, bounds=(0, None), method='highs')
        if lp_res.success:
            radii = lp_res.x * 0.9999999
            sum_r = np.sum(radii)
            
            # Strict validity check & fallback shrink if LP pushes too close to limits
            valid = True
            for i in range(N):
                for j in range(i + 1, N):
                    if np.linalg.norm(best_centers[i] - best_centers[j]) < radii[i] + radii[j] - 1e-11:
                        valid = False
                        break
                if not valid:
                    break
                    
            if not valid:
                r_eq = compute_min_clearance(best_centers.flatten()) * 0.9999999
                radii = np.full(N, r_eq)
                sum_r = np.sum(radii)
                
            return best_centers, radii, float(sum_r)
    except Exception:
        pass
        
    # Final fallback
    r_eq = compute_min_clearance(best_centers.flatten()) * 0.9999999
    return best_centers, np.full(N, r_eq), float(N * r_eq)
