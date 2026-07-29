# sol_000065 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=1e67a1f9 sum of radii=2.627559 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26
i_idx, j_idx = np.triu_indices(N, k=1)

def objective(vars_vec):
    """Objective: maximize sum of radii <=> minimize negative sum."""
    return -np.sum(vars_vec[0::3])

def constraint_func(vars_vec):
    """
    Computes inequality constraints: pairwise non-overlap.
    Boundary constraints are handled analytically via parameterization.
    Returns array where each element must be >= 0.
    """
    r = vars_vec[0::3]
    u = vars_vec[1::3]
    v = vars_vec[2::3]
    
    # Parameterization ensures circles stay within [0,1]^2
    # x = r when u=0, x = 1-r when u=1
    x = r + u * (1.0 - 2.0 * r)
    y = r + v * (1.0 - 2.0 * r)
    
    # Compute squared distances and squared sum of radii efficiently
    dx = x[:, np.newaxis] - x[np.newaxis, :]
    dy = y[:, np.newaxis] - y[np.newaxis, :]
    dist_sq = dx**2 + dy**2
    
    r_sum_sq = (r[:, np.newaxis] + r[np.newaxis, :])**2
    
    # Return constraints for i < j: dist^2 - (r_i + r_j)^2 >= 0
    return dist_sq[i_idx, j_idx] - r_sum_sq[i_idx, j_idx]

def run_packing():
    bounds = [(1e-4, 0.5), (0.0, 1.0), (0.0, 1.0)] * N
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    best_val = -np.inf
    best_sol = None
    
    inits = []
    
    # 1. Hexagonal lattice initializations with varying scales
    for scale in [0.85, 0.95, 1.05, 1.15]:
        pts = []
        r_est = 0.09 * scale
        y = r_est
        row = 0
        while len(pts) < N:
            x = r_est + (row % 2) * r_est
            while x <= 1.0 - r_est and len(pts) < N:
                pts.append([x, y])
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
        pts = np.array(pts[:N])
        r_init = np.full(N, 0.04)
        denom = 1.0 - 2.0 * r_init[0]
        u_init = (pts[:, 0] - r_init[0]) / denom
        v_init = (pts[:, 1] - r_init[0]) / denom
        inits.append(np.concatenate([r_init, u_init, v_init]))
        
    # 2. Grid initializations with varying spacings
    for sp in [0.18, 0.22, 0.28, 0.32]:
        pts = []
        for i in range(6):
            for j in range(5):
                if len(pts) < N:
                    pts.append([0.05 + i*sp, 0.05 + j*sp])
        pts = np.array(pts[:N])
        r_init = np.full(N, 0.04)
        denom = 1.0 - 2.0 * r_init[0]
        u_init = (pts[:, 0] - r_init[0]) / denom
        v_init = (pts[:, 1] - r_init[0]) / denom
        inits.append(np.concatenate([r_init, u_init, v_init]))
        
    # 3. Random feasible initializations
    for seed in range(35):
        np.random.seed(seed)
        r_init = np.random.uniform(0.03, 0.075, N)
        u_init = np.random.uniform(0.0, 1.0, N)
        v_init = np.random.uniform(0.0, 1.0, N)
        inits.append(np.concatenate([r_init, u_init, v_init]))
        
    # Primary optimization phase
    for i, x0 in enumerate(inits):
        # Enforce bounds strictly before optimization
        x0 = np.clip(x0, [1e-4, 0.0, 0.0]*N, [0.5, 1.0, 1.0]*N)
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds,
                           constraints=cons, options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            
            if res.success:
                # Verify constraints are satisfied within numerical tolerance
                c_val = constraint_func(res.x)
                if np.min(c_val) >= -1e-6:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_sol = res.x.copy()
        except Exception:
            continue
            
    # Adaptive perturbation & refinement phase around best solution
    if best_sol is not None:
        for k in range(20):
            # Decay perturbation magnitude to fine-tune
            noise_std = 0.0015 * (1.0 - 0.95 * k)
            x_pert = best_sol.copy() + np.random.randn(len(best_sol)) * noise_std
            x_pert = np.clip(x_pert, [1e-4, 0.0, 0.0]*N, [0.5, 1.0, 1.0]*N)
            
            try:
                res = minimize(objective, x_pert, method='SLSQP', bounds=bounds,
                               constraints=cons, options={'maxiter': 5000, 'ftol': 1e-13, 'disp': False})
                if res.success:
                    c_val = constraint_func(res.x)
                    if np.min(c_val) >= -1e-6:
                        val = -res.fun
                        if val > best_val:
                            best_val = val
                            best_sol = res.x.copy()
            except Exception:
                continue

    # Fallback to a safe valid configuration if optimization fails completely
    if best_sol is None:
        r_f = 0.045
        centers = np.random.rand(N, 2) * 0.8 + 0.1
        centers = np.clip(centers, r_f, 1.0 - r_f)
        best_sol = np.zeros(N * 3)
        best_sol[0::3] = r_f
        best_sol[1::3] = (centers[:, 0] - r_f) / (1.0 - 2.0 * r_f)
        best_sol[2::3] = (centers[:, 1] - r_f) / (1.0 - 2.0 * r_f)
        best_val = np.sum(r_f)
        
    # Reconstruct centers from optimized parameters
    r_opt = best_sol[0::3]
    u_opt = best_sol[1::3]
    v_opt = best_sol[2::3]
    x_opt = r_opt + u_opt * (1.0 - 2.0 * r_opt)
    y_opt = r_opt + v_opt * (1.0 - 2.0 * r_opt)
    centers = np.column_stack((x_opt, y_opt))
    
    # Ensure non-negative radii against numerical drift
    r_opt = np.maximum(r_opt, 0.0)
    
    return centers, r_opt, float(np.sum(r_opt))
