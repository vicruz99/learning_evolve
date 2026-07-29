# sol_000064 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000035 (state cfcb3616) state=39c4bccd sum of radii=2.630369 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N_CIRCLES = 26
I_IDX, J_IDX = np.triu_indices(N_CIRCLES, k=1)

def objective(vars_vec):
    """Objective: maximize sum of radii => minimize negative sum."""
    return -np.sum(vars_vec[2::3])

def constraints(vars_vec):
    """
    Inequality constraints: boundary containment and pairwise non-overlap.
    Returns array of values that must be >= 0.
    """
    cx = vars_vec[0::3]
    cy = vars_vec[1::3]
    r = vars_vec[2::3]
    
    c = []
    # Boundary constraints: r <= x <= 1-r, r <= y <= 1-r
    c.append(cx - r)
    c.append(1.0 - cx - r)
    c.append(cy - r)
    c.append(1.0 - cy - r)
    
    # Pairwise non-overlap: dist^2 >= (r_i + r_j)^2
    dx = cx[:, None] - cx[None, :]
    dy = cy[:, None] - cy[None, :]
    dist2 = dx**2 + dy**2
    
    rs = r[:, None] + r[None, :]
    
    c.append(dist2[I_IDX, J_IDX] - rs[I_IDX, J_IDX]**2)
    return np.concatenate(c)

def generate_init(seed, pattern='hex'):
    """Generates a strictly feasible initial configuration."""
    rng = np.random.RandomState(seed)
    centers = np.zeros((N_CIRCLES, 2))
    
    if pattern == 'hex':
        # Hexagonal lattice initialization
        r_est = 0.09
        y = r_est
        row = 0
        idx = 0
        while idx < N_CIRCLES and y < 1.0 - r_est + 0.01:
            x_start = r_est if row % 2 == 0 else 2.0 * r_est
            x = x_start
            while idx < N_CIRCLES and x < 1.0 - r_est + 0.01:
                centers[idx] = [x, y]
                idx += 1
                x += 2.0 * r_est
            y += np.sqrt(3.0) * r_est
            row += 1
    else:
        # 5x5 grid + center initialization
        idx = 0
        for i in range(5):
            for j in range(5):
                if idx < N_CIRCLES:
                    centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                    idx += 1
        if idx < N_CIRCLES:
            centers[idx] = [0.5, 0.5]
            
    # Add controlled jitter
    centers += rng.uniform(-0.03, 0.03, centers.shape)
    centers = np.clip(centers, 0.05, 0.95)
    
    # Compute strictly feasible initial radii
    r_safe = np.zeros(N_CIRCLES)
    for i in range(N_CIRCLES):
        d_wall = min(centers[i,0], 1.0-centers[i,0], centers[i,1], 1.0-centers[i,1])
        d_min = 1.0
        for j in range(N_CIRCLES):
            if i != j:
                d = np.linalg.norm(centers[i] - centers[j])
                if d < d_min: d_min = d
        # Scale down to guarantee strict feasibility for SLSQP start
        r_safe[i] = 0.45 * min(d_wall, d_min)
        
    x0 = np.zeros(3 * N_CIRCLES)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = r_safe
    return x0

def run_packing():
    # Fixed seed for reproducibility of the refinement phase
    np.random.seed(42)
    
    bounds = [(0.0, 1.0), (0.0, 1.0), (1e-6, 0.5)] * N_CIRCLES
    cons = {'type': 'ineq', 'fun': constraints}
    
    best_val = -np.inf
    best_x = None
    
    # Phase 1: Diverse restarts to explore global landscape
    for s in range(30):
        pat = 'hex' if s < 22 else 'grid'
        x0 = generate_init(s, pat)
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons,
                           options={'maxiter': 4000, 'ftol': 1e-12, 'disp': False})
            if res.success:
                c_val = constraints(res.x)
                if np.min(c_val) >= -1e-8:
                    val = -res.fun
                    if val > best_val:
                        best_val = val
                        best_x = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Local perturbation refinement to escape local minima
    if best_x is not None:
        for _ in range(20):
            x_p = best_x + np.random.randn(3 * N_CIRCLES) * 0.004
            x_p[0::3] = np.clip(x_p[0::3], 0.01, 0.99)
            x_p[1::3] = np.clip(x_p[1::3], 0.01, 0.99)
            x_p[2::3] = np.clip(x_p[2::3], 1e-5, 0.49)
            
            try:
                res = minimize(objective, x_p, method='SLSQP', bounds=bounds, constraints=cons,
                               options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res.success:
                    c_val = constraints(res.x)
                    if np.min(c_val) >= -1e-8:
                        val = -res.fun
                        if val > best_val:
                            best_val = val
                            best_x = res.x.copy()
            except Exception:
                pass
                
        # Phase 3: High-precision polish
        try:
            res_final = minimize(objective, best_x, method='SLSQP', bounds=bounds, constraints=cons,
                                 options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False})
            if res_final.success and np.min(constraints(res_final.x)) >= -1e-9:
                best_x = res_final.x
                best_val = -res_final.fun
        except Exception:
            pass
            
        centers = np.column_stack((best_x[0::3], best_x[1::3]))
        radii = best_x[2::3]
        return centers, radii, float(best_val)
        
    # Fallback (should rarely be reached)
    centers = np.zeros((N_CIRCLES, 2))
    radii = np.full(N_CIRCLES, 0.05)
    idx = 0
    for i in range(5):
        for j in range(5):
            if idx < N_CIRCLES:
                centers[idx] = [0.1 + 0.2*i, 0.1 + 0.2*j]
                idx += 1
    if idx < N_CIRCLES:
        centers[idx] = [0.5, 0.5]
    return centers, radii, float(np.sum(radii))
