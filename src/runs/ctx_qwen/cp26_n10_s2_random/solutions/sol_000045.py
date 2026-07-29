# sol_000045 | problem=circle_packing_26 entrypoint=run_packing
# generation=2 parent=sol_000044 (state 69bc282d) state=1a011e81 sum of radii=2.611001 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

N = 26

def objective(p):
    """Objective: minimize negative sum of radii (maximize sum of radii)."""
    return -np.sum(p[2::3])

def constraint_fun(p):
    """
    Computes inequality constraints. All returned values must be >= 0 for feasibility.
    1. Boundary: x >= r, 1-x >= r, y >= r, 1-y >= r
    2. Non-overlap: dist(i,j) >= r_i + r_j
    """
    x = p[0::3]
    y = p[1::3]
    r = p[2::3]
    
    c = np.empty(4 * N + N * (N - 1) // 2)
    c[0:N] = x - r
    c[N:2*N] = 1.0 - x - r
    c[2*N:3*N] = y - r
    c[3*N:4*N] = 1.0 - y - r
    
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    sum_r = r[:, None] + r[None, :]
    
    mask = np.triu_indices(N, k=1)
    c[4*N:] = dist[mask] - sum_r[mask]
    return c

def get_bounds():
    """Variable bounds: x,y in [0,1], r in [0, 0.5]."""
    b = []
    for _ in range(N):
        b.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])
    return b

def make_hex_config(row_counts, r_init, noise=0.0):
    """Generates initial parameters using a staggered hexagonal lattice pattern."""
    p = np.zeros(N * 3)
    idx = 0
    y = r_init
    for i, count in enumerate(row_counts):
        x_start = r_init if i % 2 == 0 else 2 * r_init
        x = x_start
        for _ in range(count):
            p[3 * idx] = x
            p[3 * idx + 1] = y
            p[3 * idx + 2] = r_init
            idx += 1
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        
    if noise > 0:
        rng = np.random.default_rng(42)
        p[0::3] += rng.normal(0, noise, N)
        p[1::3] += rng.normal(0, noise, N)
        p[0::3] = np.clip(p[0::3], 1e-4, 1 - 1e-4)
        p[1::3] = np.clip(p[1::3], 1e-4, 1 - 1e-4)
    return p

def run_packing():
    """Main optimization routine to pack 26 circles in a unit square."""
    np.random.seed(42)
    bnds = get_bounds()
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    best_p = None
    best_sum = -1.0
    
    # Diverse row configurations that sum to 26
    row_configs = [
        [6, 5, 6, 5, 4],
        [5, 6, 5, 6, 4],
        [5, 5, 6, 5, 5],
        [4, 6, 6, 6, 4],
        [5, 5, 5, 5, 6],
        [6, 4, 6, 4, 6]
    ]
    
    initial_p0_list = []
    for rc in row_configs:
        # Vary initial radius and perturbation to explore configuration space
        initial_p0_list.append(make_hex_config(rc, 0.092, 0.0))
        initial_p0_list.append(make_hex_config(rc, 0.092, 0.005))
        initial_p0_list.append(make_hex_config(rc, 0.088, 0.01))
        initial_p0_list.append(make_hex_config(rc, 0.085, 0.02))
        
    # Phase 1: Multi-start local optimization
    for p0 in initial_p0_list:
        try:
            res = opt.minimize(objective, p0, method='SLSQP', bounds=bnds, constraints=cons,
                               options={'maxiter': 8000, 'ftol': 1e-14, 'disp': False})
            curr_sum = -res.fun
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_p = res.x.copy()
        except Exception:
            pass
            
    # Phase 2: Iterative Expansion
    # Gradually scale up radii and re-optimize centers to push the packing to its limit
    if best_p is not None:
        p_curr = best_p.copy()
        for _ in range(40):
            p_curr[2::3] *= 1.002
            try:
                res = opt.minimize(objective, p_curr, method='SLSQP', bounds=bnds, constraints=cons,
                                   options={'maxiter': 4000, 'ftol': 1e-14, 'disp': False})
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_p = res.x.copy()
                    p_curr = best_p.copy()
                else:
                    break
            except Exception:
                break
                
    # Phase 3: Local escape from potential flat minima
    if best_p is not None:
        for seed in range(10):
            p_pert = best_p.copy()
            rng = np.random.default_rng(seed * 13)
            p_pert[0::3] += rng.normal(0, 0.003, N)
            p_pert[1::3] += rng.normal(0, 0.003, N)
            p_pert[0::3] = np.clip(p_pert[0::3], 1e-4, 1 - 1e-4)
            p_pert[1::3] = np.clip(p_pert[1::3], 1e-4, 1 - 1e-4)
            try:
                res = opt.minimize(objective, p_pert, method='SLSQP', bounds=bnds, constraints=cons,
                                   options={'maxiter': 3000, 'ftol': 1e-14, 'disp': False})
                curr_sum = -res.fun
                if curr_sum > best_sum:
                    best_sum = curr_sum
                    best_p = res.x.copy()
            except Exception:
                pass
                
    # Fallback initialization if optimization completely fails
    if best_p is None:
        best_p = make_hex_config([6, 5, 6, 5, 4], 0.085, 0.0)
        
    # Extract and validate
    centers = np.column_stack((best_p[0::3], best_p[1::3]))
    radii = best_p[2::3].copy()
    
    # Ensure strict feasibility for the validator (allow tiny slack for float precision)
    radii *= 0.99999
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
