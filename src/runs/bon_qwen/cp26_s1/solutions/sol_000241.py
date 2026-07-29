# sol_000241 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a213d118) state=215b2684 sum of radii=2.602631 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def get_xy(z):
    """Transform relative coordinates to absolute positions."""
    rs = z[::3]
    us = z[1::3]
    vs = z[2::3]
    xs = rs + (1.0 - 2.0 * rs) * us
    ys = rs + (1.0 - 2.0 * rs) * vs
    return xs, ys, rs

def objective(z):
    """Negative sum of radii (to be minimized)."""
    return -np.sum(z[::3])

def constr_fun(z):
    """Vectorized pairwise non-overlap constraints."""
    xs, ys, rs = get_xy(z)
    # Compute pairwise distances efficiently
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.hypot(dx, dy)
    
    # Extract upper triangle indices (i < j)
    i, j = np.triu_indices(N, 1)
    # Constraint: dist_ij - (r_i + r_j) >= 0
    return dist[i, j] - (rs[i] + rs[j])

def run_packing():
    # Bounds: r in [1e-5, 0.25], u,v in [0, 1]
    bnds = [(1e-5, 0.25), (0.0, 1.0), (0.0, 1.0)] * N
    con = {'type': 'ineq', 'fun': constr_fun}
    
    best_res = None
    best_sum = -1.0
    
    # Multiple restarts to avoid local minima
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        
        # Initialize with a spread-out grid + jitter
        grid_sz = int(np.ceil(np.sqrt(N)))
        indices = rng.choice(grid_sz**2, N, replace=False)
        u_init = (indices % grid_sz) / grid_sz
        v_init = (indices // grid_sz) / grid_sz
        
        # Add controlled randomness
        u_init += rng.uniform(-0.1, 0.1, N)
        v_init += rng.uniform(-0.1, 0.1, N)
        u_init = np.clip(u_init, 0, 1)
        v_init = np.clip(v_init, 0, 1)
        
        r_init = np.full(N, 0.08)  # Reasonable starting radius
        
        z0 = np.empty(3*N)
        z0[::3] = r_init
        z0[1::3] = u_init
        z0[2::3] = v_init
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bnds, constraints=con,
                           options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False})
            if res.fun is not None:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_res = res.x
        except Exception:
            pass
            
    # Fallback if optimizer fails completely
    if best_res is None:
        centers = np.column_stack((0.5*np.ones(N), 0.5*np.ones(N)))
        radii = np.full(N, 0.01)
        return centers, radii, float(N * 0.01)
        
    xs, ys, rs = get_xy(best_res)
    centers = np.column_stack((xs, ys))
    return centers, rs, float(np.sum(rs))
