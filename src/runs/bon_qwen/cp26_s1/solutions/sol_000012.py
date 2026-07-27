# sol_000012 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 60d0e48a) state=121eb756 sum of radii=2.516529 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(z):
    """Objective function: maximize sum of radii (minimize negative sum)."""
    return -np.sum(z[2::3])

def constraint_func(z):
    """Returns inequality constraints >= 0."""
    n = N
    # 4 boundary constraints per circle + 1 per pair
    nc = 4 * n + n * (n - 1) // 2
    c = np.zeros(nc)
    idx = 0
    
    # Boundary constraints: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    for i in range(n):
        x, y, r = z[3 * i], z[3 * i + 1], z[3 * i + 2]
        c[idx] = x - r; idx += 1
        c[idx] = 1.0 - x - r; idx += 1
        c[idx] = y - r; idx += 1
        c[idx] = 1.0 - y - r; idx += 1
        
    # Pairwise non-overlap constraints
    for i in range(n):
        xi, yi, ri = z[3 * i], z[3 * i + 1], z[3 * i + 2]
        for j in range(i + 1, n):
            xj, yj, rj = z[3 * j], z[3 * j + 1], z[3 * j + 2]
            c[idx] = (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
            idx += 1
    return c

def run_packing():
    np.random.seed(42)
    best_sum = -1.0
    best_centers = None
    best_radii = None

    # Variable bounds: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    constraints = {'type': 'ineq', 'fun': constraint_func}

    # Multi-start optimization to escape local minima
    for trial in range(5):
        z0 = np.zeros(3 * N)
        idx = 0
        
        # Base 5x5 grid configuration
        for i in range(5):
            for j in range(5):
                cx = 0.1 + 0.18 * i
                cy = 0.1 + 0.18 * j
                z0[3 * idx] = cx + np.random.uniform(-0.02, 0.02)
                z0[3 * idx + 1] = cy + np.random.uniform(-0.02, 0.02)
                z0[3 * idx + 2] = 0.06
                idx += 1
        
        # 26th circle in the center
        z0[3 * idx] = 0.5 + np.random.uniform(-0.02, 0.02)
        z0[3 * idx + 1] = 0.5 + np.random.uniform(-0.02, 0.02)
        z0[3 * idx + 2] = 0.06
        
        # Ensure initial feasibility with bounds
        z0[:2*N:3] = np.clip(z0[:2*N:3], 0.05, 0.95)
        z0[1:2*N:3] = np.clip(z0[1:2*N:3], 0.05, 0.95)
        
        try:
            res = minimize(objective, z0, method='SLSQP', bounds=bounds, 
                          constraints=constraints, options={'maxiter': 4000, 'ftol': 1e-10})
            
            current_sum = -res.fun
            if current_sum > best_sum:
                # Verify constraints are satisfied (allow tiny numerical slack)
                cons_val = constraint_func(res.x)
                if np.min(cons_val) > -1e-7:
                    best_sum = current_sum
                    z_opt = res.x
                    best_centers = z_opt.reshape(-1, 3)[:, :2]
                    best_radii = z_opt.reshape(-1, 3)[:, 2]
        except Exception:
            continue

    # Fallback if optimization fails
    if best_centers is None:
        best_centers = np.tile([0.5, 0.5], (N, 1))
        best_radii = np.zeros(N)
        best_sum = 0.0
        
    return best_centers, best_radii, best_sum
