# sol_000235 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1140c965) state=90842419 sum of radii=2.539890 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

N = 26

def objective(th):
    """Objective function to maximize sum of radii (minimized as negative)."""
    return -np.sum(th[2::3])

def compute_constraints(th):
    """Computes boundary and pairwise non-overlap constraints."""
    vals = []
    x = th[0::3]
    y = th[1::3]
    r = th[2::3]
    
    # Boundary constraints: circles must be inside [0,1]x[0,1]
    # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    vals.append(x - r)
    vals.append(1.0 - x - r)
    vals.append(y - r)
    vals.append(1.0 - y - r)
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(N):
        xi, yi, ri = x[i], y[i], r[i]
        for j in range(i + 1, N):
            dist_sq = (xi - x[j])**2 + (yi - y[j])**2
            rad_sum_sq = (ri + r[j])**2
            vals.append(np.array([dist_sq - rad_sum_sq]))
            
    return np.concatenate(vals)

def constraint_fun(th):
    """Wrapper for constraint function to be used by scipy optimizer."""
    return compute_constraints(th)

def run_packing():
    # Variable bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(0.0, 0.5)] * N
    cons = {'type': 'ineq', 'fun': constraint_fun}
    
    best_sol = None
    best_obj = -np.inf
    
    strategies = []
    
    # Strategy 1: Multiple random starts to explore configuration space
    for seed in range(5):
        rng = np.random.RandomState(seed)
        x0 = rng.rand(N) * 0.6 + 0.2
        y0 = rng.rand(N) * 0.6 + 0.2
        r0 = np.full(N, 0.02)
        theta0 = np.zeros(3 * N)
        theta0[0::3] = x0
        theta0[1::3] = y0
        theta0[2::3] = r0
        strategies.append(theta0)
        
    # Strategy 2: Regular grid start (approximates uniform packing)
    pts = np.array([[i/4.5 + 0.1, j/4.5 + 0.1] for i in range(5) for j in range(5)])
    x_grid = np.append(pts[:, 0], 0.5)
    y_grid = np.append(pts[:, 1], 0.5)
    r_grid = np.full(N, 0.03)
    theta_grid = np.zeros(3 * N)
    theta_grid[0::3] = x_grid
    theta_grid[1::3] = y_grid
    theta_grid[2::3] = r_grid
    strategies.append(theta_grid)
    
    # Strategy 3: Hexagonal-like start (denser packing pattern)
    hex_x, hex_y = [], []
    for row in range(6):
        y_val = 0.1 + row * 0.16
        n_in_row = 5 if row % 2 == 0 else 4
        for col in range(n_in_row):
            if len(hex_x) < N:
                x_val = 0.1 + col * 0.18 + (0.09 if row % 2 != 0 else 0.0)
                hex_x.append(x_val)
                hex_y.append(y_val)
    while len(hex_x) < N:
        hex_x.append(0.5 + (len(hex_x) % 2) * 0.1)
        hex_y.append(0.5)
    theta_hex = np.zeros(3 * N)
    theta_hex[0::3] = np.array(hex_x[:N])
    theta_hex[1::3] = np.array(hex_y[:N])
    theta_hex[2::3] = np.full(N, 0.03)
    strategies.append(theta_hex)
    
    # Optimize from each starting configuration
    for theta0 in strategies:
        res = minimize(objective, theta0, method='SLSQP', bounds=bounds, 
                       constraints=cons, options={'maxiter': 1500, 'ftol': 1e-10, 'disp': False})
        
        if res.success:
            curr_obj = -res.fun
            # Check constraint satisfaction with tolerance
            cons_vals = constraint_fun(res.x)
            if np.min(cons_vals) > -1e-8 and curr_obj > best_obj:
                best_obj = curr_obj
                best_sol = res.x
                
                # Refine best solution with tighter tolerances
                res2 = minimize(objective, best_sol, method='SLSQP', bounds=bounds,
                                constraints=cons, options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
                if res2.success:
                    cons_vals2 = constraint_fun(res2.x)
                    if np.min(cons_vals2) > -1e-8 and -res2.fun > best_obj:
                        best_obj = -res2.fun
                        best_sol = res2.x
                        
    # Fallback if all optimizations fail (should not happen with feasible starts)
    if best_sol is None:
        best_sol = res.x
        best_obj = -res.fun
        
    centers = np.column_stack((best_sol[0::3], best_sol[1::3]))
    radii = best_sol[2::3]
    return centers, radii, float(np.sum(radii))
