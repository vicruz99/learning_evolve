# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6773994b) state=5c627fd6 sum of radii=2.387273 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize

N_CIRCLES = 26

def compute_sum_radii(centers):
    """
    Given fixed centers, solve an LP to find radii that maximize sum(r_i)
    subject to boundary and non-overlap constraints.
    """
    n = centers.shape[0]
    # Objective: maximize sum(r_i) => minimize -sum(r_i)
    c = -np.ones(n)
    bounds = [(0, 0.5)] * n
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        lim = min(x, 1-x, y, 1-y)
        bounds[i] = (0, max(0.0, lim))
        
    # Non-overlap constraints: r_i + r_j <= dist(i, j)
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    if res.success:
        return -res.fun, res.x
    return 0.0, np.zeros(n)

def objective(params):
    """Objective function for optimization: negative sum of radii."""
    centers = params.reshape(N_CIRCLES, 2)
    centers = np.clip(centers, 0.0, 1.0)
    val, _ = compute_sum_radii(centers)
    return -val

def run_packing():
    # Initialize centers in a hexagonal pattern
    centers_init = np.zeros((N_CIRCLES, 2))
    points = []
    row = 0
    while len(points) < N_CIRCLES:
        y = 0.08 + row * 0.18 * np.sqrt(3)/2
        col = 0
        while len(points) < N_CIRCLES:
            x = 0.08 + col * 0.18 + (row % 2) * 0.09
            if x <= 1.0 and y <= 1.0:
                points.append([x, y])
            col += 1
            if col > 12: break
        row += 1
        if row > 12: break
        
    centers_init = np.array(points[:N_CIRCLES])
    
    # Optimize centers to maximize sum of radii
    res = minimize(objective, centers_init.flatten(), method='Nelder-Mead', 
                   options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-7})
    
    best_centers = res.x.reshape(N_CIRCLES, 2)
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Final compute to get consistent radii and sum
    sum_radii, radii = compute_sum_radii(best_centers)
    
    return best_centers, radii, sum_radii
