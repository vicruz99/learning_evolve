# sol_000314 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5cd869be) state=56bc4841 sum of radii=2.589318 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Global constant for problem size
N_CIRCLES = 26

def objective_func(x):
    """Objective: maximize sum of radii (minimize negative sum)"""
    radii = x[2 * N_CIRCLES:]
    return -np.sum(radii)

def constraint_func(x):
    """
    Returns an array of constraint values.
    All constraints are formulated as fun(x) >= 0.
    """
    n = N_CIRCLES
    centers = x[:2 * n].reshape((n, 2))
    radii = x[2 * n:]
    
    cons = []
    
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx * dx + dy * dy
            r_sum = radii[i] + radii[j]
            cons.append(dist_sq - r_sum * r_sum)
            
    # Boundary constraints: r <= x <= 1-r and r <= y <= 1-r
    for i in range(n):
        cons.append(centers[i, 0] - radii[i])          # x - r >= 0
        cons.append(1.0 - centers[i, 0] - radii[i])    # 1 - x - r >= 0
        cons.append(centers[i, 1] - radii[i])          # y - r >= 0
        cons.append(1.0 - centers[i, 1] - radii[i])    # 1 - y - r >= 0
        
    return np.array(cons)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = N_CIRCLES
    
    # Initial configuration: hexagonal packing layout
    r0 = 0.08
    centers = np.zeros((n, 2))
    radii = np.full(n, r0)
    
    # Arrange circles in rows with hexagonal spacing
    row_counts = [6, 5, 6, 5, 4]  # Sums to 26
    idx = 0
    for i, count in enumerate(row_counts):
        y = r0 + i * r0 * np.sqrt(3)
        for j in range(count):
            x = r0 + j * 2 * r0
            if i % 2 == 1:
                x += r0  # Shift odd rows
            if idx < n:
                centers[idx] = [x, y]
                idx += 1
                
    # Flatten variables: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.hstack([centers.flatten(), radii])
    
    # Add small deterministic perturbation to break symmetry and improve convergence
    np.random.seed(42)
    x0 += np.random.normal(0, 0.0005, size=x0.shape)
    
    # Bounds: x,y in [0,1], r in [1e-5, 0.25]
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(1e-5, 0.25)] * n
    
    # Setup constraints for SLSQP
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run optimization
    res = minimize(
        objective_func,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=cons,
        options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False}
    )
    
    # Extract and format results
    res_centers = res.x[:2 * n].reshape((n, 2))
    res_radii = np.maximum(res.x[2 * n:], 0.0)  # Ensure non-negative radii
    
    sum_radii = float(np.sum(res_radii))
    
    return res_centers, res_radii, sum_radii
