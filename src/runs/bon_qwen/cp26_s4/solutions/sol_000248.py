# sol_000248 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 240eaf0d) state=72e792d3 sum of radii=2.614283 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Top-level helper functions as required
def compute_objective(vars, n):
    """
    Objective function to maximize sum of radii.
    vars layout: [x_0, ..., x_{n-1}, y_0, ..., y_{n-1}, r_0, ..., r_{n-1}]
    We minimize the negative sum of radii.
    """
    r_start = 2 * n
    return -np.sum(vars[r_start:])

def compute_constraints(vars, n):
    """
    Computes constraint values.
    Returns a vector where all elements must be >= 0.
    Constraints:
    1. Boundary: x >= r, x <= 1-r, y >= r, y <= 1-r
    2. Non-overlap: dist(i, j) >= r_i + r_j
    """
    xs = vars[:n]
    ys = vars[n:2*n]
    rs = vars[2*n:]
    
    # Boundary constraints (4 per circle)
    # x - r >= 0
    # 1 - x - r >= 0
    # y - r >= 0
    # 1 - y - r >= 0
    c_bound = np.concatenate([
        xs - rs,
        1.0 - xs - rs,
        ys - rs,
        1.0 - ys - rs
    ])
    
    # Overlap constraints (1 per unique pair i, j)
    # dist_sq >= (r_i + r_j)^2
    # Using lower triangle indices for unique pairs
    i_idx, j_idx = np.tril_indices(n, k=-1)
    
    dx = xs[i_idx] - xs[j_idx]
    dy = ys[i_idx] - ys[j_idx]
    dist_sq = dx**2 + dy**2
    
    r_sum = rs[i_idx] + rs[j_idx]
    r_sum_sq = r_sum**2
    
    c_overlap = dist_sq - r_sum_sq
    
    return np.concatenate([c_bound, c_overlap])

def run_packing():
    n = 26
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
    
    # Constraints definition
    constraints = [{'type': 'ineq', 'fun': compute_constraints, 'args': (n,)}]
    
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Generate multiple initial configurations to avoid local minima
    configs = []
    
    # Config 1: Perturbed 5x5 Grid with one extra circle
    base_grid = []
    for i in range(5):
        for j in range(5):
            base_grid.append([0.1 + i * 0.2, 0.1 + j * 0.2])
    # Add 26th circle in a gap (e.g., center of a 2x2 block of circles)
    # Gap at (0.4, 0.4) is surrounded by (0.3,0.3), (0.3,0.5), (0.5,0.3), (0.5,0.5)
    base_grid.append([0.4, 0.4])
    
    # Perturb to break symmetry
    config1 = np.array(base_grid[:n]) + np.random.randn(n, 2) * 0.005
    config1 = np.clip(config1, 0.05, 0.95)
    configs.append(config1)
    
    # Config 2: Hexagonal Packing
    hex_pts = []
    r_geom = 0.095 # Radius used for geometric placement
    # Row 0
    for i in range(6):
        x = r_geom + i * 2 * r_geom
        if x + r_geom <= 1.0:
            hex_pts.append([x, r_geom])
    # Row 1 (shifted)
    for i in range(5):
        x = 2 * r_geom + i * 2 * r_geom
        if x + r_geom <= 1.0:
            hex_pts.append([x, r_geom + np.sqrt(3)*r_geom])
    # Row 2
    for i in range(6):
        x = r_geom + i * 2 * r_geom
        if x + r_geom <= 1.0:
            hex_pts.append([x, 2 * r_geom + 2 * np.sqrt(3)*r_geom])
    # Row 3
    for i in range(5):
        x = 2 * r_geom + i * 2 * r_geom
        if x + r_geom <= 1.0:
            hex_pts.append([x, 3 * r_geom + 3 * np.sqrt(3)*r_geom])
    # Row 4
    for i in range(5):
        x = r_geom + i * 2 * r_geom
        if x + r_geom <= 1.0:
            hex_pts.append([x, 4 * r_geom + 4 * np.sqrt(3)*r_geom])
    
    hex_arr = np.array(hex_pts)
    # Keep valid points
    mask = (hex_arr[:, 0] >= 0) & (hex_arr[:, 0] <= 1) & (hex_arr[:, 1] >= 0) & (hex_arr[:, 1] <= 1)
    hex_arr = hex_arr[mask]
    
    # Pad if necessary
    while len(hex_arr) < n:
        hex_arr = np.vstack([hex_arr, [np.random.rand(), np.random.rand()]])
    hex_arr = hex_arr[:n]
    # Perturb
    config2 = hex_arr + np.random.randn(n, 2) * 0.005
    config2 = np.clip(config2, 0.05, 0.95)
    configs.append(config2)
    
    # Config 3: Random initialization
    config3 = np.random.rand(n, 2) * 0.8 + 0.1
    configs.append(config3)
    
    # Run optimization for each config
    for centers in configs:
        # Start with small radii to ensure feasibility
        radii = np.full(n, 0.05)
        x0 = np.hstack([centers[:, 0], centers[:, 1], radii])
        
        try:
            res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds,
                           constraints=constraints, args=(n,),
                           options={'maxiter': 2000, 'ftol': 1e-9, 'disp': False})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = np.column_stack((res.x[:n], res.x[n:2*n]))
                    best_radii = res.x[2*n:]
        except Exception:
            pass
    
    # Fallback if optimization failed
    if best_centers is None:
        # Simple grid packing
        x_coords = np.linspace(0.1, 0.9, 5)
        y_coords = np.linspace(0.1, 0.9, 5)
        centers_fallback = np.array([(x, y) for x in x_coords for y in y_coords])
        centers_fallback = centers_fallback[:n]
        radii_fallback = np.full(n, 0.05)
        return centers_fallback, radii_fallback, 0.05 * n
        
    return best_centers, best_radii, best_sum
