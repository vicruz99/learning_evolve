# sol_000019 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0f0997f0) state=06ce35b6 sum of radii=2.575747 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialize with a dense 5x5 grid (25 circles) + 1 circle in a gap
    grid_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    grid_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    centers = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
    # Add the 26th circle in a gap between grid points
    centers = np.vstack([centers, [0.2, 0.2]])
    
    # Start with a small feasible radius
    radii = np.full(n, 0.01)
    
    # Flatten to optimization vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
    
    # 2. Define Constraints
    def constraints(vars):
        xs = vars[0::3]
        ys = vars[1::3]
        rs = vars[2::3]
        
        # Boundary constraints (4 per circle)
        # x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
        c_box = np.concatenate([xs - rs, 1.0 - xs - rs, ys - rs, 1.0 - ys - rs])
        
        # Overlap constraints: dist^2 - (r_i + r_j)^2 >= 0
        # Vectorized for all pairs i < j
        xs_col = xs[:, None]
        ys_col = ys[:, None]
        rs_col = rs[:, None]
        
        dx = xs_col - xs
        dy = ys_col - ys
        dist_sq = dx**2 + dy**2
        sum_r_sq = (rs_col + rs)**2
        
        # Upper triangular indices (i < j)
        triu_idx = np.triu_indices(n, k=1)
        c_overlap = dist_sq[triu_idx] - sum_r_sq[triu_idx]
        
        return np.concatenate([c_box, c_overlap])

    # 3. Objective Function (Maximize sum of radii)
    def objective(vars):
        return -np.sum(vars[2::3])

    # 4. Bounds (x, y in [0, 1], r in [0, 0.5])
    bounds = [(0.0, 1.0) for _ in range(3 * n)]
    for i in range(2, 3 * n, 3):
        bounds[i] = (0.0, 0.5)

    # 5. Optimization
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints={'type': 'ineq', 'fun': constraints},
        options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False}
    )
    
    # 6. Extract and validate results
    final_centers = np.column_stack((res.x[0::3], res.x[1::3]))
    final_radii = res.x[2::3]
    
    return final_centers, final_radii, np.sum(final_radii)
