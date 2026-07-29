# sol_000276 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9068c8d6) state=23764963 sum of radii=2.620761 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generates and optimizes a packing of 26 circles in a unit square to maximize sum of radii.
    """
    N = 26
    
    # --- 1. Heuristic Initialization (Hexagonal Pattern) ---
    centers = []
    # We aim for a 6-row hexagonal layout to accommodate 26 circles efficiently.
    # Row structure (lengths): 5, 4, 5, 4, 5, 3 = 26 circles
    
    r_init = 0.10 # Initial guess for radius to scale the grid
    v_spacing = np.sqrt(3) * r_init
    
    # Row configurations: (row_index, num_circles, is_shifted)
    row_configs = [
        (0, 5, False),
        (1, 4, True),
        (2, 5, False),
        (3, 4, True),
        (4, 5, False),
        (5, 3, True)
    ]
    
    for row_idx, count, shifted in row_configs:
        # Y coordinate for the row
        y = r_init + row_idx * v_spacing
        
        # X coordinates
        # Unshifted rows: centers at r, 3r, 5r...
        # Shifted rows: centers at 2r, 4r, 6r...
        start_x = r_init if not shifted else 2 * r_init
        
        # Generate x positions for this row
        x_coords = [start_x + i * (2 * r_init) for i in range(count)]
        
        for x in x_coords:
            centers.append([x, y])
            
    centers = np.array(centers)
    radii = np.full(N, r_init)
    
    # --- 2. Numerical Optimization ---
    # Optimize variables: [x1, y1, x2, y2, ..., x26, y26, r1, r2, ..., r26]
    # Total variables: 26 * 3 = 78
    
    # Flatten initial state
    x0 = np.concatenate([centers.flatten(), radii])
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0, 1)] * (2 * N) + [(0, 0.5)] * N

    def objective(p):
        # Variables: p[0..51] are x,y coords, p[52..77] are radii
        c = p[:2 * N].reshape(N, 2)
        r = p[2 * N:]
        
        # Objective: Maximize sum of radii -> Minimize negative sum
        return -np.sum(r)

    def constraint_overlap(p):
        c = p[:2 * N].reshape(N, 2)
        r = p[2 * N:]
        
        constraints = []
        for i in range(N):
            for j in range(i + 1, N):
                # Distance between centers
                dist = np.linalg.norm(c[i] - c[j])
                # Constraint: dist >= r_i + r_j  =>  dist - (r_i + r_j) >= 0
                constraints.append(dist - (r[i] + r[j]))
        return np.array(constraints)

    def constraint_boundary(p):
        c = p[:2 * N].reshape(N, 2)
        r = p[2 * N:]
        
        constraints = []
        for i in range(N):
            x, y = c[i]
            rad = r[i]
            # Circle must be inside [0,1]x[0,1]
            # x - r >= 0
            constraints.append(x - rad)
            # 1 - (x + r) >= 0  =>  1 - x - r >= 0
            constraints.append(1 - x - rad)
            # y - r >= 0
            constraints.append(y - rad)
            # 1 - (y + r) >= 0  =>  1 - y - r >= 0
            constraints.append(1 - y - rad)
        return np.array(constraints)

    # Define constraints for scipy
    # SLSQP requires constraints to be dictionaries or NonlinearConstraint
    # Here we use the older constraint format for clarity: {'type': 'ineq', 'fun': func}
    
    cons = [
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_boundary}
    ]

    # Run optimization
    # maxiter can be increased if needed, but 1000 is usually sufficient for this size
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                      options={'maxiter': 2000, 'ftol': 1e-12})
    
    # Extract optimized parameters
    opt_centers = result.x[:2 * N].reshape(N, 2)
    opt_radii = result.x[2 * N:]
    
    return opt_centers, opt_radii, np.sum(opt_radii)
