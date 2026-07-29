# sol_000325 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ef4a4e64) state=5908b566 sum of radii=2.504352 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization and SLSQP optimization.
    """
    n = 26
    
    # --- 1. Initialization ---
    # Initialize using a hexagonal lattice pattern for high density
    centers = []
    rows = 6
    # Approximate row spacing for hexagonal packing
    y_spacing = 0.18 
    
    for i in range(rows):
        y = 0.1 + i * y_spacing
        # Alternate row lengths to fit 26 circles (5, 4, 5, 4, 5, 3)
        if i % 2 == 0:
            count = 5
            x_start = 0.1
        else:
            count = 4
            x_start = 0.2
            
        for j in range(count):
            if len(centers) < n:
                x = x_start + j * 0.2
                centers.append([x, y])
    
    centers = np.array(centers[:n])
    
    # Initial variable vector: [x1, y1, ..., x26, y26, r]
    x0 = np.concatenate([centers.flatten(), [0.09]])
    
    # --- 2. Optimization Setup ---
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0) for _ in range(2 * n)]  # Bounds for centers
    bounds.append((0.0, 0.5))                     # Bound for radius

    def objective(v):
        # Objective: Minimize negative radius (Maximize r)
        return -v[2 * n]

    def constraints_generator(v):
        r = v[2 * n]
        centers_arr = v[:2 * n].reshape(n, 2)
        cons = []

        # Boundary constraints: r <= x <= 1-r  =>  x-r >= 0, 1-x-r >= 0
        # We group these to minimize callback overhead
        for i in range(n):
            x, y = centers_arr[i]
            # Left
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i, coord=0: v[2*idx+coord] - v[2*n]})
            # Right
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i, coord=0: 1.0 - v[2*idx+coord] - v[2*n]})
            # Bottom
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i, coord=1: v[2*idx+coord] - v[2*n]})
            # Top
            cons.append({'type': 'ineq', 'fun': lambda v, idx=i, coord=1: 1.0 - v[2*idx+coord] - v[2*n]})

        # Distance constraints: dist >= 2r
        # dist^2 >= (2r)^2 => dist^2 - 4r^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda v, i=i, j=j: 
                        (v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2 - 4 * (v[2*n]**2)
                })
        return cons

    constraints = constraints_generator(x0)

    # --- 3. Run Optimizer ---
    try:
        res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                       options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
        final_centers = res.x[:2 * n].reshape(n, 2)
        final_radii = np.full(n, res.x[2 * n])
    except Exception:
        # Fallback to initialization if optimization fails
        final_centers = centers
        final_radii = np.full(n, 0.09)

    # --- 4. Safety Correction ---
    # Iteratively shrink radius slightly to ensure strict validity against 1e-12 tolerance
    valid = False
    while not valid:
        # Check validity manually
        valid = True
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((final_centers[i] - final_centers[j])**2))
                if d < final_radii[i] + final_radii[j] - 1e-12:
                    valid = False
                    # Reduce radius to resolve overlap
                    deficit = (final_radii[i] + final_radii[j] - d) + 1e-6
                    final_radii[i] -= deficit
                    final_radii[j] -= deficit
        
        # Check boundaries
        for i in range(n):
            if final_centers[i][0] < final_radii[i] - 1e-12:
                final_radii[i] = final_centers[i][0] + 1e-12
                valid = False
            if final_centers[i][0] > 1.0 - final_radii[i] + 1e-12:
                final_radii[i] = 1.0 - final_centers[i][0] + 1e-12
                valid = False
            if final_centers[i][1] < final_radii[i] - 1e-12:
                final_radii[i] = final_centers[i][1] + 1e-12
                valid = False
            if final_centers[i][1] > 1.0 - final_radii[i] + 1e-12:
                final_radii[i] = 1.0 - final_centers[i][1] + 1e-12
                valid = False

    return final_centers, final_radii, np.sum(final_radii)
