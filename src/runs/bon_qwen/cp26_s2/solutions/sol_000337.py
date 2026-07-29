# sol_000337 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state d755ba05) state=d0001488 sum of radii=1.874569 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the positions and radii of 26 circles in a unit square
    to maximize the sum of radii.
    """
    n_circles = 26
    
    # --- Initialization ---
    # We start with a hexagonal-like pattern to encourage high density.
    # A 5x5 grid has 25 circles of r=0.1. 
    # To fit 26, we might need to adjust.
    # Let's try a pattern with rows of varying lengths.
    # Hexagonal packing allows better density.
    
    centers = []
    radii_init = []
    
    # Strategy: 6 rows.
    # Row lengths: 5, 5, 5, 5, 5, 1? No, that's sparse.
    # Let's try to pack them in a rectangular grid first, then perturb.
    # 5x5 grid is 25. Add one in a gap.
    # But equal radii 0.1 doesn't fit 26.
    # Let's start with radius 0.08 to ensure no overlap, then optimize.
    
    # Generate a grid of points
    # We want 26 points.
    # Let's do 5 rows of 5, and 1 row of 1?
    # Or just scatter them in a dense manner.
    
    # Let's try a hexagonal arrangement initialization.
    # Rows alternating 5 and 4?
    # 5, 4, 5, 4, 5, 3 -> 26.
    # This might be tall.
    
    # Let's stick to a slightly perturbed grid for robustness, 
    # as it's easier to converge to a valid state.
    # 5 rows, 5 columns = 25. 
    # Add 26th at a random valid spot or center.
    
    # Grid coordinates for 5x5
    grid_x = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    grid_y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    
    xs, ys = np.meshgrid(grid_x, grid_y)
    points = np.vstack([xs.flatten(), ys.flatten()]).T # 25 points
    
    # Add 26th point in a gap, e.g., (0.2, 0.2)
    points = np.vstack([points, [0.2, 0.2]])
    
    # Initial radii
    r_init = 0.06 * np.ones(n_circles) # Safe starting radius
    
    # Variables: [x1, y1, r1, x2, y2, r2, ...]
    # Size: 26 * 3 = 78
    x0 = np.zeros(n_circles * 3)
    for i in range(n_circles):
        x0[3*i] = points[i, 0]
        x0[3*i+1] = points[i, 1]
        x0[3*i+2] = r_init[i]
        
    # Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n_circles
    
    # --- Objective and Constraints ---
    # We use a penalty method for simplicity and robustness with scipy's L-BFGS-B or SLSQP.
    # SLSQP handles constraints, but penalty method with L-BFGS-B can be faster for many constraints.
    # Let's use L-BFGS-B with a large penalty for violations.
    
    penalty_weight = 10000.0
    
    def objective(vars):
        centers = vars[:n_circles*2].reshape(n_circles, 2)
        radii = vars[n_circles*2:]
        
        obj = -np.sum(radii)
        
        # Boundary constraints
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            # x - r >= 0  => r - x <= 0
            # r - (1-x) <= 0 => x + r <= 1
            # Same for y
            
            # Penalty for x boundaries
            if x < r:
                obj += penalty_weight * (r - x)**2
            if x + r > 1:
                obj += penalty_weight * (x + r - 1)**2
                
            # Penalty for y boundaries
            if y < r:
                obj += penalty_weight * (r - y)**2
            if y + r > 1:
                obj += penalty_weight * (y + r - 1)**2
                
        # Overlap constraints
        # dist >= r_i + r_j
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt((centers[i,0] - centers[j,0])**2 + (centers[i,1] - centers[j,1])**2)
                sum_r = radii[i] + radii[j]
                if dist < sum_r:
                    obj += penalty_weight * (sum_r - dist)**2
                    
        return obj

    # Run optimization
    # We try to minimize the objective (which is negative sum of radii + penalties)
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-12})
    
    # Extract results
    centers_final = res.x[:n_circles*2].reshape(n_circles, 2)
    radii_final = res.x[n_circles*2:]
    
    # Post-processing: Clamp radii and centers to be strictly valid
    # Sometimes optimization leaves tiny violations.
    # We can iteratively shrink radii slightly to ensure validity if needed,
    # but with high penalty it should be very close.
    
    # Check validity and adjust if necessary
    # If any circle is slightly out, pull it in.
    for i in range(n_circles):
        x, y = centers_final[i]
        r = radii_final[i]
        
        # Enforce boundaries strictly
        r = min(r, x, 1-x, y, 1-y)
        # If r becomes 0 or negative, set to 0
        if r < 0: r = 0
        radii_final[i] = r
        
    # Check overlaps and reduce radii if needed
    # This is a simple heuristic to fix overlaps: reduce radii of overlapping circles.
    # To be safe, we can run a few passes of reduction.
    changed = True
    while changed:
        changed = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dist = np.sqrt((centers_final[i,0] - centers_final[j,0])**2 + 
                               (centers_final[i,1] - centers_final[j,1])**2)
                req_dist = radii_final[i] + radii_final[j]
                if dist < req_dist - 1e-12:
                    # Overlap detected. Reduce radii.
                    # Reduce both proportionally or just one.
                    # Simple fix: scale down both radii so they touch.
                    if req_dist > 1e-12:
                        scale = dist / req_dist
                        radii_final[i] *= scale
                        radii_final[j] *= scale
                        changed = True
    
    # Ensure non-negative
    radii_final = np.maximum(radii_final, 0.0)
    
    sum_radii = np.sum(radii_final)
    
    return centers_final, radii_final, sum_radii

# Validation check (optional, but good for debugging)
# if __name__ == "__main__":
#     c, r, s = run_packing()
#     print(f"Sum of radii: {s}")
#     print(validate_packing(c, r))
