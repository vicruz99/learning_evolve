# sol_000345 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 03d022f0) state=ab64d684 sum of radii=2.557122 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    
    # 1. Generate Initial Configuration
    # We start with a hexagonal-like arrangement to provide a good starting point.
    # A simple grid is often a good fallback if hexagonal logic is complex to parameterize perfectly for N=26.
    # Let's try to distribute points as evenly as possible.
    
    centers = []
    
    # Attempt a hexagonal lattice pattern
    # Approximate spacing for 26 circles
    # Area per circle ~ 1/26 ~ 0.038. Hex area ~ sqrt(3)/2 * s^2. 
    # s ~ 0.21. Radius ~ 0.1.
    
    # Let's generate a grid of points and then filter/select to get 26 points
    # Or just construct rows.
    # Rows offset by half spacing.
    
    # Let's try 6 rows.
    # Row heights ~ 1/6?
    # Actually, let's just use a dense grid and pick 26 points that are well spaced.
    # Or simply initialize random valid points? No, structured is better.
    
    # Heuristic: Generate points on a lattice and keep first 26.
    # Lattice spacing.
    # If we have 5 columns and 6 rows, that's 30 points.
    # We can remove 4.
    
    # Let's create a set of candidate points on a grid
    # Grid size roughly 6x5 or 7x4
    cols = 6
    rows = 5
    # We want 26 points. 6*5 = 30. We can remove 4.
    # Or maybe 7x4 = 28.
    
    # Let's try to generate points with some spacing
    # x coordinates: linspace(0.1, 0.9, 6) -> 6 points
    # y coordinates: linspace(0.1, 0.9, 5) -> 5 points
    # This gives 30 points in [0.1, 0.9] x [0.1, 0.9].
    # Radius 0.1 would fit.
    
    # Let's create a list of candidate centers
    candidates = []
    x_vals = np.linspace(0.1, 0.9, 6)
    y_vals = np.linspace(0.1, 0.9, 5)
    
    for y in y_vals:
        for x in x_vals:
            candidates.append([x, y])
            
    # We need 26. We have 30.
    # Remove 4 points. Which ones?
    # Maybe the corners or edges? 
    # Actually, let's just take the first 26.
    # But to maximize sum of radii, we want them spread out.
    # A subset of a grid is fine.
    
    initial_centers = np.array(candidates[:n_circles])
    
    # Initial radii: small enough to not overlap, e.g., 0.05
    initial_radii = np.full(n_circles, 0.05)
    
    # Combine into variable vector: [x0, y0, r0, x1, y1, r1, ...]
    x0 = np.zeros(n_circles * 3)
    for i in range(n_circles):
        x0[i*3] = initial_centers[i, 0]
        x0[i*3 + 1] = initial_centers[i, 1]
        x0[i*3 + 2] = initial_radii[i]
        
    # 2. Define Objective Function
    def objective(vars_vec):
        # Sum of radii is sum of every 3rd element starting from index 2
        radii = vars_vec[2::3]
        return -np.sum(radii) # Minimize negative sum
        
    # 3. Define Constraints
    # Constraints are functions returning value >= 0
    
    constraints = []
    
    # Boundary constraints:
    # r_i <= x_i  => x_i - r_i >= 0
    # r_i <= 1 - x_i => 1 - x_i - r_i >= 0
    # r_i <= y_i => y_i - r_i >= 0
    # r_i <= 1 - y_i => 1 - y_i - r_i >= 0
    
    for i in range(n_circles):
        idx = i * 3
        xi = idx
        yi = idx + 1
        ri = idx + 2
        
        # x >= r
        def make_con_x_min(i):
            def con(v):
                return v[i*3] - v[i*3 + 2]
            return con
        
        # x <= 1 - r  => 1 - x - r >= 0
        def make_con_x_max(i):
            def con(v):
                return 1.0 - v[i*3] - v[i*3 + 2]
            return con
            
        # y >= r
        def make_con_y_min(i):
            def con(v):
                return v[i*3 + 1] - v[i*3 + 2]
            return con
            
        # y <= 1 - r
        def make_con_y_max(i):
            def con(v):
                return 1.0 - v[i*3 + 1] - v[i*3 + 2]
            return con
            
        constraints.append({'type': 'ineq', 'fun': make_con_x_min(i)})
        constraints.append({'type': 'ineq', 'fun': make_con_x_max(i)})
        constraints.append({'type': 'ineq', 'fun': make_con_y_min(i)})
        constraints.append({'type': 'ineq', 'fun': make_con_y_max(i)})
        
        # r >= 0
        def make_con_r_min(i):
            def con(v):
                return v[i*3 + 2]
            return con
        constraints.append({'type': 'ineq', 'fun': make_con_r_min(i)})

    # Non-overlap constraints:
    # dist(c_i, c_j) >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    
    # Note: SLSQP works with gradients. Squared terms are smooth.
    # However, for numerical stability, we might want to avoid squaring if numbers get large,
    # but here everything is in [0,1].
    
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            idx_i = i * 3
            idx_j = j * 3
            
            def make_overlap_con(i, j):
                def con(v):
                    xi, yi, ri = v[i*3], v[i*3 + 1], v[i*3 + 2]
                    xj, yj, rj = v[j*3], v[j*3 + 1], v[j*3 + 2]
                    
                    dist_sq = (xi - xj)**2 + (yi - yj)**2
                    r_sum_sq = (ri + rj)**2
                    
                    return dist_sq - r_sum_sq
                return con
            
            constraints.append({'type': 'ineq', 'fun': make_overlap_con(i, j)})

    # 4. Bounds
    # x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (radius can't be > 0.5)

    # 5. Optimization
    # Using SLSQP
    # It might be sensitive to initial guess. Our grid is valid (r=0.05, dist=0.2).
    # dist > 2r (0.2 > 0.1), so valid.
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
    
    # Extract results
    final_centers = np.zeros((n_circles, 2))
    final_radii = np.zeros(n_circles)
    
    for i in range(n_circles):
        idx = i * 3
        final_centers[i, 0] = result.x[idx]
        final_centers[i, 1] = result.x[idx + 1]
        final_radii[i] = result.x[idx + 2]
        
    sum_radii = np.sum(final_radii)
    
    # Post-processing check: ensure validity
    # If optimizer failed to satisfy constraints strictly due to tolerance,
    # we might need to shrink radii slightly. But SLSQP with tight tol should be fine.
    # However, just to be safe, we can verify and clip if necessary, 
    # but the problem asks to return the result of the function.
    
    return final_centers, final_radii, sum_radii

# Helper to run and print if executed directly, but function is required.
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
