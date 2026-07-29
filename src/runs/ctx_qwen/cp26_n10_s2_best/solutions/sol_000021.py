# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1b9ac6cc) state=360d2b95 sum of radii=2.553523 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    """
    n = 26
    
    # 1. Initialization: Hexagonal-like grid
    # We create a grid of points and select 26.
    # A 6x5 grid has 30 points. We will take the first 26.
    # This provides a dense, non-overlapping initial state.
    
    # Spacing for initial grid to allow radius ~0.05
    # 5 rows, 6 columns
    # x range [0, 1], y range [0, 1]
    
    xs = np.linspace(0.15, 0.85, 6) # 6 points
    ys = np.linspace(0.1, 0.9, 5)  # 5 points
    
    centers = []
    # Interleave rows to create hexagonal offset effect roughly
    # Row 0: even indices (0, 2, 4) -> 3 circles?
    # Let's just fill row by row from a 6x5 grid.
    count = 0
    for y in ys:
        for x in xs:
            if count < n:
                centers.append([x, y])
                count += 1
    
    centers = np.array(centers)
    initial_radii = np.ones(n) * 0.05
    
    # Flatten to 1D array: [x1...x26, y1...y26, r1...r26]
    x0 = np.concatenate([centers[:, 0], centers[:, 1], initial_radii])
    
    # 2. Define Objective and Constraints
    def objective(vars_vec):
        radii = vars_vec[52:]
        return -np.sum(radii) # Minimize negative sum
    
    def constraint_boundaries(vars_vec):
        # Returns an array of values that must be >= 0
        xs = vars_vec[0:26]
        ys = vars_vec[26:52]
        rs = vars_vec[52:]
        
        # x - r >= 0
        c1 = xs - rs
        # 1 - x - r >= 0
        c2 = 1 - xs - rs
        # y - r >= 0
        c3 = ys - rs
        # 1 - y - r >= 0
        c4 = 1 - ys - rs
        # r >= 0
        c5 = rs
        
        return np.concatenate([c1, c2, c3, c4, c5])
    
    def constraint_no_overlap(vars_vec):
        # Returns array of (dist^2 - (r_i + r_j)^2) >= 0
        xs = vars_vec[0:26]
        ys = vars_vec[26:52]
        rs = vars_vec[52:]
        
        constraints = []
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = (xs[i] - xs[j])**2 + (ys[i] - ys[j])**2
                sum_r = rs[i] + rs[j]
                # Constraint: dist_sq - sum_r^2 >= 0
                constraints.append(dist_sq - sum_r**2)
        return np.array(constraints)

    # 3. Setup Constraints for SLSQP
    # Boundary constraints
    bound_const = {
        'type': 'ineq',
        'fun': constraint_boundaries
    }
    
    # Overlap constraints
    overlap_const = {
        'type': 'ineq',
        'fun': constraint_no_overlap
    }
    
    cons = [bound_const, overlap_const]
    
    # Bounds for variables (x, y in [0,1], r in [0, 1])
    bounds = [(0, 1)] * 52 + [(0, 1)] * 26
    
    # 4. Run Optimization
    # Use SLSQP method
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-9})
    
    # 5. Extract Results
    final_vars = res.x
    final_centers = np.column_stack((final_vars[0:26], final_vars[26:52]))
    final_radii = final_vars[52:]
    
    # 6. Post-processing / Validation fix
    # Sometimes optimizers drift slightly outside boundaries due to float errors.
    # Clamp centers and reduce radii if necessary to be strictly valid.
    # This is a safety step.
    
    # Clamp centers
    final_centers[:, 0] = np.clip(final_centers[:, 0], 0, 1)
    final_centers[:, 1] = np.clip(final_centers[:, 1], 0, 1)
    
    # Adjust radii to be within boundaries
    # r <= x, r <= 1-x, r <= y, r <= 1-y
    margin_x = np.minimum(final_centers[:, 0], 1 - final_centers[:, 0])
    margin_y = np.minimum(final_centers[:, 1], 1 - final_centers[:, 1])
    max_r_by_boundary = np.minimum(margin_x, margin_y)
    
    # Also ensure non-overlap strictly
    # We can iteratively reduce radii if overlaps exist, though optimizer should have handled it.
    # A simple pass:
    for _ in range(10): # Few iterations to settle
        for i in range(n):
            r_current = final_radii[i]
            if r_current > max_r_by_boundary[i]:
                final_radii[i] = max_r_by_boundary[i]
                r_current = final_radii[i]
            
            for j in range(n):
                if i == j: continue
                dist = np.sqrt((final_centers[i, 0] - final_centers[j, 0])**2 + 
                               (final_centers[i, 1] - final_centers[j, 1])**2)
                allowed_r = dist - final_radii[j]
                if r_current > allowed_r:
                    final_radii[i] = max(0, allowed_r)
                    r_current = final_radii[i]

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper to verify locally if needed (not required by prompt but good practice)
# validate_packing is provided in the prompt context.

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(validate_packing(c, r))
