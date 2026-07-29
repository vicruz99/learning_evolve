# sol_000342 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6a87b209) state=861cf855 sum of radii=0.000325 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    
    # Helper function to compute distance matrix
    def get_dist_matrix(cx, cy):
        # cx, cy are 1D arrays of length n
        # Returns 2D array of distances
        # Using broadcasting for efficiency
        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        dists = np.sqrt(dx**2 + dy**2)
        return dists

    # Penalty weight for constraints
    PENALTY_WEIGHT = 10000.0

    def objective(params):
        # params shape: (3 * n,)
        # Order: x_coords, y_coords, radii
        
        cx = params[:n]
        cy = params[n:2*n]
        r = params[2*n:]
        
        # Primary objective: maximize sum of radii -> minimize negative sum
        obj_val = -np.sum(r)
        
        # 1. Boundary Constraints
        # Circle i must be inside [0,1]x[0,1]
        # x - r >= 0  => r - x <= 0
        # x + r <= 1  => x + r - 1 <= 0
        # y - r >= 0  => r - y <= 0
        # y + r <= 1  => y + r - 1 <= 0
        
        # Left boundary violation: max(0, r - x)
        viol_left = np.maximum(0, r - cx)
        obj_val += PENALTY_WEIGHT * np.sum(viol_left**2)
        
        # Right boundary violation: max(0, r - (1 - x))
        viol_right = np.maximum(0, r - (1.0 - cx))
        obj_val += PENALTY_WEIGHT * np.sum(viol_right**2)
        
        # Bottom boundary violation: max(0, r - y)
        viol_bottom = np.maximum(0, r - cy)
        obj_val += PENALTY_WEIGHT * np.sum(viol_bottom**2)
        
        # Top boundary violation: max(0, r - (1 - y))
        viol_top = np.maximum(0, r - (1.0 - cy))
        obj_val += PENALTY_WEIGHT * np.sum(viol_top**2)
        
        # 2. Overlap Constraints
        # dist(i, j) >= r[i] + r[j]
        # Violation: max(0, r[i] + r[j] - dist(i, j))
        
        # Compute pairwise distances
        # Since n=26 is small, we can do a double loop or vectorized ops
        # Vectorized approach:
        # diff_x shape (n, n)
        dx = cx[:, np.newaxis] - cx[np.newaxis, :]
        dy = cy[:, np.newaxis] - cy[np.newaxis, :]
        dists = np.sqrt(dx**2 + dy**2)
        
        # Matrix of sum of radii
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Violation matrix
        # We only care about i < j, but computing full matrix is fine, diagonal is 0
        viol_overlap = np.maximum(0, r_sum - dists)
        
        # Sum of squared violations (upper triangle to avoid double counting, though factor 2 doesn't matter for gradient direction much)
        # Actually, summing all entries counts each pair twice, which is fine as a scalar penalty
        obj_val += PENALTY_WEIGHT * np.sum(viol_overlap**2)
        
        return obj_val

    # Initial guess
    # Start with a 5x5 grid (25 circles) + 1 circle in a gap
    # Grid coordinates
    grid_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    x_grid, y_grid = np.meshgrid(grid_coords, grid_coords)
    
    # Flatten grid centers
    centers_grid = np.column_stack((x_grid.ravel(), y_grid.ravel()))
    
    # Add 26th circle in a gap
    # Gap at (0.2, 0.2) is good. Distance to nearest grid circles (0.1, 0.1) etc is ~0.1414.
    # Max radius there is ~0.0414.
    center_26 = np.array([0.2, 0.2])
    
    # Combine centers
    centers_init = np.vstack([centers_grid, center_26])
    
    # Initial radii
    # Grid circles can be 0.1. 26th circle smaller.
    radii_init = np.ones(26) * 0.1
    radii_init[-1] = 0.04
    
    # Flatten to parameter vector: [x1...x26, y1...y26, r1...r26]
    x0 = np.concatenate([centers_init[:, 0], centers_init[:, 1], radii_init])
    
    # Bounds
    # x, y in [0, 1]
    # r in [0, 0.5] (diameter <= 1)
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    # Optimization
    # Using L-BFGS-B for bounds support
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                      options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
    
    # Extract results
    cx_opt = result.x[:n]
    cy_opt = result.x[n:2*n]
    r_opt = result.x[2*n:]
    
    # Post-processing to ensure strict validity
    # Clip radii and centers to be safe against numerical noise
    # Although optimizer should respect bounds, center might be exactly on boundary with r>0?
    # Bounds enforce x in [0,1], but constraint x-r>=0 is soft penalty.
    # We should manually fix any tiny violations.
    
    # Fix boundary constraints strictly
    for i in range(n):
        r_i = r_opt[i]
        x_i = cx_opt[i]
        y_i = cy_opt[i]
        
        # Ensure r_i >= 0
        r_i = max(0.0, r_i)
        
        # Adjust radius to fit in bounds
        # r <= x, r <= 1-x, r <= y, r <= 1-y
        max_r_boundary = min(x_i, 1.0 - x_i, y_i, 1.0 - y_i)
        if r_i > max_r_boundary + 1e-9:
            r_i = max_r_boundary
            r_opt[i] = r_i
            
    # Fix overlap constraints strictly
    # Iteratively reduce radii if overlapping
    changed = True
    iterations = 0
    while changed and iterations < 100:
        changed = False
        iterations += 1
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt((cx_opt[i] - cx_opt[j])**2 + (cy_opt[i] - cy_opt[j])**2)
                sum_r = r_opt[i] + r_opt[j]
                if sum_r > dist + 1e-9:
                    # Overlap detected. Reduce radii equally or proportionally?
                    # Just reduce both to meet constraint
                    # Ideally we want to maximize sum, so we reduce as little as possible.
                    # If we reduce r_i and r_j such that r_i' + r_j' = dist.
                    # Simple heuristic: reduce each by half the excess
                    excess = sum_r - dist
                    reduction = excess / 2.0
                    r_opt[i] -= reduction
                    r_opt[j] -= reduction
                    # Ensure non-negative
                    r_opt[i] = max(0.0, r_opt[i])
                    r_opt[j] = max(0.0, r_opt[j])
                    changed = True

    # Final sum
    sum_radii = np.sum(r_opt)
    
    # Construct output arrays
    centers = np.column_stack((cx_opt, cy_opt))
    radii = r_opt
    
    return centers, radii, sum_radii

if __name__ == "__main__":
    centers, radii, total_radius = run_packing()
    print(f"Sum of radii: {total_radius}")
    # Quick validation check
    # (The user will run validate_packing, but good to check locally)
    try:
        # Simple check
        valid = True
        # Boundaries
        for i in range(len(centers)):
            x, y = centers[i]
            r = radii[i]
            if x < r - 1e-12 or x > 1 - r + 1e-12 or y < r - 1e-12 or y > 1 - r + 1e-12:
                valid = False
        # Overlaps
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                d = np.sqrt((centers[i][0]-centers[j][0])**2 + (centers[i][1]-centers[j][1])**2)
                if d < radii[i] + radii[j] - 1e-12:
                    valid = False
        print(f"Valid: {valid}")
    except Exception as e:
        print(f"Error: {e}")
