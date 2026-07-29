# sol_000179 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0ae2e142) state=99b91b5b sum of radii=2.480088 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def get_initial_hexagonal_packing(n, square_size=1.0):
    """
    Generates an initial hexagonal packing of n circles.
    """
    centers = []
    radii = []
    
    # Estimate a starting radius. 
    # For n=26, a rough estimate for equal circles in hexagonal packing
    # density is high. Let's start with r=0.09 which fits comfortably.
    r_init = 0.085 
    
    # Hexagonal grid parameters
    # Row height: sqrt(3) * r
    # Horizontal spacing: 2 * r
    
    y = r_init
    row_idx = 0
    
    while len(centers) < n:
        x_start = r_init if row_idx % 2 == 0 else 2 * r_init # Shift odd rows
        x = x_start
        
        while x + r_init <= square_size and len(centers) < n:
            centers.append([x, y])
            radii.append(r_init)
            x += 2 * r_init
        
        y += math.sqrt(3) * r_init
        row_idx += 1
        
    return np.array(centers), np.array(radii)

def calculate_objective_and_gradient(v, n, penalty_weight):
    """
    Calculates the objective (negative sum of radii) and gradient.
    v: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = v[3 * i]
        centers[i, 1] = v[3 * i + 1]
        radii[i] = v[3 * i + 2]
        
    obj = -np.sum(radii)
    
    # Penalty terms
    penalty = 0.0
    
    # 1. Boundary penalties
    # x - r >= 0 => r - x <= 0
    # x + r <= 1 => x + r - 1 <= 0
    # Same for y
    for i in range(n):
        x, y, r = centers[i, 0], centers[i, 1], radii[i]
        
        # Left
        viol = r - x
        if viol > 0:
            penalty += penalty_weight * viol**2
        
        # Right
        viol = x + r - 1.0
        if viol > 0:
            penalty += penalty_weight * viol**2
            
        # Bottom
        viol = r - y
        if viol > 0:
            penalty += penalty_weight * viol**2
            
        # Top
        viol = y + r - 1.0
        if viol > 0:
            penalty += penalty_weight * viol**2

    # 2. Overlap penalties
    # dist >= r_i + r_j => r_i + r_j - dist <= 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            dist = math.sqrt(dist_sq)
            
            # Avoid division by zero if centers coincide
            if dist < 1e-9:
                dist = 1e-9
                
            sum_r = radii[i] + radii[j]
            viol = sum_r - dist
            
            if viol > 0:
                penalty += penalty_weight * viol**2
                
    total_obj = obj + penalty
    return total_obj

def run_packing():
    n = 26
    
    # 1. Initialization
    centers_init, radii_init = get_initial_hexagonal_packing(n)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3 * i] = centers_init[i, 0]
        x0[3 * i + 1] = centers_init[i, 1]
        x0[3 * i + 2] = radii_init[i]
        
    # Bounds: x in [0, 1], y in [0, 1], r >= 0
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, None)) # r (upper bound not strictly necessary but good practice)
        
    # 2. Optimization
    # We use a high penalty weight to enforce constraints.
    # The problem is non-convex, so results depend on initialization.
    # L-BFGS-B is efficient for bound-constrained problems.
    
    penalty_weight = 1000.0
    
    # Define objective wrapper
    def objective(v):
        return calculate_objective_and_gradient(v, n, penalty_weight)
        
    # Run optimization
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': 2000})
    
    # 3. Extract results
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = res.x[3 * i]
        final_centers[i, 1] = res.x[3 * i + 1]
        final_radii[i] = res.x[3 * i + 2]
        
    # 4. Post-processing to ensure strict validity
    # Clip radii to be non-negative
    final_radii = np.maximum(final_radii, 0.0)
    
    # Ensure centers are within bounds (though optimizer should handle this)
    final_centers[:, 0] = np.clip(final_centers[:, 0], 0.0, 1.0)
    final_centers[:, 1] = np.clip(final_centers[:, 1], 0.0, 1.0)
    
    # Adjust radii to strictly satisfy boundary constraints
    # r <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = final_centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if final_radii[i] > max_r:
            final_radii[i] = max_r
            
    # Adjust radii to strictly satisfy non-overlap constraints
    # This is a relaxation step. If circles overlap, shrink them.
    # We can iteratively shrink the larger of the two or scale both.
    # A simple pass:
    for iteration in range(50):
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dx = final_centers[i, 0] - final_centers[j, 0]
                dy = final_centers[i, 1] - final_centers[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                sum_r = final_radii[i] + final_radii[j]
                
                if dist < sum_r - 1e-12:
                    # Overlap detected
                    overlap_found = True
                    # Shrink radii to just touch
                    # Ideally distribute shrinkage. 
                    # Simplest: reduce sum_r to dist
                    # Reduce proportionally?
                    if sum_r > 1e-12:
                        scale = dist / sum_r
                        final_radii[i] *= scale
                        final_radii[j] *= scale
                    else:
                        final_radii[i] = 0.0
                        final_radii[j] = 0.0
                        
        if not overlap_found:
            break
            
    # Re-check boundary constraints after shrinking radii (radii decreased, so safe)
    # But check again just in case
    for i in range(n):
        x, y = final_centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        if final_radii[i] > max_r:
            final_radii[i] = max_r

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper function to run and print for testing (optional)
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
