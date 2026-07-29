# sol_000063 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 608ae89b) state=8e8e6b00 sum of radii=0.000003 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def compute_distance_matrix(centers):
    """Compute pairwise Euclidean distance matrix for a set of centers."""
    # centers shape: (n, 2)
    # dist matrix shape: (n, n)
    n = centers.shape[0]
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=2))

def compute_overlap_penalty(centers, radii):
    """
    Compute penalty for overlaps between circles.
    Penalty is squared if overlap exists, 0 otherwise.
    Overlap condition: dist < r1 + r2
    Penalty term: max(0, r1 + r2 - dist)^2
    """
    n = len(radii)
    dist_matrix = compute_distance_matrix(centers)
    radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Calculate overlap amount
    overlap = radii_sum_matrix - dist_matrix
    
    # Only consider overlaps > 0
    # Using np.maximum to be vectorized
    violation = np.maximum(0, overlap)
    
    # Sum of squared violations
    penalty = np.sum(violation ** 2)
    return penalty

def compute_boundary_penalty(centers, radii):
    """
    Compute penalty for circles outside the unit square [0,1]x[0,1].
    Constraints: r <= x <= 1-r  => x-r >= 0 and 1-r-x >= 0
    Same for y.
    """
    n = len(radii)
    x = centers[:, 0]
    y = centers[:, 1]
    
    # Violations
    # x < r => r - x > 0
    v_x1 = np.maximum(0, radii - x)
    # x > 1-r => x + r - 1 > 0
    v_x2 = np.maximum(0, x + radii - 1)
    
    v_y1 = np.maximum(0, radii - y)
    v_y2 = np.maximum(0, y + radii - 1)
    
    penalty = np.sum(v_x1**2 + v_x2**2 + v_y1**2 + v_y2**2)
    return penalty

def cost_function(vars, radii_weight=1.0, overlap_weight=1e6, boundary_weight=1e6):
    """
    Objective function to minimize.
    vars: flattened array [x1, y1, r1, x2, y2, r2, ...]
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = vars[3*i]
        centers[i, 1] = vars[3*i + 1]
        radii[i] = vars[3*i + 2]
        
    # Objective: Maximize sum of radii -> Minimize negative sum
    obj = -np.sum(radii) * radii_weight
    
    # Penalties
    obj += overlap_weight * compute_overlap_penalty(centers, radii)
    obj += boundary_weight * compute_boundary_penalty(centers, radii)
    
    return obj

def run_packing():
    n = 26
    
    # Initialization: Grid layout
    # We want to pack 26 circles. A 5x5 grid has 25. 
    # Let's try a hexagonal-like distribution or just a dense grid.
    # 6 columns, 5 rows = 30 spots. We pick 26.
    # Or just random perturbation of a grid.
    
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Start with small radii
    
    # Create a rough grid
    # 6 columns
    cols = 6
    rows = 5
    count = 0
    
    # Spacing
    x_step = 1.0 / (cols + 1)
    y_step = 1.0 / (rows + 1)
    
    for r in range(rows):
        for c in range(cols):
            if count < n:
                centers[count, 0] = (c + 1) * x_step
                centers[count, 1] = (r + 1) * y_step
                count += 1
            else:
                break
    
    # Flatten for optimizer
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i + 1] = centers[i, 1]
        x0[3*i + 2] = radii[i]
        
    # Bounds
    bounds = []
    for i in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 0.5)) # r
    
    # Optimization parameters
    # We use a penalty method. High weights for constraints.
    # L-BFGS-B is good for bounds.
    
    # To make it robust, we might run a few iterations with increasing penalty or radii.
    # But for a single shot, let's try high penalty.
    
    res = minimize(cost_function, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-12})
    
    # Extract solution
    best_vars = res.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i, 0] = best_vars[3*i]
        final_centers[i, 1] = best_vars[3*i + 1]
        final_radii[i] = best_vars[3*i + 2]
        
    # Validation and small correction if needed (numerical errors)
    # The penalty method might leave tiny overlaps. 
    # We can try to shrink radii slightly if overlaps exist, but validation function handles 1e-12 tolerance.
    
    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii

# Helper functions are defined at top level as requested
# No closures or lambdas used in the logic above.

if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
