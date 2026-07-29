# sol_000055 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 67b9141d) state=1b8812ad sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # --- Initialization ---
    # Hexagonal packing initialization
    # We generate points in a hexagonal lattice to provide a dense initial configuration.
    # Start with a small radius to ensure no initial overlaps.
    r_init = 0.05
    
    centers = []
    
    # Generate points in hexagonal lattice
    # Row spacing: r * sqrt(3)
    # Col spacing: 2 * r
    # We generate more points than needed and pick the first n.
    
    points = []
    y = r_init
    row_idx = 0
    
    while y + r_init <= 1.0:
        x = r_init
        # Alternate row offsets for hexagonal pattern
        # Even rows: x starts at r
        # Odd rows: x starts at 2r (shifted by r)
        if row_idx % 2 == 1:
            x = 2 * r_init
            
        while x + r_init <= 1.0:
            points.append([x, y])
            x += 2 * r_init
        y += r_init * np.sqrt(3)
        row_idx += 1
        
    # If we generated enough points, take the first n
    if len(points) >= n:
        centers = np.array(points[:n])
    else:
        # Fallback to random distribution if lattice generation fails (unlikely)
        centers = np.random.rand(n, 2) * 0.6 + 0.2
        
    # Initial radii
    radii = np.full(n, r_init)
    
    # Flatten parameters for optimizer: [x1, y1, r1, x2, y2, r2, ...]
    x0 = []
    for i in range(n):
        x0.extend([centers[i, 0], centers[i, 1], radii[i]])
    x0 = np.array(x0)
    
    # --- Optimization Setup ---
    # We use a penalty method to handle constraints within an unconstrained (bounded) optimizer.
    # Objective: Minimize -Sum(radii) + Penalty * Violations
    
    def make_objective(penalty_weight):
        def objective(v):
            params = v.reshape(n, 3)
            x = params[:, 0]
            y = params[:, 1]
            r = params[:, 2]
            
            # Base objective: maximize sum of radii (so minimize negative sum)
            obj_val = -np.sum(r)
            
            # 1. Boundary Violations
            # Circle i must be inside [0,1]x[0,1]
            # Constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
            # Violation is max(0, r - x)^2 etc.
            
            # Left/Bottom boundaries: r - x <= 0 => violation if r > x
            err = np.maximum(0, r - x)
            obj_val += penalty_weight * np.sum(err**2)
            
            # Right/Top boundaries: r - (1-x) <= 0 => violation if r > 1-x
            err = np.maximum(0, r - (1.0 - x))
            obj_val += penalty_weight * np.sum(err**2)
            
            err = np.maximum(0, r - y)
            obj_val += penalty_weight * np.sum(err**2)
            
            err = np.maximum(0, r - (1.0 - y))
            obj_val += penalty_weight * np.sum(err**2)
            
            # 2. Overlap Violations
            # Distance between centers >= sum of radii
            # dist >= r_i + r_j  =>  r_i + r_j - dist <= 0
            
            # Compute pairwise differences
            # x[:, None] creates a column vector, broadcasting handles the matrix
            dx = x[:, None] - x[None, :]
            dy = y[:, None] - y[None, :]
            dr = r[:, None] + r[None, :]
            
            # Euclidean distance
            dist = np.sqrt(dx**2 + dy**2)
            
            # Violation amount: r_i + r_j - dist
            # We only care if this is positive (overlap)
            diff = dr - dist
            
            # Diagonal is 0 (distance to self), no self-overlap
            np.fill_diagonal(diff, 0)
            
            # Sum of squared positive violations
            violations = np.maximum(0, diff)
            obj_val += penalty_weight * np.sum(violations**2)
            
            return obj_val
        return objective

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot exceed 0.5 in unit square)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    bounds = tuple(bounds)
    
    # --- Execution ---
    best_params = None
    best_sum_radii = -np.inf
    best_valid = False
    
    # We run multiple restarts with perturbations to escape local optima
    num_restarts = 8
    penalty = 2000.0 # High penalty to enforce constraints strictly
    
    for i in range(num_restarts):
        # Create a perturbed initial guess
        # Small noise to explore neighborhood
        current_x0 = x0 + np.random.normal(0, 0.005, size=x0.shape)
        
        # Clip to bounds roughly to help optimizer
        # x, y in [0,1], r in [0, 0.5]
        current_x0[0::3] = np.clip(current_x0[0::3], 0, 1)
        current_x0[1::3] = np.clip(current_x0[1::3], 0, 1)
        current_x0[2::3] = np.clip(current_x0[2::3], 0, 0.5)
        
        # Optimize
        # L-BFGS-B is efficient for bound-constrained problems
        res = minimize(make_objective(penalty), 
                       current_x0, 
                       method='L-BFGS-B', 
                       bounds=bounds, 
                       options={'maxiter': 1500, 'ftol': 1e-12})
        
        # Evaluate the result
        final_params = res.x.reshape(n, 3)
        r_final = final_params[:, 2]
        sum_r = np.sum(r_final)
        
        # Check validity (approximate)
        # If the penalty term was minimized, violations should be low.
        # We can compute the objective value to check.
        obj_val = make_objective(penalty)(res.x)
        theoretical_min = -sum_r # If valid, obj should be close to -sum_r
        
        # If objective is close to -sum_r, it's likely valid.
        # Allow small numerical error.
        if obj_val <= -sum_r + 1e-6:
            if sum_r > best_sum_radii:
                best_sum_radii = sum_r
                best_params = res.x
                best_valid = True
    
    # Extract best result
    if best_params is not None:
        centers_final = best_params.reshape(n, 3)[:, :2]
        radii_final = best_params.reshape(n, 3)[:, 2]
    else:
        # Fallback to initial if optimization failed
        centers_final = centers
        radii_final = radii
        best_sum_radii = np.sum(radii_final)

    # Final safety check and adjustment
    # Ensure no negative radii
    radii_final = np.maximum(radii_final, 0.0)
    
    # Ensure centers are within bounds relative to radii
    # Although optimizer should have handled this, clamp if necessary
    for i in range(n):
        x, y = centers_final[i]
        r = radii_final[i]
        # Clamp x
        if x < r: x = r
        if x > 1 - r: x = 1 - r
        # Clamp y
        if y < r: y = r
        if y > 1 - r: y = 1 - r
        centers_final[i] = [x, y]
        
    # Re-calculate sum
    final_sum = np.sum(radii_final)
    
    return centers_final, radii_final, final_sum
