# sol_000200 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 7f4d5c4f) state=4e514716 sum of radii=2.334050 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def get_distance_matrix(centers):
    """Calculate pairwise distance matrix for centers."""
    # centers shape (n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    return np.sqrt(np.sum(diff**2, axis=2))

def evaluate_objective_and_violations(params, n=26, P=1000.0):
    """
    Objective: Maximize sum of radii.
    We minimize negative sum + penalty.
    """
    # Unpack parameters
    # params: [x1, y1, r1, x2, y2, r2, ...]
    # Or better: reshape directly
    centers = params[:2*n].reshape(n, 2)
    radii = params[2*n:]
    
    # Ensure radii are non-negative (though bounds should handle this)
    radii = np.maximum(radii, 0)
    
    # Objective: Negative sum of radii
    obj = -np.sum(radii)
    
    # Penalties
    penalty = 0.0
    
    # 1. Boundary penalties
    # Circle i must be inside [0,1]x[0,1]
    # x - r >= 0  => r - x <= 0
    # x + r <= 1  => x + r - 1 <= 0
    # Same for y
    
    # Violations for boundary
    # Left: x - r >= 0  -> violation if r - x > 0
    left_viol = np.maximum(0, radii - centers[:, 0])
    # Right: x + r <= 1 -> violation if x + r - 1 > 0
    right_viol = np.maximum(0, centers[:, 0] + radii - 1)
    # Bottom: y - r >= 0
    bottom_viol = np.maximum(0, radii - centers[:, 1])
    # Top: y + r <= 1
    top_viol = np.maximum(0, centers[:, 1] + radii - 1)
    
    boundary_viol = np.sum(left_viol**2) + np.sum(right_viol**2) + \
                    np.sum(bottom_viol**2) + np.sum(top_viol**2)
    
    # 2. Overlap penalties
    # dist(i, j) >= r_i + r_j
    # violation if r_i + r_j - dist > 0
    
    # Vectorized distance calculation
    # Centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # Radii sum matrix
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Only consider upper triangle to avoid double counting and self
    # But for penalty, double counting is fine (just scales P)
    # However, diagonal is 0, r_sum diagonal is 2r, violation 2r? 
    # We should ignore diagonal.
    
    # Create a mask for i < j
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    overlap_violations = np.maximum(0, r_sum - dists)
    overlap_viol = np.sum(overlap_violations[mask]**2)
    
    total_penalty = boundary_viol + overlap_viol
    obj += P * total_penalty
    
    return obj, total_penalty

def run_packing():
    np.random.seed(42)
    n = 26
    best_sum = 0
    best_centers = None
    best_radii = None
    
    # Strategy: Try multiple initial configurations and pick the best
    # Configurations:
    # 1. Grid with perturbation
    # 2. Hexagonal-like packing
    
    num_runs = 20
    
    for run in range(num_runs):
        # Initialize centers and radii
        # Start with a 5x5 grid pattern for 25 circles, plus 1
        # 5 rows, 5 cols.
        # x coords: 0.1, 0.3, 0.5, 0.7, 0.9
        # y coords: 0.1, 0.3, 0.5, 0.7, 0.9
        # Radius 0.09 (safe start)
        
        # Generate a base grid
        x_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_coords = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        
        # Create 25 points
        grid_points = []
        for y in y_coords:
            for x in x_coords:
                grid_points.append([x, y])
        grid_points = np.array(grid_points)
        
        # Add 26th point in a gap
        # Center of square (0.5, 0.5) is occupied.
        # Try a gap, e.g., (0.2, 0.2)
        extra_point = [0.2, 0.2]
        if run % 2 == 0:
            extra_point = [0.2, 0.8] # alternate position
        else:
            extra_point = [0.8, 0.2]
            
        centers_init = np.vstack([grid_points, [extra_point]])
        
        # Perturb centers slightly to break symmetry and help optimization
        noise_scale = 0.02
        centers_init += np.random.normal(0, noise_scale, size=centers_init.shape)
        
        # Clamp centers to [0, 1]
        centers_init = np.clip(centers_init, 0.001, 0.999)
        
        # Initial radii
        radii_init = np.full(n, 0.08)
        
        # Combine into params vector
        params_init = np.concatenate([centers_init.flatten(), radii_init])
        
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for i in range(n):
            bounds.append((0, 1)) # x
            bounds.append((0, 1)) # y
            bounds.append((0, 0.5)) # r
        
        # Optimization
        # Use L-BFGS-B for box constraints
        # Penalty weight P. Higher P enforces constraints more strictly but might make landscape rough.
        P = 5000.0 
        
        # Define objective for scipy (takes only params)
        def objective(params):
            val, _ = evaluate_objective_and_violations(params, n, P)
            return val
            
        result = minimize(objective, params_init, method='L-BFGS-B', bounds=bounds, 
                          options={'maxiter': 2000, 'ftol': 1e-12})
        
        if result.success or result.fun < 0: # fun is negative sum + penalty
            # Extract results
            centers_opt = result.x[:2*n].reshape(n, 2)
            radii_opt = result.x[2*n:]
            
            # Check validity and adjust if needed
            # The penalty method might not be perfectly strict.
            # We can shrink radii slightly to ensure validity.
            
            # First, check current sum
            current_sum = np.sum(radii_opt)
            
            # If penalty was low, solution might be valid.
            # Let's verify and repair.
            valid, centers_rep, radii_rep = repair_packing(centers_opt, radii_opt)
            
            if valid:
                final_sum = np.sum(radii_rep)
                if final_sum > best_sum:
                    best_sum = final_sum
                    best_centers = centers_rep
                    best_radii = radii_rep
    
    # If no valid solution found (unlikely), return a safe grid
    if best_centers is None:
        # Fallback to simple grid
        x_c = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        y_c = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        pts = []
        for y in y_c:
            for x in x_c:
                pts.append([x, y])
        pts.append([0.2, 0.2])
        centers = np.array(pts)
        radii = np.full(26, 0.09) # Safe radius
        return centers, radii, np.sum(radii)
        
    return best_centers, best_radii, best_sum

def repair_packing(centers, radii):
    """
    Ensures validity by shrinking radii if overlaps or boundary violations exist.
    Returns (True, centers, radii) if valid, else (False, None, None) if unrecoverable (rare).
    """
    n = len(radii)
    # We perform a few iterations of shrinking
    for _ in range(10):
        valid = True
        # Check boundary
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Clamp radius to boundary distance
            max_r_bound = min(x, 1-x, y, 1-y)
            if r > max_r_bound + 1e-10:
                radii[i] = max(0, max_r_bound)
                valid = False # Needs recheck
        
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                sum_r = radii[i] + radii[j]
                if sum_r > dist + 1e-10:
                    # Overlap detected. Reduce radii proportionally.
                    # Ideally move centers, but shrinking is safer for quick fix.
                    # Reduce both by half the excess?
                    excess = sum_r - dist
                    reduction = excess / 2.0 + 1e-6 # slight extra margin
                    radii[i] = max(0, radii[i] - reduction)
                    radii[j] = max(0, radii[j] - reduction)
                    valid = False

        if valid:
            return True, centers, radii
            
    # Final check
    # If still not valid, maybe try to reduce radii more aggressively?
    # Or just return what we have, but validation will fail.
    # Let's try one more aggressive reduction if needed.
    # Actually, if repair fails, we might return a known valid solution.
    
    # Let's assume it works.
    return validate_packing_manual(centers, radii), centers, radii

def validate_packing_manual(centers, radii):
    """Helper to check validity without printing"""
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

# Import for validation check at end
import numpy as np

# We need to import the validate_packing function logic or just assume the environment has it?
# The prompt says "We will run the below validation function". 
# So we don't need to implement it, but our run_packing must produce valid output.
# The repair function is internal.
