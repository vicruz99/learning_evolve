# sol_000354 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c8413f9) state=a011d31e sum of radii=1.921894 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize
import math

def run_packing():
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    
    # 1. Initialization: Greedy Max-Min Dispersion
    # Generate a dense grid of candidate points
    grid_size = 40
    x_grid = np.linspace(0.05, 0.95, grid_size)
    y_grid = np.linspace(0.05, 0.95, grid_size)
    candidates = np.array([[x, y] for x in x_grid for y in y_grid])
    
    selected_indices = []
    selected_centers = []
    
    # Start with a point near the center to ensure good distribution
    center_idx = np.argmin(np.hypot(candidates[:, 0] - 0.5, candidates[:, 1] - 0.5))
    selected_indices.append(center_idx)
    selected_centers.append(candidates[center_idx].copy())
    candidates = np.delete(candidates, center_idx, axis=0)
    
    # Greedily select remaining points to maximize min-distance
    for _ in range(n_circles - 1):
        # Calculate min distance from each candidate to the current set of selected centers
        # dists shape: (num_candidates, n_selected)
        dists = np.linalg.norm(candidates[:, np.newaxis, :] - np.array(selected_centers), axis=2)
        min_dists = np.min(dists, axis=1)
        
        # Pick candidate with largest minimum distance
        best_idx = np.argmax(min_dists)
        selected_indices.append(best_idx) # Not strictly needed as we modify candidates
        selected_centers.append(candidates[best_idx].copy())
        
        # Remove chosen candidate
        candidates = np.delete(candidates, best_idx, axis=0)
        
        # Early exit if no candidates left (should not happen with grid_size=40)
        if candidates.size == 0:
            break
            
    centers = np.array(selected_centers)
    radii = np.full(n_circles, 0.01) # Small initial radii
    
    # 2. Force-Directed Growing Algorithm
    # Iteratively grow radii and resolve overlaps
    growth_rate = 1.0005
    max_iterations = 2000
    
    for _ in range(max_iterations):
        # Grow radii
        radii *= growth_rate
        
        # Resolve Overlaps and Boundaries
        # We iterate multiple times per growth step to ensure stability
        for _ in range(5): 
            moved = False
            
            # Check Boundary Constraints
            for i in range(n_circles):
                x, y = centers[i]
                r = radii[i]
                
                # Left/Right boundaries
                if x - r < 0:
                    centers[i, 0] = r
                    moved = True
                elif x + r > 1:
                    centers[i, 0] = 1 - r
                    moved = True
                    
                # Top/Bottom boundaries
                if y - r < 0:
                    centers[i, 1] = r
                    moved = True
                elif y + r > 1:
                    centers[i, 1] = 1 - r
                    moved = True
            
            # Check Pairwise Overlaps
            # Vectorized calculation for efficiency
            # Centers: (N, 2)
            # Diffs: (N, N, 2)
            # Dist matrix: (N, N)
            
            # Calculate distance matrix
            # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            
            # Distance matrix is symmetric, lower triangle i > j
            # We want to check pairs (i, j) with i > j
            # Indices for upper triangle
            rows, cols = np.triu_indices(n_circles, k=1)
            
            for i, j in zip(rows, cols):
                d = dists[i, j]
                r_sum = radii[i] + radii[j]
                
                if d < r_sum:
                    # Overlap detected
                    overlap = r_sum - d
                    
                    # Direction from j to i
                    if d > 1e-9:
                        direction = (centers[i] - centers[j]) / d
                    else:
                        # If centers coincide, push randomly or apart
                        direction = np.random.rand(2) * 2 - 1
                        direction /= np.linalg.norm(direction)
                    
                    # Push apart
                    push_amount = overlap / 2.0
                    
                    centers[i] += direction * push_amount
                    centers[j] -= direction * push_amount
                    moved = True
            
            if not moved:
                break
                
    # 3. Final Refinement using LP/Solver
    # Fix centers, maximize sum of radii subject to constraints
    # This is a Linear Programming problem for radii.
    # Maximize sum(r_i)
    # s.t. r_i <= dist(center_i, boundary)
    #      r_i + r_j <= dist(center_i, center_j)
    #      r_i >= 0
    
    # We can use scipy.optimize.linprog
    # Variables: r_0, ..., r_25
    c_obj = -np.ones(n_circles) # Minimize negative sum
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    # r_i - x_i <= 0  =>  1*r_i <= x_i
    # r_i - (1-x_i) <= 0 => 1*r_i <= 1-x_i
    for i in range(n_circles):
        x, y = centers[i]
        # 4 constraints per circle
        # r_i <= x
        row = np.zeros(n_circles)
        row[i] = 1
        A_ub.append(row)
        b_ub.append(x)
        
        # r_i <= 1-x
        A_ub.append(row)
        b_ub.append(1 - x)
        
        # r_i <= y
        A_ub.append(row)
        b_ub.append(y)
        
        # r_i <= 1-y
        A_ub.append(row)
        b_ub.append(1 - y)
        
    # Overlap constraints: r_i + r_j <= dist(i, j)
    # We only need to check pairs that are close, but checking all is safe for N=26
    # 26*25/2 = 325 constraints
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n_circles)
            row[i] = 1
            row[j] = 1
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    bounds = [(0, None) for _ in range(n_circles)]
    
    try:
        res = scipy.optimize.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
        else:
            # Fallback to current radii if LP fails (unlikely)
            pass
    except Exception:
        pass
        
    # Clip radii to ensure non-negative (linprog should handle this)
    radii = np.maximum(radii, 0)
    
    # Final validation and clean-up
    # Ensure centers are valid (they should be, but just in case projection happened)
    # Re-verify constraints and adjust if necessary? 
    # The LP solution guarantees constraints are met for the FIXED centers.
    # So the resulting (centers, radii) is valid.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper to run and print
if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    
    # Validation check (optional, using the provided logic structure)
    # In a real scenario, we would call validate_packing
    # But here we just output the function.
