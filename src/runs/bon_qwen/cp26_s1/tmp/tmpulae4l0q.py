import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """Solves the LP to find the maximum sum of radii for fixed centers."""
    n = len(centers)
    
    # Objective: Maximize sum(radii) -> Minimize -sum(radii)
    c = -np.ones(n)
    
    # Constraints Matrix A_ub * radii <= b_ub
    constraints = []
    bounds = []
    
    # 1. Distance constraints: r_i + r_j <= dist_ij
    # We flatten the pairs to rows
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1
            row[j] = 1
            constraints.append((row, dist))
            
    # 2. Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1 - x, y, 1 - y)
        row = np.zeros(n)
        row[i] = 1
        constraints.append((row, max_r))
        
    # Separate rows and bounds
    A_ub = [c_row for c_row, _ in constraints]
    b_ub = [b_val for _, b_val in constraints]
    
    # Bounds for radii (non-negative)
    bounds = [(0, None)] * n
    
    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        return res.x, -res.fun
    else:
        # Fallback if LP fails (shouldn't happen with valid centers)
        return np.zeros(n), 0.0

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square.
    """
    np.random.seed(42)
    n = 26
    
    # --- 1. Initialization: Hexagonal Grid ---
    centers = []
    # Estimate spacing for 26 circles. 5x5 grid has 25.
    # We'll use a base spacing and adjust.
    row_spacing = 1.0 / 5.0 # ~0.2
    
    # Generate a hexagonal-like grid
    # We want to fit 26 circles. 5 rows of 5 is 25. 
    # Let's try 6 rows of ~4 or 5 circles.
    # Or just fill a grid and trim/adjust.
    
    # Simple dense packing init
    cols = 6
    rows = 5
    x_step = 1.0 / (cols + 1)
    y_step = 1.0 / (rows + 1)
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= n: break
            x = (c + 1) * x_step
            y = (r + 1) * y_step
            # Offset odd rows for hexagonal packing
            if r % 2 == 1:
                x += x_step / 2
            centers.append([x, y])
            count += 1
        if count >= n: break
        
    centers = np.array(centers)
    
    # --- 2. Optimization Loop ---
    # We will iterate to improve the sum of radii by moving centers.
    
    best_centers = centers.copy()
    best_radii, best_sum = get_optimal_radii(best_centers)
    
    # Parameters for the search
    step_size = 0.02
    
    for iteration in range(500):
        # Compute current optimal radii
        current_radii, current_sum = get_optimal_radii(best_centers)
        
        # If we are already very good, try finer tuning
        if iteration > 400:
            step_size = 0.001
        elif iteration > 200:
            step_size = 0.005

        improved = False
        
        # Try to move each center
        # Heuristic: Move center i in the direction that increases its "freedom"
        # A simple random perturbation is often robust for this kind of non-convex problem
        
        # Shuffle order of centers to update to avoid bias
        indices = np.random.permutation(n)
        
        for i in indices:
            # Try random moves
            for _ in range(10): # Try 10 perturbations per center per iteration
                # Generate a random direction
                move = np.random.uniform(-step_size, step_size, 2)
                new_center = best_centers[i] + move
                
                # Ensure within bounds (with small padding for safety)
                if 0 <= new_center[0] <= 1 and 0 <= new_center[1] <= 1:
                    # Create a copy of centers
                    trial_centers = best_centers.copy()
                    trial_centers[i] = new_center
                    
                    # Check feasibility/optimality quickly? 
                    # Actually, just solve LP. It's fast for 26 vars.
                    trial_radii, trial_sum = get_optimal_radii(trial_centers)
                    
                    if trial_sum > current_sum + 1e-6:
                        best_centers = trial_centers
                        current_radii = trial_radii
                        current_sum = trial_sum
                        improved = True
                        # Accept move and stop searching for this center for now
                        break 
            
            if not improved and iteration % 50 == 0:
                 # If stuck, maybe shake the whole configuration slightly?
                 # (Not implemented to keep it simple)
                 pass

        # Every 100 iterations, try a larger random jump to escape local minima
        if iteration > 0 and iteration % 100 == 0:
            trial_centers = best_centers + np.random.uniform(-0.05, 0.05, best_centers.shape)
            # Clip to bounds
            trial_centers = np.clip(trial_centers, 0.01, 0.99)
            trial_radii, trial_sum = get_optimal_radii(trial_centers)
            if trial_sum > best_sum:
                best_centers = trial_centers
                best_radii = trial_radii
                best_sum = trial_sum

    # Final calculation
    final_radii, final_sum = get_optimal_radii(best_centers)
    
    # Round values to avoid precision issues near boundaries
    final_radii = np.clip(final_radii, 0, None)
    
    # Ensure strict validity with a tiny epsilon reduction if needed
    # But the LP constraints are hard, so it should be valid.
    # Just return.
    
    return best_centers, final_radii, float(final_sum)