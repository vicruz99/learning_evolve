# sol_000058 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 05a03f22) state=2d2aac42 sum of radii=2.315748 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def solve_radii(centers):
    """
    Given a set of centers, solve the Linear Programming problem to find
    the radii that maximize sum(r) subject to packing constraints.
    
    Args:
        centers: np.array of shape (n, 2)
        
    Returns:
        radii: np.array of shape (n)
    """
    n = centers.shape[0]
    
    # Variables: r_0, r_1, ..., r_{n-1}
    # Objective: maximize sum(r_i) -> minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Non-negativity: -r_i <= 0 (handled by bounds)
    # 2. Boundary constraints: r_i <= min(x_i, 1-x_i, y_i, 1-y_i)
    # 3. Overlap constraints: r_i + r_j <= dist(i, j)
    
    A_ub = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        x, y = centers[i]
        r_max_boundary = min(x, 1.0 - x, y, 1.0 - y)
        # r_i <= r_max_boundary  =>  r_i <= r_max_boundary
        # In form A x <= b: [0... 1 ... 0] r <= r_max_boundary
        row = np.zeros(n)
        row[i] = 1.0
        A_ub.append(row)
        b_ub.append(r_max_boundary)
        
    # Overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            # r_i + r_j <= dist
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub.append(row)
            b_ub.append(dist)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds for variables r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback if LP fails (e.g. infeasible due to numerical issues)
            # Return very small radii
            return np.full(n, 1e-6)
    except Exception:
        return np.full(n, 1e-6)

def get_score(centers):
    """
    Calculate the sum of radii for a given set of centers.
    """
    radii = solve_radii(centers)
    return np.sum(radii)

def run_packing():
    """
    Main function to run the packing optimization.
    Returns (centers, radii, sum_radii).
    """
    n = 26
    
    # --- Initialization ---
    # Try a hexagonal packing initialization
    # Estimate radius for 26 circles in hexagonal packing roughly.
    # Area ~ 1, density ~ 0.9, 26 * pi * r^2 ~ 0.9 => r ~ 0.105
    # But boundary effects reduce this. Let's start with r=0.1
    r_init = 0.1
    
    centers_init = np.zeros((n, 2))
    
    # Generate hexagonal grid points
    # Spacing dx = 2*r, dy = sqrt(3)*r
    # But we don't know r yet. Let's just generate a dense grid and scale.
    # Or generate centers for r=1 and scale to fit.
    
    # Let's try to fit a hexagonal pattern of radius 1 into a box, then scale.
    # Hexagonal pattern: rows shifted.
    # Rows 0, 2, 4... at x = 0, 2, 4...
    # Rows 1, 3, 5... at x = 1, 3, 5... (shifted by 1 unit in x if spacing is 2)
    # Actually standard hex grid:
    # (0, 0), (2, 0), (4, 0)...
    # (1, sqrt(3)), (3, sqrt(3))...
    
    # We need 26 points.
    # Let's pick a grid size that gives roughly 26 points.
    # Maybe 6 columns, 5 rows?
    # Col 0: 5 points
    # Col 1: 5 points (shifted y)
    # ...
    
    # Let's just generate random points inside [0,1] and optimize?
    # Random start is often robust for this kind of problem if local search is good.
    # But grid is better.
    
    # Grid initialization
    # 5x5 grid is 25 points. Add one in the middle?
    # Or 6x5 grid (30 points) and remove 4?
    
    # Let's use a 6x5 rectangular grid of points, scale to fit, then remove 4 outer ones?
    # No, let's just place 26 points in a hexagonal arrangement.
    
    points = []
    # Hexagonal packing logic
    # dx = 1, dy = sqrt(3)/2 for unit diameter circles?
    # Let's assume spacing 1 for now.
    
    # We want to fill the square.
    # Let's create a pattern.
    # 5 rows.
    # Row 0: 6 points
    # Row 1: 5 points (shifted)
    # Row 2: 6 points
    # Row 3: 5 points
    # Row 4: 4 points?
    # Sum = 6+5+6+5+4 = 26. Perfect.
    
    # Let's define coordinates for these.
    # Let width be W, height be H.
    # Horizontal spacing s_x = W / (max_cols - 1 + 0.5)?
    # This is getting complicated to fit exactly.
    
    # Simpler: Just scatter points in a grid and let optimization handle it.
    # 5x6 grid = 30 points. We need 26.
    # Remove 4 corner points?
    
    xs = np.linspace(0, 1, 6)
    ys = np.linspace(0, 1, 5)
    grid_points = np.array([(x, y) for x in xs for y in ys])
    # 30 points.
    # Remove 4 points to get 26. Which ones?
    # Maybe the ones closest to corners? Or just random.
    # Let's remove indices 0, 1, 29, 28 (corners)?
    # Actually, corners are good for large circles.
    # Maybe remove points from the middle? No.
    # Let's just keep first 26.
    centers_init = grid_points[:26].copy()
    
    # Add some noise to avoid symmetry traps
    np.random.seed(42)
    centers_init += np.random.uniform(-0.02, 0.02, size=centers_init.shape)
    # Clip to [0,1]
    centers_init = np.clip(centers_init, 0.01, 0.99)
    
    best_centers = centers_init.copy()
    best_score = get_score(best_centers)
    best_radii = solve_radii(best_centers)
    
    # --- Optimization Loop ---
    # Local search with random perturbations
    # We can try to adjust centers to increase sum of radii.
    
    current_centers = best_centers.copy()
    step_size = 0.02 # Initial step size for perturbation
    
    # We can run multiple restarts or just one long run.
    # Let's do a few iterations of perturbation.
    
    iterations = 1000
    for _ in range(iterations):
        # Perturb all centers
        perturbation = np.random.uniform(-step_size, step_size, size=current_centers.shape)
        new_centers = current_centers + perturbation
        
        # Clip to valid range [epsilon, 1-epsilon] to allow some radius
        new_centers = np.clip(new_centers, 0.001, 0.999)
        
        new_score = get_score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            best_radii = solve_radii(best_centers)
            # If we improve, maybe keep step size or increase slightly?
            # But usually we want to explore.
            # If we accept, we stay there.
            # If we don't accept, we revert.
            current_centers = new_centers.copy()
        else:
            # Revert
            pass
            
        # Reduce step size occasionally to refine
        if _ % 100 == 0 and _ > 0:
            step_size *= 0.9
            
    # Final refinement with smaller steps
    step_size = 0.001
    for _ in range(500):
        perturbation = np.random.uniform(-step_size, step_size, size=best_centers.shape)
        new_centers = best_centers + perturbation
        new_centers = np.clip(new_centers, 0.001, 0.999)
        
        new_score = get_score(new_centers)
        
        if new_score > best_score:
            best_score = new_score
            best_centers = new_centers.copy()
            best_radii = solve_radii(best_centers)
            
    # --- Validation and Return ---
    # The radii from LP are guaranteed to satisfy constraints for the given centers.
    # However, due to float precision, we should double check or re-solve to be safe.
    # But linprog is usually robust.
    
    # Ensure centers are valid (within [0,1])
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Re-solve radii to ensure consistency with final clipped centers
    final_radii = solve_radii(best_centers)
    
    # Calculate final sum
    final_sum = np.sum(final_radii)
    
    return best_centers, final_radii, final_sum
