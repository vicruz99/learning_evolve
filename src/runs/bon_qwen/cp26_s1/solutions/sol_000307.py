# sol_000307 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 39d28b7b) state=7b2d17d0 sum of radii=1.300000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
import scipy.optimize as opt

# Set a random seed for reproducibility during the generation phase
np.random.seed(42)

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        
        x, y = centers[i]
        r = radii[i]
        
        # Check if circles are inside the unit square with tolerance
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    # Check for overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False

    return True

def get_constraint_matrix(centers):
    """
    Computes the distance matrix between centers.
    """
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    # Zero out diagonal
    np.fill_diagonal(dists, 0)
    return dists

def push_apart(centers, radii, dists, max_iter=50):
    """
    Resolves overlaps by pushing centers apart iteratively.
    """
    n = len(centers)
    centers = centers.copy()
    radii = radii.copy()

    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                r_sum = radii[i] + radii[j]
                d = dists[i, j]
                
                if d < r_sum and d > 1e-12:
                    overlap = r_sum - d
                    # Direction from j to i
                    vec = centers[i] - centers[j]
                    # Normalize
                    vec_norm = vec / d
                    # Push apart equally
                    push_vec = vec_norm * (overlap / 2.0)
                    
                    # Move centers
                    new_i = centers[i] + push_vec
                    new_j = centers[j] - push_vec
                    
                    # Boundary checks for i
                    if (new_i[0] >= radii[i] and new_i[0] <= 1 - radii[i] and
                        new_i[1] >= radii[i] and new_i[1] <= 1 - radii[i]):
                        centers[i] = new_i
                    else:
                        # If moving violates boundary, try to move only j
                        if (new_j[0] >= radii[j] and new_j[0] <= 1 - radii[j] and
                            new_j[1] >= radii[j] and new_j[1] <= 1 - radii[j]):
                            centers[j] = new_j
                            # Recalculate distance
                            dists[i, j] = dists[j, i] = np.sqrt(np.sum((centers[i] - centers[j])**2))
                            moved = True
                            continue
                        
                        # If neither can move fully, move as much as possible
                        # (simplified: just move i slightly towards center if possible, or do nothing)
                        pass 
                    
                    # Recalculate distance
                    dists[i, j] = dists[j, i] = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    moved = True
        
        # Update diagonal-free distance matrix for remaining checks? 
        # Actually, just updating the specific pair is O(1), but we need full matrix for next pass.
        dists = get_constraint_matrix(centers)

        if not moved:
            break
            
    return centers, radii

def adjust_radii_and_fix(centers, radii, step=0.0005):
    """
    Attempts to increase radii uniformly and fixes overlaps.
    """
    n = len(radii)
    dists = get_constraint_matrix(centers)
    
    # Calculate max possible uniform increase
    min_slack = 1.0
    
    # Check boundary slack
    for i in range(n):
        slack = min(
            centers[i, 0] - radii[i],
            1.0 - (centers[i, 0] + radii[i]),
            centers[i, 1] - radii[i],
            1.0 - (centers[i, 1] + radii[i])
        )
        min_slack = min(min_slack, slack)
        
    # Check pairwise slack
    for i in range(n):
        for j in range(i+1, n):
            slack = dists[i, j] - (radii[i] + radii[j])
            # Since we increase both, slack reduces by 2 * increase
            # slack_new = slack - 2*increase >= 0 => increase <= slack / 2
            min_slack = min(min_slack, slack / 2.0)
            
    if min_slack > 1e-7:
        # Increase radii
        increase = min(min_slack, step)
        radii += increase
        # Recalculate dists is not needed as centers didn't move, 
        # but slack calculation was based on old radii.
        
    # If we increased radii, we might now be touching boundaries or each other.
    # Push apart to resolve
    centers, radii = push_apart(centers, radii, get_constraint_matrix(centers), max_iter=20)
    
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal grid
    # We want to place 26 points roughly evenly.
    # A 5x5 grid is 25 points. We add 1.
    # Let's try a 6x5 arrangement (30 points) and pick 26 best?
    # Or just 5 rows.
    
    # Generate hexagonal grid
    rows = 6
    cols_per_row = []
    
    # Try to fit 26 points
    # Row pattern: 5, 4, 5, 4, 5, 3 -> Sum 26
    # Or 4, 5, 4, 5, 4, 4 -> Sum 26
    # Let's use a generic generator
    points = []
    x_min, x_max = 0.05, 0.95
    y_min, y_max = 0.05, 0.95
    
    # Simple grid init for robustness
    grid_x = np.linspace(x_min, x_max, 6)
    grid_y = np.linspace(y_min, y_max, 5)
    
    init_centers = []
    for y in grid_y:
        for x in grid_x:
            init_centers.append([x, y])
            
    # We have 30 points. Pick 26.
    # We can just take the first 26.
    init_centers = np.array(init_centers[:n])
    
    # Initialize radii small
    radii = np.full(n, 0.05)
    
    # 2. Optimization Loop
    # We will iteratively try to grow radii and resolve conflicts
    
    best_sum = 0
    best_centers = init_centers.copy()
    best_radii = radii.copy()
    
    # Coarse to fine step size
    step_sizes = [0.001, 0.0005, 0.0002, 0.0001, 0.00005]
    
    centers = init_centers.copy()
    current_radii = radii.copy()
    
    for step in step_sizes:
        for _ in range(100): # Iterations per step size
            # Try to increase all radii
            # Calculate max feasible increase
            dists = get_constraint_matrix(centers)
            max_increase = 1.0
            
            # Boundary constraints
            for i in range(n):
                slack = min(
                    centers[i, 0] - current_radii[i],
                    1.0 - (centers[i, 0] + current_radii[i]),
                    centers[i, 1] - current_radii[i],
                    1.0 - (centers[i, 1] + current_radii[i])
                )
                max_increase = min(max_increase, slack)
            
            # Pairwise constraints
            # dist >= r_i + r_j
            # If we increase both by delta, dist >= r_i + delta + r_j + delta
            # delta <= (dist - r_i - r_j) / 2
            for i in range(n):
                for j in range(i+1, n):
                    slack = dists[i, j] - current_radii[i] - current_radii[j]
                    max_increase = min(max_increase, slack / 2.0)
            
            if max_increase > 1e-7:
                current_radii += max_increase
                # After increasing, we are tight. Push to relax geometry.
                centers, current_radii = push_apart(centers, current_radii, get_constraint_matrix(centers), max_iter=10)
            
            # Check if valid and better
            if validate_packing(centers, current_radii):
                current_sum = np.sum(current_radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = current_radii.copy()
            else:
                # If invalid (due to numerical issues in push_apart), revert step or shrink slightly
                # This case shouldn't happen if push_apart is robust, but safety:
                current_radii -= max_increase * 1.1 
                centers, current_radii = push_apart(centers, current_radii, get_constraint_matrix(centers), max_iter=50)

    # Final refinement using scipy optimization on the best found config
    # This helps polish the result
    # Variables: x, y for all circles. Radii are fixed to the max found or optimized?
    # Actually, let's just optimize centers for the best_radii to ensure validity,
    # or try to squeeze a bit more radius.
    
    # Let's try a local optimization on the best config
    # Objective: Maximize sum of radii. 
    # But radii are dependent on positions. 
    # It's easier to maximize min_slack or just use the iterative result.
    
    # The iterative result should be quite good.
    # Let's return the best valid packing found.
    
    return best_centers, best_radii, best_sum
