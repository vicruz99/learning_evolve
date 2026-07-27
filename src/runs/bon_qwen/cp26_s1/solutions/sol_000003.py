# sol_000003 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6882cd8b) state=29c01aba sum of radii=2.374077 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import warnings

# Suppress warnings from scipy for cleaner output
warnings.filterwarnings('ignore')

def solve_radii_lp(centers):
    """
    Given centers, solve the LP to find radii that maximize sum(r_i).
    Constraints:
    1. r_i <= dist(center_i, boundary)
    2. r_i + r_j <= dist(center_i, center_j)
    3. r_i >= 0
    """
    n = centers.shape[0]
    
    # Distance to boundaries
    # x - r >= 0 => r <= x
    # 1 - x - r >= 0 => r <= 1 - x
    # Same for y
    dists_to_boundary = np.minimum(
        np.minimum(centers[:, 0], 1 - centers[:, 0]),
        np.minimum(centers[:, 1], 1 - centers[:, 1])
    )
    
    # Distance matrix between centers
    # Compute pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # LP Setup
    # Variables: r_1, ..., r_n
    # Objective: Max sum(r_i) => Min -sum(r_i)
    c_obj = -np.ones(n)
    
    # Constraints A_ub @ r <= b_ub
    # 1. r_i <= dists_to_boundary[i]
    #    Diagonal matrix
    A_ub_boundary = np.eye(n)
    b_ub_boundary = dists_to_boundary
    
    # 2. r_i + r_j <= dists[i, j] for i < j
    #    Each constraint is a row with 1 at i and 1 at j
    num_pairs = n * (n - 1) // 2
    A_ub_pairs = np.zeros((num_pairs, n))
    b_ub_pairs = np.zeros(num_pairs)
    
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub_pairs[idx, i] = 1.0
            A_ub_pairs[idx, j] = 1.0
            b_ub_pairs[idx] = dists[i, j]
            idx += 1
            
    # Combine constraints
    A_ub = np.vstack([A_ub_boundary, A_ub_pairs])
    b_ub = np.concatenate([b_ub_boundary, b_ub_pairs])
    
    # Bounds for r_i: [0, infinity)
    bounds = [(0, None) for _ in range(n)]
    
    # Solve LP
    # Using 'highs' method which is robust and fast
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            radii = res.x
            sum_radii = -res.fun
            return radii, sum_radii
        else:
            # Fallback if LP fails (rare)
            # Return small radii
            radii = np.ones(n) * 1e-5
            return radii, np.sum(radii)
    except Exception:
        radii = np.ones(n) * 1e-5
        return radii, np.sum(radii)

def get_force_vector(centers, radii):
    """
    Compute a heuristic force vector to push circles apart.
    We want to increase distances between circles to allow larger radii.
    Force on circle i is sum of vectors (c_i - c_j) / dist if touching/overlapping.
    Actually, we just want to maximize the 'clearance'.
    A simple gradient ascent on the minimum clearance or similar.
    But simpler: just random perturbation search is often effective for low N.
    Here we return 0 and rely on the optimizer or random search outside.
    """
    return np.zeros_like(centers)

def run_packing():
    """
    Main function to pack 26 circles.
    """
    n_circles = 26
    
    # Strategy: Start with a hexagonal-like grid, then optimize centers.
    
    # 1. Initial Configuration
    # A hexagonal packing is denser than square packing.
    # We try to fit rows.
    # Let's try to arrange them in rows with alternating offsets.
    # Approximate radius for 26 circles in square is around 0.105.
    # Diameter ~ 0.21.
    # 5 circles fit in width 1 (5 * 0.21 = 1.05 > 1). So maybe 4 or 5 per row.
    # Let's try 5 rows.
    # Row 1: 5 circles
    # Row 2: 5 circles (shifted)
    # Row 3: 5 circles
    # Row 4: 5 circles
    # Row 5: 6 circles? Or 5, 5, 5, 5, 6?
    # 26 circles. 5x5=25. Need 1 more.
    # Maybe 6, 5, 5, 5, 5?
    
    # Let's generate a grid of points and then optimize.
    # We can place them in a grid and let the optimizer move them.
    # 5x5 grid is too regular.
    # Let's use a perturbed grid.
    
    np.random.seed(42) # For reproducibility
    
    # Initial centers: 5 rows, varying columns
    # Try to space them roughly evenly
    centers = np.zeros((n_circles, 2))
    
    # Heuristic placement
    # 6 circles in first row, 5 in second, 5 in third, 5 in fourth, 5 in fifth?
    # Total 26.
    # But vertical space might be tight.
    # Let's just do a dense random pack or grid.
    # Grid 5x5 + 1.
    
    # Let's create a 5x6 grid and pick 26 points? No, 5x6=30.
    # Let's place points on a triangular lattice.
    # Spacing dx, dy.
    # dy = sqrt(3)/2 * dx.
    # Fit in 1x1.
    # Try to fit roughly 26 points.
    
    # Let's just use a randomized grid initialization
    # 5 rows
    rows = 5
    cols = 6 # 5 rows * 5 cols = 25. Maybe 6 cols in some rows?
    
    # Simple initialization: uniform grid with noise
    # 26 points. sqrt(26) approx 5.1.
    # Let's do a 6x5 grid (30 points) and remove 4?
    # Or just random valid points.
    
    # Better: Hexagonal lattice points inside square.
    points = []
    # Grid spacing
    spacing = 0.2 
    for r in range(7): # rows
        y = r * spacing * np.sqrt(3)/2 + 0.1 # offset
        for c in range(7): # cols
            x = c * spacing + (r % 2) * (spacing / 2) + 0.1
            if 0 <= x <= 1 and 0 <= y <= 1:
                points.append([x, y])
    
    # Shuffle and pick 26
    np.random.shuffle(points)
    if len(points) > n_circles:
        centers = np.array(points[:n_circles])
    else:
        # Fallback to random
        centers = np.random.rand(n_circles, 2)
        
    # Ensure centers are strictly inside to allow some radius
    centers = centers * 0.9 + 0.05 
    
    # 2. Optimization Loop
    # We will use a simple hill climbing with random perturbations.
    # At each step, solve LP for radii.
    # Try to move centers to improve sum_radii.
    
    current_radii, current_sum = solve_radii_lp(centers)
    best_centers = centers.copy()
    best_radii = current_radii.copy()
    best_sum = current_sum
    
    # Parameters for search
    step_size = 0.02
    iterations = 2000 # Number of attempts
    
    for k in range(iterations):
        # Decay step size
        current_step = step_size * (1.0 - k / iterations)
        if current_step < 1e-6:
            current_step = 1e-6
            
        # Pick a random circle to move
        idx = np.random.randint(0, n_circles)
        
        # Save old position
        old_pos = centers[idx].copy()
        
        # Perturb
        # Try moving in random direction
        delta = np.random.randn(2) * current_step
        new_pos = old_pos + delta
        
        # Check bounds (keep strictly inside to allow radius > 0)
        # Clamp to [0.01, 0.99] roughly? 
        # Actually LP handles boundary constraints, but center must be valid.
        # If center is at 0, radius must be 0.
        # So keep centers in [0, 1].
        new_pos = np.clip(new_pos, 0.0, 1.0)
        
        centers[idx] = new_pos
        
        # Solve LP
        new_radii, new_sum = solve_radii_lp(centers)
        
        # Accept if better
        if new_sum > best_sum:
            best_sum = new_sum
            best_centers = centers.copy()
            best_radii = new_radii.copy()
        else:
            # Revert if worse (simple hill climbing)
            # Or maybe accept with prob? 
            # Let's stick to hill climbing for stability, 
            # but maybe add some random restarts or simulated annealing if stuck.
            # For now, revert.
            centers[idx] = old_pos
            
        # Occasionally try moving multiple circles or larger steps?
        # Or restart if stuck?
        if k % 100 == 0 and k > 0:
             # Small random restart of one circle to escape local optima
             idx = np.random.randint(0, n_circles)
             centers[idx] = np.random.rand(2)
             # Recalculate
             current_radii, current_sum = solve_radii_lp(centers)
             if current_sum > best_sum:
                 best_sum = current_sum
                 best_centers = centers.copy()
                 best_radii = current_radii.copy()
                 
    # 3. Final Polish
    # The hill climbing might leave some circles suboptimal.
    # We can try a local optimization using scipy minimize on the centers
    # to maximize sum_radii (which is a black box function of centers).
    # But since sum_radii is non-smooth, derivative-free methods are needed.
    # Let's try Nelder-Mead on the best_centers found.
    
    # Define objective for minimize (negative sum)
    def objective(centers_flat):
        centers_2d = centers_flat.reshape(-1, 2)
        _, s = solve_radii_lp(centers_2d)
        return -s

    # Flatten best centers
    x0 = best_centers.flatten()
    
    # Use a local optimizer
    # Nelder-Mead is good for non-smooth functions
    try:
        res_opt = opt.minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-9})
        if res_opt.success or (-res_opt.fun > best_sum):
            best_centers = res_opt.x.reshape(-1, 2)
            best_radii, best_sum = solve_radii_lp(best_centers)
    except Exception:
        pass
        
    # Final validation and return
    # Ensure radii and centers are valid
    # The LP ensures validity, but let's double check with the validation logic provided
    # Actually we just return.
    
    return best_centers, best_radii, float(best_sum)
