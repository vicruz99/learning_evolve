# sol_000007 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ca1ebfe6) state=70120274 sum of radii=2.615898 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

def get_optimal_radii_and_sum(centers):
    """
    Solves the LP to find radii that maximize sum(r_i) for fixed centers.
    Constraints:
    1. r_i + r_j <= distance(centers[i], centers[j]) for all i < j
    2. r_i <= distance(centers[i], boundary) for all i
    3. r_i >= 0
    """
    n = centers.shape[0]
    
    # 1. Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    # Precompute max possible radius for each center based on boundaries
    # x in [0, 1], y in [0, 1]
    dist_left = centers[:, 0]
    dist_right = 1.0 - centers[:, 0]
    dist_bottom = centers[:, 1]
    dist_top = 1.0 - centers[:, 1]
    
    max_r_boundary = np.minimum(np.minimum(dist_left, dist_right), 
                                np.minimum(dist_bottom, dist_top))
    
    # 2. Pairwise constraints: r_i + r_j <= dist_ij
    # We need to construct the LP matrix.
    # Variables: r_0, r_1, ..., r_{n-1}
    # Objective: Maximize sum(r) => Minimize -sum(r)
    
    c_obj = -np.ones(n)
    
    # Constraints matrix A_ub x <= b_ub
    # We will collect constraints
    # Constraint 1: r_i <= max_r_boundary[i]  => 1*r_i <= max_r_boundary[i]
    # Constraint 2: r_i + r_j <= dist_ij
    
    # Number of constraints: n (boundary) + n*(n-1)/2 (pairs)
    n_pairs = n * (n - 1) // 2
    n_constraints = n + n_pairs
    
    # Initialize A_ub
    # Using list of lists for efficiency before converting to matrix
    A_ub_rows = []
    b_ub = []
    
    # Boundary constraints
    for i in range(n):
        row = np.zeros(n)
        row[i] = 1.0
        A_ub_rows.append(row)
        b_ub.append(max_r_boundary[i])
        
    # Pairwise constraints
    # Distance matrix
    # dist_matrix[i, j] = dist(centers[i], centers[j])
    # We only need upper triangle
    dists = cdist(centers, centers)
    
    for i in range(n):
        for j in range(i + 1, n):
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            A_ub_rows.append(row)
            b_ub.append(dists[i, j])
            
    A_ub = np.array(A_ub_rows)
    b_ub = np.array(b_ub)
    
    # Bounds for variables: r_i >= 0
    bounds = [(0, None)] * n
    
    # Solve LP
    # Method 'highs' is usually fast and robust
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        radii = res.x
        sum_radii = -res.fun # Because we minimized -sum
    else:
        # Fallback if LP fails (unlikely)
        radii = np.zeros(n)
        sum_radii = 0.0
        
    return radii, sum_radii

def compute_forces(centers, radii):
    """
    Computes repulsive forces based on tight constraints.
    """
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    # Precompute distances
    dists = cdist(centers, centers)
    
    # Threshold for "tight" constraint
    tol = 1e-4
    
    for i in range(n):
        # Boundary forces
        # If r_i is close to distance to boundary, push center away from boundary
        x, y = centers[i]
        r = radii[i]
        
        # Left boundary (x=0)
        if x - r < tol:
            forces[i, 0] += 1.0
        # Right boundary (x=1)
        if (1.0 - x) - r < tol:
            forces[i, 0] -= 1.0
        # Bottom boundary (y=0)
        if y - r < tol:
            forces[i, 1] += 1.0
        # Top boundary (y=1)
        if (1.0 - y) - r < tol:
            forces[i, 1] -= 1.0
            
        # Pairwise forces
        for j in range(n):
            if i == j:
                continue
            dist = dists[i, j]
            r_sum = radii[i] + radii[j]
            if dist - r_sum < tol and dist > 1e-9:
                # Repulsive force along vector c_i - c_j
                # Normalize vector
                vec = centers[i] - centers[j]
                # Avoid division by zero
                if dist > 1e-9:
                    forces[i] += vec / dist
                    
    return forces

def run_packing():
    # 1. Initialize centers on a hexagonal lattice
    n = 26
    centers = np.zeros((n, 2))
    
    # Try to fit 26 circles. 
    # Approximate rows: 5, 5, 5, 5, 6? Or 5, 6, 5, 6, 4?
    # Let's try a grid-like approach first to get spread out points
    # Then hexagonal refinement
    
    # Simple initialization: random points might get stuck, 
    # but a structured grid is better.
    # Let's place them in a 5x5 grid (25) + 1 in center?
    # Or just fill a rectangle.
    
    # Better: Hexagonal packing approximation
    # Rows of circles.
    # Let's determine row lengths.
    # 26 = 5 + 5 + 5 + 5 + 6 (sum 26)
    # Or 6 + 5 + 6 + 5 + 4?
    # Let's try 5 rows with roughly 5 circles each.
    # Rows 0, 2, 4 have 5 circles. Rows 1, 3 have 6 circles? No, 5+6+5+6+4 = 26.
    # Or 5, 5, 6, 5, 5 = 26.
    
    row_counts = [5, 6, 5, 6, 4] # Sum = 26. Wait 5+6+5+6+4 = 26.
    # Let's verify spacing.
    # Hexagonal spacing.
    # Width 1. Height 1.
    
    # Let's just generate a hexagonal grid and pick the first 26 points that fit.
    # Or simpler:
    # Grid 6x5 = 30 points. Remove 4.
    # Spacing dx = 1/7, dy = 1/6?
    
    # Let's use a more robust initialization:
    # Place centers on a perturbed grid.
    rows = 6
    cols = 5
    # 6*5 = 30. We need 26.
    # We will keep 26 points.
    
    # Generate grid points
    x_coords = np.linspace(0.1, 0.9, cols)
    y_coords = np.linspace(0.1, 0.9, rows)
    
    grid_points = []
    for y in y_coords:
        for x in x_coords:
            grid_points.append([x, y])
            
    # We have 30 points. Select 26.
    # Which ones to remove? Corners might be good to keep? 
    # Actually, centers should be away from boundaries.
    # Let's just take the first 26.
    centers = np.array(grid_points[:n])
    
    # Optimization loop
    # Parameters
    max_iter = 2000
    step_size_init = 0.05
    min_step_size = 1e-5
    
    best_sum_radii = 0.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    # Pre-allocate for speed? Not really needed for N=26
    
    for iteration in range(max_iter):
        # Decay step size
        step_size = step_size_init * (1.0 - iteration / max_iter)
        if step_size < min_step_size:
            step_size = min_step_size
            
        # 1. Compute optimal radii
        radii, current_sum = get_optimal_radii_and_sum(centers)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # 2. Compute forces
        forces = compute_forces(centers, radii)
        
        # 3. Update centers
        # Center of mass force to keep inside?
        # Forces are already pushing away from boundaries if tight.
        # But we might want a slight pull towards center to avoid getting stuck at edges?
        # No, boundary forces handle it.
        
        centers = centers + step_size * forces
        
        # Project back to [0, 1] strictly? 
        # Centers must be valid. But radii check handles boundary.
        # However, if center goes outside, radius becomes 0 or invalid.
        # Let's clamp centers to a safe range [1e-5, 1-1e-5]
        # Actually, centers can be on boundary, but then radius is 0.
        # It's better to keep centers strictly inside to allow non-zero radii.
        centers = np.clip(centers, 1e-4, 1.0 - 1e-4)
        
    # Final local search: try perturbing each center individually
    # To escape local minima
    improvement = True
    while improvement:
        improvement = False
        radii, current_sum = get_optimal_radii_and_sum(centers)
        
        # Try moving each circle
        for i in range(n):
            current_c = centers[i]
            # Directions: 4 neighbors + diagonals?
            directions = [
                [0.01, 0], [-0.01, 0], [0, 0.01], [0, -0.01],
                [0.007, 0.007], [-0.007, 0.007], [0.007, -0.007], [-0.007, -0.007]
            ]
            
            best_local_sum = current_sum
            best_local_c = current_c
            
            for d in directions:
                trial_c = current_c + d
                # Clamp
                trial_c = np.clip(trial_c, 1e-4, 1.0 - 1e-4)
                
                # Check if valid movement (not too close to others? handled by LP)
                # Temporarily move
                old_c = centers[i]
                centers[i] = trial_c
                
                _, trial_sum = get_optimal_radii_and_sum(centers)
                
                if trial_sum > best_local_sum:
                    best_local_sum = trial_sum
                    best_local_c = trial_c
                    
                centers[i] = old_c # Revert
            
            if best_local_sum > current_sum:
                centers[i] = best_local_c
                improvement = True
                current_sum = best_local_sum # Update current sum for next iteration?
                # Actually need to recompute radii for whole system
                radii, current_sum = get_optimal_radii_and_sum(centers)

    # Final check and cleanup
    radii, final_sum = get_optimal_radii_and_sum(centers)
    
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        # Fallback to best found
        centers = best_centers
        radii = best_radii
        final_sum = best_sum_radii

    return centers, radii, final_sum

if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(f"Centers:\n{c}")
    # print(f"Radii:\n{r}")
