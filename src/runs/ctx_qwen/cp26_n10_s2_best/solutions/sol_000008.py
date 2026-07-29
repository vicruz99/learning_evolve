# sol_000008 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state accdaaf6) state=1470ff70 sum of radii=1.950000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def solve_radii_lp(centers):
    """
    Given fixed centers, solve the LP to maximize sum of radii.
    Variables: r_0, ..., r_{n-1}
    Maximize: sum(r_i)
    Subject to:
      r_i + r_j <= distance(centers[i], centers[j]) for all i < j
      0 <= r_i <= centers[i][0]
      0 <= r_i <= 1 - centers[i][0]
      0 <= r_i <= centers[i][1]
      0 <= r_i <= 1 - centers[i][1]
    """
    n = centers.shape[0]
    if n == 0:
        return np.array([])

    # Variables: r_0, ..., r_{n-1}
    # Objective: maximize sum(r) => minimize -sum(r)
    c_obj = -np.ones(n)

    # Inequality constraints: A_ub @ x <= b_ub
    # 1. Pairwise constraints: r_i + r_j <= dist_ij
    A_ub = []
    b_ub = []
    
    # Precompute distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centers[i] - centers[j])
            dists[i, j] = d
            dists[j, i] = d
            
    # Add pairwise constraints
    # For each pair (i, j), row with 1 at i, 1 at j
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    row_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dists[i, j]
            row_idx += 1

    # 2. Boundary constraints: r_i <= x_i, r_i <= 1-x_i, etc.
    # These are upper bounds on variables, can be handled by bounds in linprog
    # Or added to A_ub. Bounds is cleaner.
    
    # Bounds for each r_i: (0, min_boundary_dist)
    bounds = []
    for i in range(n):
        x, y = centers[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        # Clamp to 0 if center is outside (shouldn't happen if valid centers)
        max_r = max(0.0, max_r)
        bounds.append((0.0, max_r))

    # Solve LP
    # method='highs' is usually robust and fast
    try:
        res = opt.linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
        else:
            # Fallback to small radii if LP fails
            return np.zeros(n)
    except Exception:
        return np.zeros(n)

def update_centers_repulsion(centers, radii, dt=0.01, iterations=10):
    """
    Simple force-directed relaxation to push overlapping or touching circles apart.
    """
    n = centers.shape[0]
    for _ in range(iterations):
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        for i in range(n):
            for j in range(i + 1, n):
                vec = centers[j] - centers[i]
                dist = np.linalg.norm(vec)
                if dist < 1e-9:
                    # Prevent division by zero, push random direction
                    vec = np.random.rand(2) * 0.01
                    dist = np.linalg.norm(vec)
                
                required_dist = radii[i] + radii[j]
                
                # If touching or overlapping, apply force
                # Force magnitude proportional to overlap
                overlap = required_dist - dist
                if overlap > 0:
                    # Stronger force for larger overlap
                    # Also apply some force even if just touching to explore?
                    # Let's stick to strict overlap resolution + slight repulsion for tight fits
                    # Actually, to increase radii, we want to increase distance even if not overlapping
                    # but the LP handles the radii. Here we just want to move centers to allow larger radii.
                    # A good heuristic is to push apart if dist <= required_dist + epsilon
                    force_mag = max(overlap, 0.0) # Only push if overlapping?
                    # Actually, for optimization, we might want to push if they are "constraining" each other.
                    # But let's keep it simple: resolve overlaps.
                    
                    # Wait, if we only resolve overlaps, we might get stuck.
                    # We need to increase distances to allow larger radii.
                    # Let's apply a repulsive force based on inverse distance squared, scaled by radii sum?
                    # Or just push if dist < required_dist + margin.
                    pass

                # Let's use a generic repulsion that favors separation
                # Force ~ (r_i + r_j) / dist^2
                # But this might push them out of bounds.
                
                # Better: If dist < r_i + r_j + margin, push.
                margin = 0.0 # We want to resolve overlaps first
                if dist < radii[i] + radii[j] + 1e-5:
                     force_mag = (radii[i] + radii[j] - dist) * 10.0 # Stiff spring
                     dir_vec = vec / dist
                     forces[i] -= dir_vec * force_mag
                     forces[j] += dir_vec * force_mag

        # Apply forces
        centers += forces * dt
        
        # Project back to [0,1] x [0,1] with margin?
        # Centers must be such that r_i >= 0.
        # But here we just move centers. Radii will be re-calculated.
        # We must ensure centers stay in [0,1].
        centers = np.clip(centers, 0.0, 1.0)
        
        # Also, ensure centers are not exactly on boundary if we want space?
        # Actually, centers can be on boundary if radius is 0.
        
    return centers

def generate_hexagonal_grid(n_circles):
    """
    Generate a hexagonal grid of points inside [0,1]x[0,1].
    """
    # Try to fit points with some spacing
    # Approximate spacing s = 1/sqrt(n)
    s = 0.15 
    
    points = []
    y = 0.1 # Start slightly inside
    row_idx = 0
    
    while y <= 0.9:
        x = 0.1
        offset = (row_idx % 2) * (s / 2)
        if offset > 0:
            x = 0.1 + offset
            
        while x <= 0.9:
            points.append([x, y])
            x += s
        y += s * math.sqrt(3) / 2
        row_idx += 1
        
    # If we have too few, reduce spacing
    if len(points) < n_circles:
        # Retry with smaller spacing
        s = 0.1
        points = []
        y = 0.05
        row_idx = 0
        while y <= 0.95:
            x = 0.05
            offset = (row_idx % 2) * (s / 2)
            if offset > 0:
                x = 0.05 + offset
            while x <= 0.95:
                points.append([x, y])
                x += s
            y += s * math.sqrt(3) / 2
            row_idx += 1

    points = np.array(points[:n_circles])
    return points

def run_packing():
    n = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Run multiple restarts
    num_restarts = 20
    
    for restart in range(num_restarts):
        # 1. Initialize centers
        if restart == 0:
            # Hexagonal grid
            centers = generate_hexagonal_grid(n)
            if centers.shape[0] < n:
                # Fill remaining with random
                missing = n - centers.shape[0]
                random_centers = np.random.rand(missing, 2)
                centers = np.vstack([centers, random_centers])
        else:
            # Random initialization with some structure (grid perturbation)
            centers = np.random.rand(n, 2)
            # Sort to avoid clustering? No, just random.
            # Maybe start with a grid and perturb
            grid_y, grid_x = np.mgrid[0:1:6j, 0:1:5j] # 30 points
            grid = np.column_stack((grid_x.flatten(), grid_y.flatten()))
            # Shuffle and take 26
            np.random.shuffle(grid)
            centers = grid[:n]
            # Add noise
            centers += np.random.normal(0, 0.01, centers.shape)
            centers = np.clip(centers, 0, 1)

        # 2. Iterative optimization
        # We iterate between solving LP for radii and updating centers
        for step in range(50): # 50 iterations of refinement
            # Solve for radii
            radii = solve_radii_lp(centers)
            
            # Check if valid
            # The LP ensures constraints are met for these centers
            # But we need to check if radii are reasonable
            current_sum = np.sum(radii)
            
            if current_sum > best_sum:
                best_sum = current_sum
                best_centers = centers.copy()
                best_radii = radii.copy()
            
            # Update centers to potentially improve sum
            # We want to increase distances between circles that are limiting radii
            # Heuristic: Repulsion proportional to radii sum
            dt = 0.05 * (1.0 / (step + 1)) # Decay step size
            
            forces = np.zeros_like(centers)
            for i in range(n):
                for j in range(i + 1, n):
                    vec = centers[j] - centers[i]
                    dist = np.linalg.norm(vec)
                    if dist < 1e-9:
                        dist = 1e-9
                        vec = np.random.rand(2) * 0.01 # random push
                    
                    # We want to increase dist if it's close to r_i + r_j
                    # Specifically, if r_i + r_j >= dist, they are touching/overlapping
                    # But LP ensures r_i + r_j <= dist.
                    # The constraint is tight if r_i + r_j approx dist.
                    # To increase r_i or r_j, we need to increase dist.
                    # So apply repulsion if dist is small relative to radii?
                    # Or just always repel?
                    
                    # Force: F ~ 1/dist^2
                    # Scale by radii to emphasize larger circles?
                    force_mag = 1.0 / (dist * dist)
                    
                    # Direction
                    dir_vec = vec / dist
                    
                    forces[i] -= dir_vec * force_mag
                    forces[j] += dir_vec * force_mag
            
            # Boundary repulsion (keep away from walls to allow radius)
            # If center is close to 0, push right. If close to 1, push left.
            # Radius is limited by distance to boundary.
            # To maximize radius, center should be far from boundary?
            # Actually, r <= min(x, 1-x). Max r is at x=0.5.
            # So push centers towards 0.5?
            # But they also need space from each other.
            
            # Let's just use pairwise repulsion.
            
            centers += forces * dt
            
            # Clip to [0, 1]
            # But strictly, center must be >= r. 
            # If we clip, r might become invalid in next step, but LP handles it.
            # However, if center is at 0, r must be 0.
            # It's okay to be at 0.
            centers = np.clip(centers, 0.0, 1.0)
            
            # Ensure distinctness?
            # LP handles overlaps by reducing radii.
            
        # After iterations, do a final solve
        radii = solve_radii_lp(centers)
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()

    # Final validation check
    # Just in case numerical issues occurred
    # Re-run LP on best centers to ensure consistency
    final_radii = solve_radii_lp(best_centers)
    final_sum = np.sum(final_radii)
    
    # If final_radii are NaN or invalid, fallback
    if np.isnan(final_radii).any():
        final_radii = np.zeros(n)
        final_sum = 0.0
        
    return best_centers, final_radii, final_sum

# Helper to ensure top level functions only
if __name__ == "__main__":
    # This block is for local testing if needed, but run_packing is the entry point.
    pass
