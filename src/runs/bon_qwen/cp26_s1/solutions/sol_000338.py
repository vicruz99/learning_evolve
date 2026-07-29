# sol_000338 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 29661f66) state=656f3550 sum of radii=2.426156 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed simulation to spread points and maximize minimum distances.
    """
    n = 26
    np.random.seed(42)
    
    # 1. Initialization: Hexagonal Grid
    # We generate a hexagonal lattice and select points that fit well, or just perturb a grid.
    # A 6x6 grid has 36 points. We can pick 26 or start with 5x5=25 and add one.
    # Let's create a hexagonal pattern explicitly.
    # Hexagonal packing: rows shifted by 0.5 spacing.
    
    centers = []
    # Try to fit points in a hexagonal pattern. 
    # Spacing 's' roughly 1/sqrt(n) ~ 0.2.
    # Let's just place them randomly in a grid first and let forces organize them.
    # Actually, a dense initial configuration helps finding the global optimum faster?
    # No, spread out is better to avoid local minima of clumps.
    # But we need them inside [0,1].
    
    # Strategy: 5 rows. 6, 5, 6, 5, 4 points? Sum = 26.
    # Or 5, 5, 5, 5, 6?
    # Let's try a random initialization near a grid to break symmetry.
    
    # 5x5 grid (25 points) + 1 point in center?
    # Grid points: 0.1, 0.3, 0.5, 0.7, 0.9
    # If we use 0.1 to 0.9, spacing 0.2.
    # Add point at 0.5, 0.5? No, that's already there.
    # Add point at 0.25, 0.25?
    
    # Let's just use a randomized grid.
    grid_x = np.linspace(0.1, 0.9, 6) # 6 points
    grid_y = np.linspace(0.1, 0.9, 5) # 5 points
    points = []
    for y in grid_y:
        for x in grid_x:
            points.append([x, y])
    # This gives 30 points. We need 26.
    # Let's remove 4 random points or just pick 26.
    # But 30 points in 1x1 with spacing 0.2 fits (diameter 0.2, radius 0.1).
    # Actually 6 points of width 0.2 take 1.0 width? 
    # 0.1 + 5*0.2 = 1.1 > 1.0. So 6 points of diameter 0.2 don't fit.
    # Max radius for 6 points in a row is 1/12 ~ 0.083.
    # For 5 points, 1/10 = 0.1.
    # So a grid with 6 columns forces small radii.
    # We prefer 5 columns.
    
    # Let's generate points based on 5 columns.
    # 5 columns, 5 rows = 25.
    # We need 1 more.
    # Let's add a point in a gap.
    
    # Initialize with 25 points in 5x5 grid, spaced to allow r=0.1
    # Centers at 0.1, 0.3, 0.5, 0.7, 0.9
    x_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    y_coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    initial_points = []
    for y in y_coords:
        for x in x_coords:
            initial_points.append([x, y])
            
    # Add 26th point. Where?
    # Maybe slightly perturbed to start optimization.
    # Let's add a point near the center of a gap, e.g., (0.2, 0.2) -> overlap.
    # Better: Perturb all points slightly and add a random point.
    
    centers = np.array(initial_points, dtype=float)
    # Add one random point
    extra_point = np.random.uniform(0.05, 0.95, 2)
    centers = np.vstack([centers, extra_point])
    
    # Add small random noise to break symmetry
    centers += np.random.normal(0, 0.001, centers.shape)
    
    # Optimization Parameters
    n_iterations = 5000
    # Initial step size (learning rate)
    step_size = 0.01 
    # Decay for step size
    decay = 0.9995
    
    # To track best solution
    best_sum_radii = -1.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)

    for t in range(n_iterations):
        # Compute pairwise distances
        # dists[i, j] = distance between i and j
        # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Ensure diagonal is inf so min doesn't pick self
        np.fill_diagonal(dists, np.inf)
        
        # Compute max possible radius for each circle
        # Constraint 1: Distance to nearest neighbor / 2
        min_dists_to_neighbors = np.min(dists, axis=1)
        radii_from_neighbors = min_dists_to_neighbors / 2.0
        
        # Constraint 2: Distance to boundary
        # x >= r, x <= 1-r => r <= x and r <= 1-x
        # y >= r, y <= 1-r => r <= y and r <= 1-y
        dist_to_left = centers[:, 0]
        dist_to_right = 1.0 - centers[:, 0]
        dist_to_bottom = centers[:, 1]
        dist_to_top = 1.0 - centers[:, 1]
        
        dist_to_boundary = np.minimum.reduce([dist_to_left, dist_to_right, dist_to_bottom, dist_to_top])
        
        # The valid radius is the minimum of these constraints
        radii = np.minimum(radii_from_neighbors, dist_to_boundary)
        
        # Ensure radii are non-negative
        radii = np.maximum(radii, 0.0)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute Forces
        forces = np.zeros_like(centers)
        
        # 1. Repulsion from neighbors
        # We want to increase distance between points to increase radii.
        # Force proportional to 1/d^2 (Coulomb-like) or just push apart if close.
        # Since we want to maximize radius, we care most about the tightest constraints.
        # Let's apply a repulsive force between all pairs, stronger when closer.
        
        # Vectorized force computation
        # Force on i due to j: F_ij = (pos_i - pos_j) / dist_ij^3 * strength
        # But we need to be careful with 0 distance.
        
        # Mask for distinct pairs
        mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(mask, False)
        
        # Compute unit vectors
        # Avoid division by zero
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        unit_vectors = diff / safe_dists[:, :, np.newaxis]
        
        # Force magnitude: 1 / dist^2 is standard repulsion
        # We scale it by current radii sum or something?
        # Let's use a constant strength that decays?
        # Or strength based on how much we can expand?
        # Simple repulsion works well to distribute points.
        force_magnitude = 1.0 / (safe_dists**2)
        
        # Accumulate forces
        # forces[i] += sum_j (unit_vector_ij * force_mag_ij)
        # Note: diff[i,j] = pos_i - pos_j. So vector points i -> j? No, i - j points from j to i.
        # Wait, diff = centers[i] - centers[j]. So vector from j to i.
        # Repulsion on i should be in direction (i - j). Correct.
        
        # Sum over j
        # forces += np.sum(unit_vectors * force_magnitude[:, :, np.newaxis], axis=1)
        # This is O(N^2) which is fine for N=26.
        
        forces += np.sum(unit_vectors * force_magnitude[:, :, np.newaxis], axis=1)
        
        # 2. Boundary Forces
        # Push away from walls.
        # If x is close to 0, push right (+x). If x close to 1, push left (-x).
        # Force ~ 1/x or similar.
        # But we must respect the fact that if x < r, we are violating constraint.
        # However, our radius calculation respects constraints, so x >= r is guaranteed?
        # Wait, radii is calculated based on current positions.
        # So x >= radii is true.
        # But we want to move points to allow LARGER radii.
        # If x is small, r is limited by x. To increase r, we must increase x.
        # So force should push towards center (0.5).
        
        # Simple force: F = 0.5 - pos (bias to center)
        # But this is weak.
        # Let's use a repulsion from walls.
        # Wall at 0: Force +1/x. Wall at 1: Force -1/(1-x).
        
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Avoid singularity
        eps = 1e-4
        fx = 1.0 / (x + eps) - 1.0 / (1.0 - x + eps)
        fy = 1.0 / (y + eps) - 1.0 / (1.0 - y + eps)
        
        forces[:, 0] += fx * 0.5 # Scale factor
        forces[:, 1] += fy * 0.5
        
        # Normalize forces or clamp?
        # Forces can be large.
        # Apply forces with step size
        centers += forces * step_size
        
        # Clamp to valid range [r, 1-r] is hard because r changes.
        # Clamp to [0, 1] strictly.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Decay step size
        step_size *= decay
        
    # Final calculation of radii for best_centers
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dists = np.min(dists, axis=1)
    
    dist_left = best_centers[:, 0]
    dist_right = 1.0 - best_centers[:, 0]
    dist_bottom = best_centers[:, 1]
    dist_top = 1.0 - best_centers[:, 1]
    dist_boundary = np.min(np.stack([dist_left, dist_right, dist_bottom, dist_top], axis=1), axis=1)
    
    radii = np.minimum(min_dists / 2.0, dist_boundary)
    radii = np.maximum(radii, 0.0) # Safety
    
    sum_radii = np.sum(radii)
    
    # Validation check (internal)
    # The logic ensures validity, but let's ensure no NaN
    if np.any(np.isnan(best_centers)) or np.any(np.isnan(radii)):
        # Fallback to grid if optimization failed
        x = np.linspace(0.1, 0.9, 5)
        y = np.linspace(0.1, 0.9, 5)
        centers = np.array([(xx, yy) for xx in x for yy in y])
        # Add one more
        centers = np.vstack([centers, [0.5, 0.5]]) # Overlap, but fallback
        # Just return grid for 25 + 1 tiny?
        # Actually, if optimization fails, return something valid.
        # But with seed 42 it should work.
        pass

    return best_centers, radii, sum_radii
