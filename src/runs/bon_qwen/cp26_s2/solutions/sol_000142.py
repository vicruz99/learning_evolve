# sol_000142 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 24d569ae) state=567c19d4 sum of radii=1.294450 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns centers, radii, and sum_of_radii for 26 circles in a unit square.
    """
    n = 26
    
    # --- Helper Functions ---
    
    def min_pairwise_dist(centers):
        """Calculate the minimum distance between any pair of centers."""
        # Vectorized distance calculation
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        # Mask diagonal (distance to self is 0)
        np.fill_diagonal(dists, np.inf)
        return np.min(dists)

    def boundary_clearance(centers, r):
        """Calculate minimum distance of any circle to the boundary."""
        # Distance to left/right walls
        dx = np.minimum(centers[:, 0] - r, 1.0 - (centers[:, 0] + r))
        # Distance to bottom/top walls
        dy = np.minimum(centers[:, 1] - r, 1.0 - (centers[:, 1] + r))
        return np.minimum(dx, dy).min()

    def objective(coords_flat):
        """
        Objective for scipy.optimize.
        We want to maximize the minimum distance between points.
        Here we minimize the negative of the min distance.
        """
        centers = coords_flat.reshape(-1, 2)
        dist = min_pairwise_dist(centers)
        return -dist

    def bounds_func():
        """Return bounds for center coordinates [0, 1]."""
        return [(0, 1)] * (2 * n)

    # --- Initialization: Hexagonal Lattice ---
    
    # We attempt to place points on a hexagonal grid.
    # For N=26, a 5x5 grid is too small (25 points).
    # A hexagonal packing allows more density.
    # Let's try to fit points in a pattern.
    # A common pattern for N=26 might be roughly 6 rows or a distorted 5x5.
    
    # Start with a dense random perturbation of a grid to break symmetry
    # Grid 6x5 = 30 points, take first 26? Or just random.
    # Better: Generate hex grid points and keep first 26 that fit.
    
    r_init = 0.05 # Initial radius for layout
    centers = []
    
    # Hexagonal grid generation
    # Row spacing: sqrt(3) * diameter = sqrt(3) * 0.1 = 0.1732
    # Col spacing: diameter = 0.1
    # We want to fill the square.
    
    # Let's just use a good heuristic: 5 rows, shifting alternate rows
    # We want 26 circles. 5, 6, 5, 6, 4? or 6, 5, 6, 5, 4?
    # Let's try to pack them densely in a loop
    
    current_y = r_init
    row_idx = 0
    while len(centers) < n:
        row_x = r_init
        if row_idx % 2 == 1:
            row_x += r_init # Shift by radius (half diameter) for hex packing
        
        while len(centers) < n:
            if row_x + r_init <= 1.0:
                centers.append([row_x, current_y])
                row_x += 2 * r_init
            else:
                break # Row full
        
        current_y += np.sqrt(3) * r_init
        if current_y + r_init > 1.0:
            # If we run out of vertical space but need more circles, 
            # reduce r_init or change strategy. 
            # But for N=26, this should fit.
            pass
        row_idx += 1

    centers = np.array(centers[:n])
    
    # Random perturbation to avoid grid locking
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    # Clip to valid range
    centers = np.clip(centers, 0, 1)

    # --- Stage 1: Optimization using Scipy ---
    
    # We optimize the positions to maximize the minimum distance.
    # The max radius will be min_dist / 2.
    # However, boundary constraints also matter.
    # Let's maximize a composite score: min(min_dist, 2*boundary_dist)
    # Actually, simply maximizing min_dist often pushes points to center, 
    # ignoring boundaries. We need to respect boundaries.
    # Let's add a penalty for being too close to boundaries in the objective 
    # or just constrain bounds in scipy.
    # But scipy bounds are per variable. We can use bounds [r, 1-r] but r is unknown.
    # Let's use bounds [0, 1] and penalize boundary violation in objective.
    
    def objective_with_boundary(coords_flat):
        centers = coords_flat.reshape(-1, 2)
        
        # Pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        # Boundary distances (assuming radius is half of min_dist? No, radius is variable)
        # To maximize radius r, we need min_dist >= 2r and boundary_dist >= r.
        # So r <= min_dist / 2 and r <= boundary_dist.
        # Thus r = min(min_dist/2, min_boundary_dist).
        # We want to maximize this r.
        
        # Boundary clearance: distance from center to nearest wall
        # Wall is at 0 or 1. Center x must be in [r, 1-r].
        # Clearance is min(x, 1-x).
        clear_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        clear_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        min_clearance = np.minimum(clear_x, clear_y).min()
        
        r_est = min(min_dist / 2.0, min_clearance)
        return -r_est

    # Run optimization multiple times to avoid local minima
    best_r = 0
    best_centers = centers
    
    for attempt in range(5):
        # Add small noise
        x0 = centers.flatten() + np.random.uniform(-0.02, 0.02, 2*n)
        x0 = np.clip(x0, 0, 1)
        
        res = minimize(objective_with_boundary, x0, method='L-BFGS-B', 
                       bounds=bounds_func(), options={'maxiter': 2000})
        
        if res.fun < -best_r: # Minimizing negative, so more negative is better
            best_r = -res.fun
            best_centers = res.x.reshape(-1, 2)

    # --- Stage 2: Force-Directed Refinement ---
    
    # The scipy solution might be good, but force-directed can often squeeze 
    # more density by handling the "hard" constraints more physically.
    # We assume equal radii r = best_r.
    # We try to increase r slightly and relax.
    
    r_current = best_r
    # Safety margin for numerical issues in validation
    r_current = max(r_current, 0.01) 
    
    # Try to increase radius
    # We will simulate a system where circles repel if dist < 2r
    # and are pushed away from walls if dist < r
    
    centers = best_centers.copy()
    # Randomize slightly to break symmetries found by scipy
    centers += np.random.uniform(-0.001, 0.001, centers.shape)
    centers = np.clip(centers, 0.001, 0.999)

    # Parameters
    steps = 2000
    dt = 0.01
    k_repulse = 100.0
    k_wall = 50.0
    damping = 0.9
    
    velocities = np.zeros_like(centers)
    
    # We try to find a stable radius higher than best_r
    target_r = best_r * 1.01 # Try to increase by 1%
    
    # We can binary search or just try to relax at a fixed r
    # Let's try to relax at a sequence of increasing radii
    
    r_try = best_r
    improved = True
    while improved:
        improved = False
        # Try to pack at r_try
        for step in range(500):
            forces = np.zeros_like(centers)
            
            # 1. Pairwise repulsion
            # Vectorized pairwise diffs
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            
            # Overlap amount
            overlap = np.maximum(0, 2 * r_try - dists)
            # Force direction: normalize diff
            # Avoid division by zero
            safe_dists = np.where(dists < 1e-9, 1e-9, dists)
            norm_diff = diff / safe_dists[:, :, np.newaxis]
            
            # Sum forces
            # force_ij = k * overlap * direction
            # This is O(N^2)
            repulsive_forces = k_repulse * overlap[:, :, np.newaxis] * norm_diff
            forces += np.sum(repulsive_forces, axis=1)
            
            # 2. Wall repulsion
            # Left wall
            mask_left = centers[:, 0] < r_try
            forces[mask_left, 0] += k_wall * (r_try - centers[mask_left, 0])
            # Right wall
            mask_right = centers[:, 0] > 1.0 - r_try
            forces[mask_right, 0] -= k_wall * (centers[mask_right, 0] - (1.0 - r_try))
            # Bottom wall
            mask_bottom = centers[:, 1] < r_try
            forces[mask_bottom, 1] += k_wall * (r_try - centers[mask_bottom, 1])
            # Top wall
            mask_top = centers[:, 1] > 1.0 - r_try
            forces[mask_top, 1] -= k_wall * (centers[mask_top, 1] - (1.0 - r_try))
            
            # Update velocities
            velocities = velocities * damping + forces * dt
            centers += velocities * dt
            
            # Clamp
            centers = np.clip(centers, 0.0, 1.0)
            
        # Check if valid
        # Re-calculate min distance and clearance
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_d = np.min(dists)
        
        clear_x = np.minimum(centers[:, 0], 1.0 - centers[:, 0])
        clear_y = np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        min_c = np.minimum(clear_x, clear_y).min()
        
        achievable_r = min(min_d / 2.0, min_c)
        
        if achievable_r >= r_try - 1e-5:
            # Success, try slightly larger
            r_try = r_try + (achievable_r - r_try) * 0.5 + 0.0001
            improved = True
            best_r = achievable_r
            best_centers = centers.copy()
        else:
            # Failed, decrease r_try
            r_try = (r_try + best_r) / 2
            if r_try - best_r < 1e-6:
                break

    # Final radius
    final_r = best_r
    final_centers = best_centers
    
    # Ensure validity and clip
    final_centers = np.clip(final_centers, final_r, 1.0 - final_r)
    
    # Radii array
    radii = np.full(n, final_r)
    
    sum_radii = np.sum(radii)
    
    return final_centers, radii, sum_radii

# Validation check (not part of returned code, but for internal logic)
# The function returns the required tuple.
