# sol_000192 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9dd6f42d) state=a27d4a88 sum of radii=2.253006 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Strategy: 
    1. Initialize centers in a perturbed hexagonal grid.
    2. Use a force-directed simulation to resolve overlaps and grow radii.
    3. Refine using local optimization (Nelder-Mead) to maximize the minimum clearance.
    """
    n = 26
    rng = np.random.default_rng(42)
    
    # --- 1. Initialization ---
    # Estimate radius to build the grid. 
    # Known optimal r for 26 circles is approx 0.1014.
    est_r = 0.105 
    
    centers = []
    # Hexagonal spacing
    row_spacing = est_r * np.sqrt(3)
    col_spacing = est_r * 2.0
    
    row_idx = 0
    # Generate enough points to fill the square
    max_rows = 10
    while len(centers) < n + 10 and row_idx < max_rows:
        y_center = est_r + row_idx * row_spacing
        # Offset odd rows to create hexagonal pattern
        offset = (est_r * 0.5) if row_idx % 2 == 1 else 0.0
        x_center = est_r + offset
        
        while x_center <= 1.0 - est_r:
            centers.append([x_center, y_center])
            x_center += col_spacing
            if len(centers) > n + 10:
                break
        row_idx += 1
    
    # Select first 26 points
    centers = np.array(centers[:n])
    
    # Add small random perturbation to avoid symmetry traps
    centers += rng.normal(0, 0.005, size=centers.shape)
    
    # Clip to valid range [0, 1] initially
    centers = np.clip(centers, 0.0, 1.0)

    # --- 2. Force-Directed Simulation (Inflation & Resolution) ---
    # We simulate a system where circles repel each other and walls.
    # We gradually increase the target radius to pack them tighter.
    
    current_r = 0.05
    radii = np.ones(n) * current_r
    velocities = np.zeros_like(centers)
    
    # Simulation parameters
    repulsion_k = 2.0
    damping = 0.85
    step_count = 3000
    
    for step in range(step_count):
        # Slowly increase target radius
        # We aim to reach around 0.101
        target_r = min(0.108, current_r + 0.0001)
        delta_r = target_r - current_r
        
        # Update radii
        current_r += delta_r
        radii[:] = current_r
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Circle-Circle repulsion (Vectorized)
        # diffs[i, j] = centers[i] - centers[j]
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Mask diagonal (self-distance) to infinity
        np.fill_diagonal(dists, np.inf)
        
        # Minimum required distance for non-overlap
        min_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount (positive if overlapping)
        overlaps = min_dist - dists
        overlaps = np.maximum(0, overlaps)
        
        # Force direction: normalized difference vector
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        normals = diffs / safe_dists[:, :, np.newaxis]
        
        # Force magnitude proportional to overlap
        force_contribs = overlaps[:, :, np.newaxis] * repulsion_k * normals
        
        # Sum forces on each particle (axis 1 sums over j)
        forces += np.sum(force_contribs, axis=1)
        
        # Boundary repulsion
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Left wall (x < r) -> push right (+)
        left_penetration = radii - x
        forces[:, 0] += np.maximum(0, left_penetration) * repulsion_k * 2.0
        
        # Right wall (x > 1-r) -> push left (-)
        right_penetration = x - (1.0 - radii)
        forces[:, 0] -= np.maximum(0, right_penetration) * repulsion_k * 2.0
        
        # Bottom wall (y < r) -> push up (+)
        bottom_penetration = radii - y
        forces[:, 1] += np.maximum(0, bottom_penetration) * repulsion_k * 2.0
        
        # Top wall (y > 1-r) -> push down (-)
        top_penetration = y - (1.0 - radii)
        forces[:, 1] -= np.maximum(0, top_penetration) * repulsion_k * 2.0
        
        # Update velocities and positions
        velocities = velocities * damping + forces * 0.1
        centers += velocities * 0.5
        
        # Safety clip to keep inside square
        centers = np.clip(centers, 0.0, 1.0)

    # --- 3. Local Optimization ---
    # Refine the position to maximize the minimum clearance (radius).
    # Objective: Maximize r such that all circles fit.
    # Equivalent to maximizing min(min_pairwise_dist/2, min_boundary_dist).
    
    def objective_func(X_flat):
        X = X_flat.reshape(n, 2)
        
        # Pairwise distances
        diffs = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = dists.min()
        
        # Boundary distances
        min_x = X[:, 0].min()
        max_x = X[:, 0].max()
        min_y = X[:, 1].min()
        max_y = X[:, 1].max()
        
        min_boundary_dist = min(min_x, 1.0 - max_x, min_y, 1.0 - max_y)
        
        # The feasible radius is limited by both
        feasible_r = min(min_pair_dist / 2.0, min_boundary_dist)
        
        # We want to maximize feasible_r
        return -feasible_r

    # Flatten current centers
    X0 = centers.flatten()
    
    # Try to optimize
    try:
        res = minimize(objective_func, X0, method='Nelder-Mead', 
                       options={'xatol': 1e-7, 'fatol': 1e-7, 'maxiter': 2000})
        if res.success and res.fun < 0: 
            centers_opt = res.x.reshape(n, 2)
            best_r = -res.fun
            if best_r > 0:
                centers = centers_opt
                radii = np.ones(n) * best_r
    except Exception:
        pass

    # Final calculation of max r
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = dists.min()
    
    min_x = centers[:, 0].min()
    max_x = centers[:, 0].max()
    min_y = centers[:, 1].min()
    max_y = centers[:, 1].max()
    min_boundary_dist = min(min_x, 1.0 - max_x, min_y, 1.0 - max_y)
    
    final_r = min(min_pair_dist / 2.0, min_boundary_dist)
    final_r = max(0.0, final_r)
    
    radii = np.ones(n) * final_r
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
