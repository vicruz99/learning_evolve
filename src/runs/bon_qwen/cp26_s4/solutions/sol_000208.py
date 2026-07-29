# sol_000208 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1af9ddd7) state=855a830a sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed layout with iterative inflation.
    """
    n = 26
    np.random.seed(42)  # For reproducibility

    # 1. Initialization
    # Start with a grid layout to ensure good initial distribution
    # 5x5 grid gives 25 circles, we place the 26th near the center or corner
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)  # Initial small radii

    # Grid placement
    grid_size = 5
    step = 1.0 / (grid_size + 1)
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            centers[idx, 0] = step * (c + 1)
            centers[idx, 1] = step * (r + 1)
            idx += 1
    
    # Place the 26th circle in a gap or random position
    # A 5x5 grid leaves gaps. Let's place it slightly perturbed in center
    if n > idx:
        centers[idx] = [0.5, 0.5] # Center might be crowded, let's try a corner gap
        centers[idx] = [0.05, 0.05] 

    # Add slight random perturbation to break symmetry
    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    
    # Clip initial centers to valid range [0, 1]
    centers = np.clip(centers, 0.0, 1.0)

    # 2. Optimization Parameters
    dt = 0.002          # Time step for physics simulation
    damping = 0.9       # Velocity damping
    growth_rate = 1.5e-4 # Rate at which radii grow
    repulsion_k = 200.0 # Strength of repulsion force
    wall_k = 300.0      # Strength of wall repulsion
    num_iters = 8000    # Number of simulation steps
    
    velocities = np.zeros_like(centers)
    
    # Pre-allocate arrays for performance
    # centers shape (n, 2)
    # radii shape (n,)
    
    for t in range(num_iters):
        # --- Inflation Step ---
        # Increase radii. Decrease rate over time to settle.
        current_growth = growth_rate * (1.0 - t / num_iters) # Linear decay
        radii += current_growth
        
        # --- Force Calculation ---
        forces = np.zeros_like(centers)
        
        # Pairwise Repulsion (Vectorized)
        # Compute distance matrix (n, n)
        # diffs[i, j] = centers[j] - centers[i]
        diffs = centers[np.newaxis, :, :] - centers[:, np.newaxis, :] # (n, n, 2)
        
        # Squared distances
        sq_dists = np.sum(diffs**2, axis=2)
        dists = np.sqrt(sq_dists)
        
        # Radii sums
        rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount: positive if overlapping
        overlap = rad_sums - dists
        np.fill_diagonal(overlap, 0) # No self-overlap
        
        # Mask for overlapping pairs
        is_overlapping = overlap > 1e-9
        
        # Compute direction vectors (unit vectors)
        # Avoid division by zero
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        dir_vecs = diffs / safe_dists[:, :, np.newaxis]
        
        # Force magnitude proportional to overlap
        force_magnitudes = repulsion_k * np.where(is_overlapping, overlap, 0.0)
        
        # Accumulate forces
        # forces[i] -= sum_j (force_mag_ij * dir_vec_ij)
        # Note: dir_vecs[i,j] points from i to j.
        # If i overlaps j, i should be pushed away from j (opposite to dir_vecs[i,j])
        # So force on i is -force_mag * dir_vecs[i,j]
        
        # Sum over axis 1 (j)
        forces -= np.sum(force_magnitudes[:, :, np.newaxis] * dir_vecs, axis=1)
        
        # --- Wall Repulsion ---
        # Left wall: if x < r, push right
        # Right wall: if x > 1-r, push left
        # Bottom wall: if y < r, push up
        # Top wall: if y > 1-r, push down
        
        # Left
        penetration_x_neg = np.maximum(0, radii - centers[:, 0])
        forces[:, 0] += wall_k * penetration_x_neg
        
        # Right
        penetration_x_pos = np.maximum(0, centers[:, 0] + radii - 1.0)
        forces[:, 0] -= wall_k * penetration_x_pos
        
        # Bottom
        penetration_y_neg = np.maximum(0, radii - centers[:, 1])
        forces[:, 1] += wall_k * penetration_y_neg
        
        # Top
        penetration_y_pos = np.maximum(0, centers[:, 1] + radii - 1.0)
        forces[:, 1] -= wall_k * penetration_y_pos
        
        # --- Update Positions ---
        velocities += forces * dt
        velocities *= damping
        centers += velocities * dt
        
        # Clamp centers to [0, 1] to prevent escaping simulation box
        # Although wall forces should keep them in, clipping is a safety measure.
        # Note: Clipping to [0,1] allows centers to be slightly invalid (x < r),
        # but the wall force in next step will correct it.
        centers = np.clip(centers, 0.0, 1.0)
        
        # Occasionally add noise to escape local minima
        if t % 100 == 0 and t < num_iters * 0.8:
            velocities += np.random.normal(0, 0.001, size=velocities.shape)

    # Final cleanup: ensure radii are valid (non-negative)
    radii = np.maximum(radii, 0.0)
    
    # Ensure centers are valid (clip just in case)
    centers = np.clip(centers, 0.0, 1.0)
    
    # Compute sum of radii
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To ensure no closure issues as per rules, helper functions are top level if needed.
# In this case, everything is inside run_packing or standard libs.
# But the rules say "Make all helper functions top level".
# I have no helper functions defined outside.
# I will verify constraints before returning if needed, but the physics should handle it.

# Double check constraints logic for safety
def validate_and_fix(centers, radii):
    """Simple pass to fix minor numerical violations"""
    n = centers.shape[0]
    # Fix radii < 0
    radii = np.maximum(radii, 0.0)
    # Fix centers out of bounds
    centers = np.clip(centers, 0.0, 1.0)
    # Fix circles sticking out of bounds by moving centers in
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1.0 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1.0 - r)
        
        # If radius is too large for square (r > 0.5), shrink it
        if r > 0.5:
            radii[i] = 0.5
            
    # Fix overlaps by slightly shrinking radii if necessary (last resort)
    # Check overlaps
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
            min_dist = radii[i] + radii[j]
            if dist < min_dist - 1e-9:
                # Overlap detected, reduce radii proportionally to fit
                # This is a simple fix, optimization should have handled it
                factor = dist / min_dist
                # Reduce slightly more to be safe
                factor *= 0.999 
                radii[i] *= np.sqrt(factor) # Approximate adjustment
                radii[j] *= np.sqrt(factor)
                
    return centers, radii

# Wrap the function to include validation
def run_packing():
    centers, radii, sum_radii = _run_optimization()
    centers, radii = validate_and_fix(centers, radii)
    return centers, radii, np.sum(radii)

def _run_optimization():
    n = 26
    np.random.seed(42)

    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05)

    grid_size = 5
    step = 1.0 / (grid_size + 1)
    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            centers[idx, 0] = step * (c + 1)
            centers[idx, 1] = step * (r + 1)
            idx += 1
    
    if n > idx:
        centers[idx] = [0.05, 0.05] 

    centers += np.random.uniform(-0.01, 0.01, size=centers.shape)
    centers = np.clip(centers, 0.0, 1.0)

    dt = 0.002
    damping = 0.9
    growth_rate = 1.5e-4
    repulsion_k = 200.0
    wall_k = 300.0
    num_iters = 8000
    
    velocities = np.zeros_like(centers)
    
    for t in range(num_iters):
        current_growth = growth_rate * (1.0 - t / num_iters)
        radii += current_growth
        
        forces = np.zeros_like(centers)
        
        diffs = centers[np.newaxis, :, :] - centers[:, np.newaxis, :] 
        sq_dists = np.sum(diffs**2, axis=2)
        dists = np.sqrt(sq_dists)
        
        rad_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = rad_sums - dists
        np.fill_diagonal(overlap, 0)
        
        is_overlapping = overlap > 1e-9
        safe_dists = np.where(dists > 1e-9, dists, 1.0)
        dir_vecs = diffs / safe_dists[:, :, np.newaxis]
        
        force_magnitudes = repulsion_k * np.where(is_overlapping, overlap, 0.0)
        forces -= np.sum(force_magnitudes[:, :, np.newaxis] * dir_vecs, axis=1)
        
        penetration_x_neg = np.maximum(0, radii - centers[:, 0])
        forces[:, 0] += wall_k * penetration_x_neg
        penetration_x_pos = np.maximum(0, centers[:, 0] + radii - 1.0)
        forces[:, 0] -= wall_k * penetration_x_pos
        penetration_y_neg = np.maximum(0, radii - centers[:, 1])
        forces[:, 1] += wall_k * penetration_y_neg
        penetration_y_pos = np.maximum(0, centers[:, 1] + radii - 1.0)
        forces[:, 1] -= wall_k * penetration_y_pos
        
        velocities += forces * dt
        velocities *= damping
        centers += velocities * dt
        centers = np.clip(centers, 0.0, 1.0)
        
        if t % 100 == 0 and t < num_iters * 0.8:
            velocities += np.random.normal(0, 0.001, size=velocities.shape)

    return centers, radii, np.sum(radii)
