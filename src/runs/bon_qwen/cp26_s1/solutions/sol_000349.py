# sol_000349 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1c8d6f7d) state=f164ee47 sum of radii=0.520000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a physics-based simulation with repulsion forces and radius inflation.
    """
    n = 26
    
    # Use a fixed seed for reproducibility, but random start helps escape local optima
    rng = np.random.default_rng(42)
    
    # Initialize centers randomly within a safe margin from edges
    # Margin 0.1 ensures initial radius 0.02 fits
    centers = rng.uniform(0.1, 0.9, size=(n, 2))
    
    # Initialize radii small
    radii = np.full(n, 0.02)
    
    # Simulation parameters
    dt = 0.002          # Time step
    k_rep = 100.0       # Repulsion stiffness between circles
    k_wall = 200.0      # Wall stiffness
    inflation_rate = 0.0001 # Amount to increase radius per step when stable
    max_iters = 8000    # Total simulation steps
    
    # Preallocate force array
    forces = np.zeros((n, 2))
    
    # Track max radius reached to ensure progress
    max_sum_radii = 0.0
    
    for step in range(max_iters):
        # 1. Compute Pairwise Forces (Repulsion)
        # Vectorized difference matrix: diff[i, j] = center_i - center_j
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        
        # Euclidean distances
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Avoid division by zero for normalization
        dists_safe = np.where(dists > 1e-9, dists, 1e-9)
        
        # Calculate overlaps: (r_i + r_j) - dist
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dists
        
        # Force magnitude is proportional to overlap, but only if overlap > 0
        # Using a soft threshold or just max(0, overlap)
        force_mag = k_rep * np.maximum(overlap, 0)
        
        # Unit vectors pointing from j to i
        dirs = diff / dists_safe[:, :, np.newaxis]
        
        # Force on i from j is in direction (i - j) i.e., dirs[i, j]
        # Sum forces from all j
        forces_pair = force_mag[:, :, np.newaxis] * dirs
        forces = np.sum(forces_pair, axis=1)
        
        # 2. Compute Wall Forces
        # If circle i is too close to left wall (x < r_i), push right (+x)
        # Overlap amount: r_i - x_i
        overlap_left = radii - centers[:, 0]
        forces[:, 0] += k_wall * np.maximum(overlap_left, 0)
        
        # Right wall (x > 1 - r_i), push left (-x)
        # Overlap amount: x_i - (1 - r_i) -> force is negative
        overlap_right = centers[:, 0] - (1 - radii)
        forces[:, 0] -= k_wall * np.maximum(overlap_right, 0)
        
        # Bottom wall (y < r_i), push up (+y)
        overlap_bottom = radii - centers[:, 1]
        forces[:, 1] += k_wall * np.maximum(overlap_bottom, 0)
        
        # Top wall (y > 1 - r_i), push down (-y)
        overlap_top = centers[:, 1] - (1 - radii)
        forces[:, 1] -= k_wall * np.maximum(overlap_top, 0)
        
        # 3. Update Centers
        centers += forces * dt
        
        # Clip positions to stay strictly within [0, 1] to prevent numerical drift
        centers = np.clip(centers, 0.0, 1.0)
        
        # 4. Inflation (Increase Radii)
        # We increase radii only if the system is relatively stable (low overlaps)
        # This allows the circles to expand into available space
        current_max_overlap = np.max(overlap)
        
        # Adaptive inflation: 
        # If max overlap is very small, we are in a valid packing, expand.
        # If overlap is significant, wait for relaxation.
        if current_max_overlap < 1e-6:
            radii += inflation_rate
            
            # Optional: Slow down inflation as radii get larger to allow finer packing
            # But constant rate is simpler and often effective.
            
        # Safety check to prevent radii from becoming too large (max possible is 0.5)
        radii = np.minimum(radii, 0.49)
        
        # Check validity occasionally (not strictly necessary in loop but good for debugging)
        # sum_radii = np.sum(radii)
        
    # Final cleanup: Ensure no slight overlaps due to numerical precision
    # Run a few extra steps with 0 inflation to settle
    radii_prev = radii.copy()
    for _ in range(500):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        dists_safe = np.where(dists > 1e-9, dists, 1e-9)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dists
        force_mag = k_rep * np.maximum(overlap, 0)
        dirs = diff / dists_safe[:, :, np.newaxis]
        forces_pair = force_mag[:, :, np.newaxis] * dirs
        forces = np.sum(forces_pair, axis=1)
        
        # Wall forces
        forces[:, 0] += k_wall * np.maximum(radii - centers[:, 0], 0)
        forces[:, 0] -= k_wall * np.maximum(centers[:, 0] - (1 - radii), 0)
        forces[:, 1] += k_wall * np.maximum(radii - centers[:, 1], 0)
        forces[:, 1] -= k_wall * np.maximum(centers[:, 1] - (1 - radii), 0)
        
        centers += forces * dt * 0.1 # Smaller step for fine tuning
        centers = np.clip(centers, 0.0, 1.0)
        
        # If stable, try a tiny bit of expansion
        if np.max(overlap) < 1e-7:
            radii += 1e-5

    # Final validation check within function logic (mental check)
    # The returned arrays must be valid.
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
