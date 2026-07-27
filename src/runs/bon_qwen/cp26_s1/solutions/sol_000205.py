# sol_000205 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4a6b07ba) state=299a53e6 sum of radii=0.668757 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Optimizes circle packing in a unit square to maximize sum of radii.
    Uses a force-directed simulation with growing radii and multiple restarts.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # Parameters for simulation
    num_restarts = 20
    steps_per_restart = 2500
    lr_center = 0.005
    lr_radius = 0.0004
    stiffness = 800.0
    
    for restart_idx in range(num_restarts):
        # Initialize centers randomly in the interior to avoid immediate boundary issues
        # Range [0.2, 0.8]
        centers = np.random.rand(n, 2) * 0.6 + 0.2
        # Start with small radii
        radii = np.full(n, 0.02)
        
        for step in range(steps_per_restart):
            # 1. Calculate Boundary Forces
            # Repel from walls if circle extends beyond [0, 1]
            force = np.zeros_like(centers)
            
            # x-direction
            # Left wall: x < r -> push right
            force[:, 0] += np.maximum(0, radii - centers[:, 0]) * 1000.0
            # Right wall: x > 1 - r -> push left
            force[:, 0] -= np.maximum(0, radii - (1.0 - centers[:, 0])) * 1000.0
            
            # y-direction
            # Bottom wall: y < r -> push up
            force[:, 1] += np.maximum(0, radii - centers[:, 1]) * 1000.0
            # Top wall: y > 1 - r -> push down
            force[:, 1] -= np.maximum(0, radii - (1.0 - centers[:, 1])) * 1000.0
            
            # 2. Calculate Pairwise Repulsive Forces
            # Vectorized computation of distances and directions
            diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
            dists = np.linalg.norm(diffs, axis=2) # (n, n)
            
            # Avoid division by zero
            safe_dists = np.where(dists > 1e-9, dists, 1e-9)
            directions = diffs / safe_dists[:, :, np.newaxis] # (n, n, 2)
            
            # Calculate overlap amount
            sum_radii_mat = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlaps = np.maximum(0, sum_radii_mat - dists)
            np.fill_diagonal(overlaps, 0.0) # No self-overlap
            
            # Apply repulsive forces proportional to overlap
            # Force direction is away from neighbor (diff direction)
            pairwise_force = directions * overlaps[:, :, np.newaxis] * stiffness
            net_pairwise_force = np.sum(pairwise_force, axis=1)
            
            force += net_pairwise_force
            
            # 3. Update Centers
            # Apply force with learning rate, limiting max displacement for stability
            displacement = np.clip(force * lr_center, -0.05, 0.05)
            centers += displacement
            
            # Keep centers within [0, 1] bounds strictly
            centers = np.clip(centers, 0.0, 1.0)
            
            # 4. Grow Radii
            # Slowly increase radii to push system towards optimal packing density
            radii += lr_radius
            radii = np.minimum(radii, 0.5) # Max possible radius is 0.5
            
        # Post-processing: Ensure strict validity
        # 1. Boundary constraints: r <= min(x, 1-x, y, 1-y)
        max_r_boundary = np.minimum(
            np.minimum(centers[:, 0], 1.0 - centers[:, 0]),
            np.minimum(centers[:, 1], 1.0 - centers[:, 1])
        )
        radii = np.minimum(radii, max_r_boundary)
        radii = np.maximum(radii, 0.0)
        
        # 2. Overlap constraints: dist >= r_i + r_j
        # If overlaps exist, shrink radii minimally to resolve them
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        sum_radii_mat = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlaps = np.maximum(0, sum_radii_mat - dists)
        np.fill_diagonal(overlaps, 0.0)
        
        max_overlap = np.max(overlaps)
        if max_overlap > 1e-9:
            # Shrink all radii by half the max overlap to resolve
            shrink_amount = max_overlap / 2.0 + 1e-7
            radii -= shrink_amount
            radii = np.maximum(radii, 0.0)
            # Re-check boundary after shrink (shrinking r only helps boundary)
            radii = np.minimum(radii, max_r_boundary)
        
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
    return best_centers, best_radii, best_sum
