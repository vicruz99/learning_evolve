# sol_000140 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4c86e033) state=8275c659 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    Uses a growing circle simulation with repulsive forces.
    """
    n = 26
    # Parameters
    initial_radius = 0.01
    radius_step = 5e-6
    max_radius = 0.102  # Slightly above expected optimum 0.1014
    steps_per_growth = 500  # Relaxation steps per radius increase
    dt = 0.005
    damping = 0.9
    repulsion_strength = 100.0
    
    # Initialize centers randomly in the valid region
    # Use a small buffer to ensure they start inside
    buffer = 0.1
    centers = np.random.uniform(
        low=buffer, 
        high=1 - buffer, 
        size=(n, 2)
    )
    
    velocities = np.zeros((n, 2))
    radius = initial_radius
    
    # Precompute indices for distance matrix calculation to save time
    i_indices, j_indices = np.triu_indices(n, k=1)
    
    # Main simulation loop
    # We try to grow radius up to max_radius
    # The loop runs for a fixed number of total iterations or until radius stabilizes
    total_iterations = 200000 # Max limit to prevent hanging
    
    current_radius = initial_radius
    
    # Helper to compute forces
    def compute_forces(centers, r):
        forces = np.zeros((n, 2))
        
        # 1. Boundary forces
        # If center is too close to boundary (dist < r), push in
        # Left wall (x=0)
        dist_to_left = centers[:, 0] - r
        mask_left = dist_to_left < 0
        forces[mask_left, 0] += repulsion_strength * dist_to_left[mask_left] # Negative dist -> positive force
        
        # Right wall (x=1)
        dist_to_right = (1.0 - r) - centers[:, 0]
        mask_right = dist_to_right < 0
        forces[mask_right, 0] += repulsion_strength * dist_to_right[mask_right] # Negative dist -> negative force
        
        # Bottom wall (y=0)
        dist_to_bottom = centers[:, 1] - r
        mask_bottom = dist_to_bottom < 0
        forces[mask_bottom, 1] += repulsion_strength * dist_to_bottom[mask_bottom]
        
        # Top wall (y=1)
        dist_to_top = (1.0 - r) - centers[:, 1]
        mask_top = dist_to_top < 0
        forces[mask_top, 1] += repulsion_strength * dist_to_top[mask_top]
        
        # 2. Inter-circle repulsion
        # Vectorized distance calculation for pairs
        # Only compute for i < j
        diffs = centers[i_indices] - centers[j_indices] # Shape (M, 2)
        dists = np.linalg.norm(diffs, axis=1) # Shape (M,)
        
        # Check overlaps: dist < 2r
        overlap = 2 * r - dists
        # Only apply force if overlapping (overlap > 0)
        # We use a smooth ramp or direct force
        # Force magnitude proportional to overlap
        # Direction: away from each other
        # Normalize vector. Handle zero distance (rare)
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        unit_vecs = diffs / safe_dists[:, np.newaxis]
        
        # Apply force only if overlap > 0
        # To make it smoother, maybe use max(0, overlap)
        mag = np.maximum(0, overlap) * repulsion_strength
        
        # Add forces to respective circles
        # i indices get force in direction of unit_vec (push away from j)
        # j indices get force in direction of -unit_vec (push away from i)
        np.add.at(forces, i_indices, unit_vecs * mag[:, np.newaxis])
        np.add.at(forces, j_indices, -unit_vecs * mag[:, np.newaxis])
        
        return forces

    # Run simulation
    # We will loop, growing radius gradually
    # But to ensure convergence, we should only grow radius if overlaps are minimal
    # Or just grow and let forces resolve. Growing too fast causes chaos.
    
    # Better strategy: 
    # Loop:
    #   Run relaxation steps to clear overlaps for current radius
    #   Try to increase radius slightly
    #   If overlaps persist too much, maybe back off? 
    #   But for this problem, just pushing them apart usually works.
    
    # Let's try a simpler loop: just integrate dynamics while slowly increasing radius target.
    
    # Reset for clean start with better init?
    # Let's try a hexagonal grid init for better convergence
    # Rows of 6, 5, 6, 5, 4? Sum=26.
    # Or just random is fine with enough steps.
    
    # Optimization: Use a fixed schedule for radius growth
    # We want to reach radius ~ 0.1014
    # Let's just run a long simulation with growing radius.
    
    r_target = initial_radius
    t = 0
    
    # To speed up, we can reduce steps_per_growth if things are stable
    # But for now, fixed steps.
    
    # We will run for a max number of steps.
    max_steps = 150000
    
    for step in range(max_steps):
        # Update target radius slowly
        if r_target < max_radius:
            r_target += radius_step
            
        # Compute forces based on current radius target
        # Actually, forces should react to current radius r_target
        forces = compute_forces(centers, r_target)
        
        # Update velocities
        velocities += forces * dt
        velocities *= damping
        
        # Update centers
        centers += velocities * dt
        
        # Clamp centers to [0, 1] to prevent escape (though forces push back)
        centers = np.clip(centers, 0.0, 1.0)
        
        # Check if we can stop?
        # Maybe if radius is high and velocities are low?
        # But we don't know if it's stuck.
        
        t += 1
        
    # Final radius is the target we reached, but we should verify validity
    # The circles might have slight overlaps due to dynamics.
    # Let's perform a final relaxation pass to clean up.
    # And determine the actual valid radius.
    
    # Clean up: run optimization to maximize min distance / boundary clearance
    # We can use a simple gradient ascent on the minimum clearance
    # Or just report the r_target if valid.
    
    # Let's verify validity and maybe shrink r slightly if needed.
    # But the prompt asks to maximize sum of radii.
    # The simulation should have found a packing for r_target.
    # However, due to numerical errors, r_target might be slightly too large.
    # We can check the minimum distance between centers and boundaries.
    
    # Compute min clearance
    # Boundary clearance
    min_clearance = np.min(np.array([
        centers[:, 0], 
        1 - centers[:, 0], 
        centers[:, 1], 
        1 - centers[:, 1]
    ]))
    
    # Pairwise clearance (dist / 2)
    diffs = centers[i_indices] - centers[j_indices]
    dists = np.linalg.norm(diffs, axis=1)
    min_pair_dist = np.min(dists)
    
    # The valid radius is min(min_clearance, min_pair_dist / 2)
    # But the simulation used r_target.
    # If r_target > valid_radius, circles overlap.
    # We should return the valid radius.
    
    valid_r = min(min_clearance, min_pair_dist / 2.0)
    
    # If valid_r is much smaller than r_target, the packing failed or got stuck.
    # But with 150k steps, it should be close.
    # Let's clamp valid_r to r_target just in case (numerical noise might make valid_r slightly larger? No).
    # Actually valid_r should be <= r_target.
    
    # To be safe, let's use the computed valid_r.
    # But wait, if valid_r is small, we failed.
    # Let's assume the simulation worked.
    
    # Refine: The forces push circles apart. If they are packed, dists >= 2*r_target.
    # So min_pair_dist should be >= 2*r_target.
    # If min_pair_dist < 2*r_target, they are overlapping.
    # In that case, the effective radius is min_pair_dist / 2.
    
    # Let's re-run a quick local optimization to fix any residual overlaps and maximize r.
    # But given the constraints, just returning the centers with radius = min_clearance might be too small if min_pair_dist is the bottleneck.
    # Actually radius is limited by the tightest constraint.
    # r = min( min_clearance, min_pair_dist / 2 )
    
    # However, the objective is sum of radii. If all circles have same radius r, sum = 26*r.
    # If we can have different radii, maybe better?
    # But for this target, equal radii is likely optimal.
    
    # Let's return the configuration with the maximum valid equal radius.
    final_r = min(min_clearance, min_pair_dist / 2.0)
    
    # Ensure non-negative
    final_r = max(0.0, final_r)
    
    radii = np.full(n, final_r)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
