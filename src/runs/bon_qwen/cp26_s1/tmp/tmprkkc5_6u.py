import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Returns optimal centers, radii, and sum of radii for 26 circles in a unit square.
    """
    n = 26
    
    # --- Phase 1: Initialization ---
    # Use a hexagonal grid pattern for a high-quality starting point.
    centers = []
    y = 0.05  # Initial row y
    while len(centers) < n:
        row_shift = (len(centers) % 2) * 0.05
        x = 0.05 + row_shift
        while x < 0.95 and len(centers) < n:
            centers.append([x, y])
            x += 0.1  # Initial spacing
        y += 0.087 # Vertical spacing for hex packing approx
    centers = np.array(centers[:n])
    radii = np.full(n, 0.05)

    # --- Phase 2 & 3: Force-directed relaxation and Radius Expansion ---
    # We will simulate repulsion forces to push circles apart while growing them.
    # This allows us to find a dense packing where the sum of radii is maximized.
    
    # Parameters for simulation
    repulsion_strength = 10.0
    damping = 0.5
    growth_rate = 0.0001
    
    # Number of optimization steps
    n_steps = 2000
    
    for step in range(n_steps):
        # Gradually increase radii
        radii += growth_rate
        
        # Compute repulsion forces between all pairs
        # Also compute boundary forces to keep circles inside [0, 1]
        forces = np.zeros_like(centers)
        
        # Pairwise repulsion
        # Vectorized calculation for efficiency
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        
        # Avoid division by zero and self-interaction
        np.fill_diagonal(dists, np.inf)
        
        # Calculate overlap depth
        overlaps = radii[:, np.newaxis] + radii[np.newaxis, :] - dists
        overlaps = np.maximum(overlaps, 0)
        
        # Force direction (normalized difference vector)
        # Handle zero distance case to avoid NaNs
        dirs = np.where(dists[:, :, np.newaxis] == 0, 0, diffs / dists[:, :, np.newaxis])
        
        # Apply repulsive force proportional to overlap
        pair_forces = overlaps[:, :, np.newaxis] * dirs * repulsion_strength
        
        # Sum forces for each circle (symmetry handles double counting naturally in sum)
        forces += pair_forces.sum(axis=1)
        
        # Boundary forces
        # Push away from x=0
        for i in range(n):
            if centers[i, 0] < radii[i]:
                forces[i, 0] += (radii[i] - centers[i, 0]) * repulsion_strength
            if centers[i, 0] > 1 - radii[i]:
                forces[i, 0] -= (centers[i, 0] - (1 - radii[i])) * repulsion_strength
            if centers[i, 1] < radii[i]:
                forces[i, 1] += (radii[i] - centers[i, 1]) * repulsion_strength
            if centers[i, 1] > 1 - radii[i]:
                forces[i, 1] -= (centers[i, 1] - (1 - radii[i])) * repulsion_strength

        # Update centers
        centers += forces * 0.05 * (1 - step / n_steps) # Decrease step size over time
        
        # Hard clip to ensure validity during simulation
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1 - radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1 - radii[i])

    # --- Phase 4: Final Refinement (Unequal Radii) ---
    # Allow radii to vary slightly to maximize sum
    # Simple local search: try increasing each radius individually
    for _ in range(100):
        for i in range(n):
            current_r = radii[i]
            # Try to increase radius
            test_r = current_r + 0.0001
            if test_r > 0.5: continue # Safety cap
            
            # Check boundaries
            if centers[i, 0] - test_r < 0 or centers[i, 0] + test_r > 1: continue
            if centers[i, 1] - test_r < 0 or centers[i, 1] + test_r > 1: continue
            
            # Check overlaps
            valid = True
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < test_r + radii[j] - 1e-9:
                    valid = False
                    break
            
            if valid:
                radii[i] = test_r

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii