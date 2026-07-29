# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 68244382) state=e26426a0 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii using 
    a physics-based growing circles simulation.
    """
    n_circles = 26
    
    # --- 1. Initialization ---
    # Generate centers using a hexagonal lattice pattern for high density
    centers = []
    row = 0
    # We estimate a radius to fit a grid, but actual radii will be determined by optimization.
    # This just ensures good initial spacing.
    spacing = 0.2 
    
    # Fill rows until we have enough points
    while len(centers) < n_circles:
        for col in range(0, 6): # Enough columns to fit width
            x = (col + 0.5) * spacing
            # Offset rows for hexagonal packing
            y = row * spacing * math.sqrt(3)/2 + 0.1
            
            # Adjust x for odd rows to nest circles
            if row % 2 != 0:
                x += spacing / 2.0
                
            # Keep within bounds roughly
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers.append([x, y])
                if len(centers) >= n_circles:
                    break
        row += 1
    
    centers = np.array(centers[:n_circles])
    radii = np.ones(n_circles) * 0.05 # Start with small radii
    
    # --- 2. Optimization Loop (Physics Simulation) ---
    iterations = 2000
    # Initial growth rate and force coefficient
    growth_rate = 0.001 
    repulsion_strength = 10.0
    damping = 0.9999 # Slight damping to help convergence
    
    # Loop parameters
    for step in range(iterations):
        # Decay growth rate to settle into fine details
        current_growth = growth_rate * (1.0 - step / iterations)
        if current_growth < 1e-6:
            current_growth = 1e-6

        # --- A. Grow Radii ---
        # Attempt to increase radii. We scale them up.
        radii *= (1.0 + current_growth * 0.05)

        # --- B. Calculate Forces and Update Centers ---
        forces = np.zeros_like(centers)
        
        # 1. Inter-circle repulsion
        # Vectorized distance calculation
        # diff shape: (26, 26, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(np.maximum(dist_sq, 1e-9))
        
        # Required distance (sum of radii)
        req_dist = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Overlap amount (positive if overlapping)
        overlap = req_dist - dist
        
        # Apply force proportional to overlap along the connection line
        # Avoid self-interaction (diagonal)
        np.fill_diagonal(overlap, 0)
        
        # Force vector = overlap * (diff / dist)
        # We need to be careful with division by zero, though dist > 0 due to max(1e-9)
        # Normalize diff
        direction = diff / dist[:, :, np.newaxis]
        
        # Accumulate forces
        # forces[i] += sum_j( overlap_ij * direction_ij )
        # overlap is (N, N), direction is (N, N, 2)
        forces += np.sum(overlap[:, :, np.newaxis] * direction, axis=1)

        # 2. Wall repulsion
        # Left wall (x - r < 0) -> push right
        overlap_x_neg = radii - centers[:, 0]
        overlap_x_pos = centers[:, 0] + radii - 1
        overlap_y_neg = radii - centers[:, 1]
        overlap_y_pos = centers[:, 1] + radii - 1
        
        # Only apply force if overlapping (positive value)
        overlap_x_neg = np.maximum(0, overlap_x_neg)
        overlap_x_pos = np.maximum(0, overlap_x_pos)
        overlap_y_neg = np.maximum(0, overlap_y_neg)
        overlap_y_pos = np.maximum(0, overlap_y_pos)
        
        # Add to forces
        # Left wall pushes right (+x)
        forces[:, 0] += overlap_x_neg
        # Right wall pushes left (-x)
        forces[:, 0] -= overlap_x_pos
        # Bottom wall pushes up (+y)
        forces[:, 1] += overlap_y_neg
        # Top wall pushes down (-y)
        forces[:, 1] -= overlap_y_pos
        
        # Scale forces
        forces *= repulsion_strength
        
        # Update centers
        centers += forces * 0.01 # Step size for position update
        
        # --- C. Boundary Clamping ---
        # Ensure centers don't fly out, though forces should keep them in.
        # Also, strictly, center must be at least radius away from wall.
        # But we rely on forces for that. Let's just clamp to [0,1] to be safe.
        centers = np.clip(centers, 0, 1)
        
    # --- 3. Final Adjustment ---
    # One last pass to clean up any tiny numerical overlaps and tighten radii
    # Calculate max valid radius for each circle based on current positions
    # This is a greedy local improvement
    for i in range(n_circles):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(n_circles):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                max_r = min(max_r, dist - radii[j])
        radii[i] = max(0, max_r - 1e-9) # Tiny buffer

    # Sort circles by radius descending (optional, just for consistency)
    # But we need to keep centers aligned.
    # The validate function doesn't require sorting.
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
