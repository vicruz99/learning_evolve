# sol_000057 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9227c4d6) state=020d70c3 sum of radii=0.000211 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed simulation with adaptive radii expansion.
    """
    np.random.seed(42)
    n_circles = 26
    centers = np.zeros((n_circles, 2))
    radii = np.zeros(n_circles)
    
    # Initialization: 5x5 grid pattern for robust start
    # 25 circles in grid + 1 extra
    row = 0
    col = 0
    idx = 0
    for r in range(5):
        for c in range(5):
            if idx < n_circles:
                # Distribute in a slightly perturbed grid to avoid perfect symmetry locking
                centers[idx] = [0.1 + c * 0.2, 0.1 + r * 0.2]
                radii[idx] = 0.01  # Start with small radii to allow expansion
                idx += 1
    
    # Add the 26th circle in a gap
    if idx < n_circles:
        centers[idx] = [0.5, 0.5] # Center of the square
        radii[idx] = 0.01
        idx += 1

    # Optimization parameters
    n_steps = 1500
    dt = 0.05  # Time step for center movement
    grow_rate = 0.002  # Rate of radius growth
    repulsion_strength = 10.0  # Strength of overlap repulsion
    center_gravity = 0.1  # Pull towards center to maintain balance
    
    for step in range(n_steps):
        forces = np.zeros_like(centers)
        
        # Calculate pairwise repulsive forces
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                diff = centers[i] - centers[j]
                dist = np.linalg.norm(diff)
                sum_radii = radii[i] + radii[j]
                
                # If overlapping or very close, apply repulsion
                if dist < sum_radii + 1e-4:
                    if dist < 1e-9:
                        # Prevent division by zero, push random direction
                        force_vec = np.random.rand(2) - 0.5
                    else:
                        force_vec = diff / dist
                    
                    # Force magnitude proportional to overlap
                    overlap = sum_radii - dist
                    force_mag = repulsion_strength * overlap
                    forces[i] += force_vec * force_mag
                    forces[j] -= force_vec * force_mag

        # Apply forces to centers
        centers += forces * dt
        
        # Boundary constraints: bounce back
        for i in range(n_circles):
            for dim in range(2):
                if centers[i, dim] - radii[i] < 0:
                    centers[i, dim] = radii[i]
                    # Add small outward force if stuck
                    forces[i, dim] += 0.01 
                elif centers[i, dim] + radii[i] > 1:
                    centers[i, dim] = 1 - radii[i]
                    forces[i, dim] -= 0.01

        # Gravity towards center to keep packing compact
        gravity = (0.5 - centers) * center_gravity
        centers += gravity * dt
        
        # Re-clip centers after gravity
        for i in range(n_circles):
            for dim in range(2):
                centers[i, dim] = np.clip(centers[i, dim], radii[i], 1 - radii[i])

        # Expand radii
        # Grow rate can decrease over time for finer tuning
        current_grow = grow_rate * (1.0 - step / n_steps)
        radii += current_grow
        
        # Constrain radii by boundaries and neighbors immediately to prevent explosion
        for i in range(n_circles):
            # Boundary constraint
            r_max = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            if r_max < 0: r_max = 0
            
            # Neighbor constraints
            for j in range(n_circles):
                if i == j: continue
                dist = np.linalg.norm(centers[i] - centers[j])
                r_allowed = dist - radii[j]
                if r_allowed < r_max:
                    r_max = r_allowed
            
            if r_max < 0: r_max = 0
            radii[i] = min(radii[i], r_max)

    # Final cleanup: ensure strict constraints
    # If any radius is too large for its position, shrink it
    for i in range(n_circles):
        r = radii[i]
        x, y = centers[i]
        # Check boundaries
        if x - r < 0: r = x
        if x + r > 1: r = 1 - x
        if y - r < 0: r = y
        if y + r > 1: r = 1 - y
        
        # Check neighbors
        for j in range(n_circles):
            if i == j: continue
            dist = np.linalg.norm(centers[i] - centers[j])
            req_r = dist - radii[j]
            if req_r < r:
                r = req_r
        
        radii[i] = max(0, r)

    # Sort circles by radius descending for stability (optional but good practice)
    # Not strictly needed for output but helps debugging
    # order = np.argsort(radii)[::-1]
    # centers = centers[order]
    # radii = radii[order]

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
