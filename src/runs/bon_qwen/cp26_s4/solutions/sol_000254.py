# sol_000254 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 30e75f73) state=b4b1276c sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Returns centers and radii for 26 circles in a unit square, maximizing sum of radii.
    """
    np.random.seed(42)
    n = 26
    
    # Initial configuration based on hexagonal packing density
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.05
    
    # Row distribution: 6, 5, 6, 5, 4 circles = 26 total
    row_counts = [6, 5, 6, 5, 4]
    current_idx = 0
    
    for row_idx, count in enumerate(row_counts):
        y = 0.1 + row_idx * 0.2
        # Shift odd rows slightly to mimic hexagonal stagger
        x_start = 0.08 + (0.03 if row_idx % 2 == 1 else 0.0)
        
        for col_idx in range(count):
            x = x_start + col_idx * 0.16
            if current_idx < n:
                centers[current_idx] = [x, y]
                current_idx += 1
                
    # Physics simulation parameters
    num_iterations = 8000
    expansion_rate = 0.00005
    repulsion_strength = 100.0
    damping = 0.85
    
    for step in range(num_iterations):
        forces = np.zeros_like(centers)
        
        # 1. Apply expansion force (increase radii)
        # If no overlap is detected in a pass, we could expand more, 
        # but here we expand continuously and let repulsion handle the rest.
        # To maximize sum, we bias towards expansion.
        current_max_r = np.max(radii)
        radii += expansion_rate * (1.0 - current_max_r) 
        
        # 2. Calculate repulsion forces between circles and boundaries
        for i in range(n):
            # Boundary repulsion (keep circles inside [0,1])
            for axis in range(2):
                if centers[i, axis] - radii[i] < 0:
                    forces[i, axis] += repulsion_strength * (radii[i] - centers[i, axis])
                if centers[i, axis] + radii[i] > 1:
                    forces[i, axis] -= repulsion_strength * (centers[i, axis] + radii[i] - 1)
            
            # Circle-Circle repulsion
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                required_dist = radii[i] + radii[j]
                
                if dist < required_dist and dist > 1e-6:
                    # Overlap repulsion: push apart proportional to overlap
                    overlap = required_dist - dist
                    force_mag = repulsion_strength * overlap
                    direction = diff / dist
                    forces[i] += direction * force_mag
                    forces[j] -= direction * force_mag
                elif dist < 1e-6:
                    # Prevent division by zero if centers coincide
                    forces[i] += (np.random.rand(2) - 0.5) * 10

        # 3. Update positions
        # Apply velocity-like update with damping
        centers += forces * (0.001 + 0.0001 * (num_iterations - step) / num_iterations)
        
        # Clamp to square boundaries strictly
        centers = np.clip(centers, 0, 1)
        
    # Post-processing: Shrink radii slightly if any overlaps remain due to numerical noise
    # and ensure they are valid.
    # A small safety margin ensures validity against the 1e-12 check in the validator.
    radii -= 1e-4
    
    # Final validation and radius adjustment for strict compliance
    # We re-run a quick fix to ensure no overlaps exist with the shrunken radii
    # This is usually not needed if the simulation converged, but good for safety.
    # If a circle is still too big for its position, we scale its radius to fit neighbors.
    
    # Recalculate max possible radius for each circle given the final centers
    valid_radii = []
    for i in range(n):
        max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
        for j in range(n):
            if i != j:
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                max_r = min(max_r, dist - radii[j])
        valid_radii.append(max(0, max_r))
        
    # Use the smaller of the simulated radii and the geometrically constrained radii
    final_radii = np.minimum(radii, np.array(valid_radii))
    
    return centers, final_radii, float(np.sum(final_radii))
