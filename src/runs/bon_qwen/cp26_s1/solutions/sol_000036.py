# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b079e3ed) state=569eb492 sum of radii=1.139813 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    """
    Pack 26 circles in a unit square [0,1]x[0,1] to maximize sum of radii.
    """
    np.random.seed(42)
    
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Initial placement: 5x5 grid + 1 in center
    # Grid spacing 0.2, radius 0.1
    idx = 0
    for i in range(5):
        for j in range(5):
            centers[idx] = [0.1 + i * 0.2, 0.1 + j * 0.2]
            radii[idx] = 0.09 # Start slightly smaller
            idx += 1
    # 26th circle in the center gap
    centers[idx] = [0.5, 0.5]
    radii[idx] = 0.09

    # Optimization parameters
    iterations = 2000
    dt = 0.05  # Time step for movement
    growth_rate = 1e-5 # Rate of radius increase
    
    # Simulation loop
    for step in range(iterations):
        # Gradually increase target radius
        # We want to maximize sum of radii, so we try to expand them all
        # However, we must resolve overlaps.
        
        # Apply growth
        radii += growth_rate * (1 - step/iterations) # Decay growth over time for stability
        
        # Compute forces
        forces = np.zeros_like(centers)
        
        # Boundary forces (repel from walls)
        # We want centers to be at least radius away from walls
        # If center is too close, push away
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * 10 # Strong repulsion
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * 10
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * 10
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * 10
                
        # Inter-circle forces (repulsion if overlapping)
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = np.sqrt(np.sum(diff**2))
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-9:
                    # Overlap detected
                    # Force proportional to overlap amount
                    overlap = min_dist - dist
                    force_magnitude = overlap * 50 # Stiff spring
                    direction = diff / dist
                    
                    forces[i] += direction * force_magnitude
                    forces[j] -= direction * force_magnitude
                elif dist < 1e-9:
                    # Prevent division by zero, push randomly
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01

        # Update positions
        centers += forces * dt
        
        # Clamp positions to be somewhat central to avoid flying off, 
        # though boundary forces handle it. 
        # Just ensure we stay within [0,1] roughly, but radius constraint is stricter.
        # We rely on boundary forces to keep them valid.
        
        # Occasionally add jitter to escape local minima
        if step % 100 == 0:
            jitter = np.random.randn(n, 2) * 0.001
            centers += jitter
            centers = np.clip(centers, 0, 1)

    # Final adjustment to ensure strict validity
    # If any circle is outside, clamp it
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
        
    # Resolve any remaining tiny overlaps by shrinking radii slightly if necessary
    # Iterative shrink
    for _ in range(50):
        max_overlap = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                overlap = radii[i] + radii[j] - dist
                if overlap > max_overlap:
                    max_overlap = overlap
                    max_i, max_j = i, j
        
        if max_overlap > 1e-6:
            # Shrink the smaller radius or both
            shrink = max_overlap * 0.5
            radii[max_i] = max(0, radii[max_i] - shrink)
            radii[max_j] = max(0, radii[max_j] - shrink)
        else:
            break

    # Re-verify boundary constraints after shrink
    for i in range(n):
        r = radii[i]
        centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
        centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)

    sum_radii = np.sum(radii)
    return centers, radii, sum_radii

# Import numpy for the validate function context if needed, 
# though the prompt says we can use scientific libraries.
import numpy as np
