# sol_000217 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b505a133) state=fbdef39c sum of radii=1.352456 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from math import sqrt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Set random seed for reproducibility if needed, but not strictly required
    # np.random.seed(42)
    
    n_circles = 26
    centers = np.zeros((n_circles, 2))
    
    # Initialize centers in a hexagonal pattern
    # 6 rows to fit 26 circles: 5, 4, 5, 4, 5, 3
    # This provides a good initial distribution
    counts = [5, 4, 5, 4, 5, 3]
    y_spacing = 0.17
    x_spacing = 0.20
    
    idx = 0
    for i, count in enumerate(counts):
        y = 0.15 + i * y_spacing
        # Center the row horizontally
        x_start = (1.0 - (count - 1) * x_spacing) / 2.0
        # Add alternating shift for hexagonal structure
        if i % 2 == 1:
            x_start += x_spacing / 2.0
            
        for j in range(count):
            centers[idx, 0] = x_start + j * x_spacing
            centers[idx, 1] = y
            idx += 1
            
    # Initialize radii to a small value
    radii = np.ones(n_circles) * 0.02
    
    # Simulation parameters
    max_iter = 2000
    dt = 0.01  # Time step
    
    # Run the expansion and relaxation loop
    for iteration in range(max_iter):
        # Decay factors for cooling
        cooling = 1.0 / (1.0 + iteration * 0.01)
        growth_rate = 0.001 * cooling
        force_scale = 0.5 * cooling
        
        # 1. Expand radii
        radii += growth_rate
        
        # 2. Compute forces to resolve overlaps
        forces = np.zeros_like(centers)
        
        # Boundary forces (pushing inward)
        for i in range(n_circles):
            x, y = centers[i]
            r = radii[i]
            
            # Left wall
            if x - r < 0:
                forces[i, 0] += (r - x) * force_scale
            # Right wall
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * force_scale
            # Bottom wall
            if y - r < 0:
                forces[i, 1] += (r - y) * force_scale
            # Top wall
            if y + r > 1:
                forces[i, 1] -= (y + r - 1) * force_scale
                
        # Pairwise repulsive forces
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = centers[j, 0] - centers[i, 0]
                dy = centers[j, 1] - centers[i, 1]
                dist = sqrt(dx*dx + dy*dy)
                
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-6:
                    # Overlap detected
                    # Repulsive force proportional to overlap
                    overlap = min_dist - dist
                    fx = (dx / dist) * overlap * force_scale
                    fy = (dy / dist) * overlap * force_scale
                    
                    forces[i, 0] -= fx
                    forces[i, 1] -= fy
                    forces[j, 0] += fx
                    forces[j, 1] += fy
                elif dist < 1e-6:
                    # Prevent division by zero for identical centers
                    angle = np.random.uniform(0, 2 * np.pi)
                    forces[i, 0] -= np.cos(angle) * 0.01 * force_scale
                    forces[i, 1] -= np.sin(angle) * 0.01 * force_scale
                    forces[j, 0] += np.cos(angle) * 0.01 * force_scale
                    forces[j, 1] += np.sin(angle) * 0.01 * force_scale

        # 3. Update centers based on forces
        centers += forces * dt
        
        # 4. Hard clamp centers to stay within bounds (safety net)
        centers[:, 0] = np.clip(centers[:, 0], 0.001, 0.999)
        centers[:, 1] = np.clip(centers[:, 1], 0.001, 0.999)
        
        # 5. Hard clamp radii to ensure validity based on current positions
        # This step is crucial for the validate_packing check
        for i in range(n_circles):
            r = radii[i]
            x, y = centers[i]
            
            # Wall constraints
            r = min(r, x, 1 - x, y, 1 - y)
            
            # Neighbor constraints
            for j in range(n_circles):
                if i == j: continue
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                d = sqrt(dx*dx + dy*dy)
                r = min(r, (d - radii[j]) / 2.0)
            
            radii[i] = max(0.0, r)

    # Final cleanup to ensure strict validity
    for i in range(n_circles):
        r = radii[i]
        x, y = centers[i]
        
        # Wall constraints
        r = min(r, x, 1 - x, y, 1 - y)
        
        # Neighbor constraints
        for j in range(n_circles):
            if i == j: continue
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = sqrt(dx*dx + dy*dy)
            r = min(r, (d - radii[j]) / 2.0)
        
        radii[i] = max(0.0, r)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
