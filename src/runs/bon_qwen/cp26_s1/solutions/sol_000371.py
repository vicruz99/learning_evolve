# sol_000371 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b90f636d) state=75e3922d sum of radii=1.821175 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Pack 26 circles in a unit square to maximize the sum of radii.
    Uses a force-directed optimization with hexagonal initialization and adaptive radius growth.
    """
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a hexagonal grid pattern
    centers = np.zeros((n, 2))
    idx = 0
    # Layout: 5 rows with counts 6, 5, 6, 5, 4 sums to 26
    layout = [6, 5, 6, 5, 4]
    for i, count in enumerate(layout):
        for j in range(count):
            if idx < n:
                # Hexagonal spacing initialization
                centers[idx, 0] = (j + 0.5 * (i % 2)) * 1.0 / (count + 0.5)
                centers[idx, 1] = (i + 0.5) * 1.0 / 5.0
                idx += 1
    # Fill any remaining slots (safety fallback)
    while idx < n:
        centers[idx] = np.random.rand(2)
        idx += 1
        
    r = 0.05
    velocities = np.zeros_like(centers)
    
    # Simulation parameters
    dt = 0.05
    k_repulse = 25.0
    k_wall = 15.0
    damping = 0.75
    best_r = r
    
    # 2. & 3. Force simulation with adaptive radius growth
    for step in range(6000):
        # Gradually increase target radius
        r += 0.00003
        
        forces = np.zeros_like(centers)
        
        for i in range(n):
            # Wall repulsion forces
            for d in range(2):
                if centers[i, d] < r:
                    forces[i, d] += k_wall * (r - centers[i, d])
                if centers[i, d] > 1 - r:
                    forces[i, d] -= k_wall * (centers[i, d] - (1 - r))
                    
            # Circle-circle repulsion forces
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = math.hypot(dx, dy)
                
                if dist < 2 * r and dist > 1e-6:
                    # Repulsive force proportional to overlap
                    overlap = 2 * r - dist
                    f = k_repulse * overlap / dist
                    forces[i, 0] += dx * f
                    forces[i, 1] += dy * f
                    forces[j, 0] -= dx * f
                    forces[j, 1] -= dy * f
                elif dist <= 1e-6:
                    # Break symmetry if circles coincide
                    rx, ry = np.random.randn(2) * 0.01
                    forces[i, 0] += rx
                    forces[i, 1] += ry
                    forces[j, 0] -= rx
                    forces[j, 1] -= ry
                    
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        centers += velocities
        
        # Keep centers strictly within bounds
        centers = np.clip(centers, 0.0, 1.0)
        
        # Periodically validate and track best radius
        if step % 100 == 0:
            valid = True
            for i in range(n):
                if (centers[i, 0] < r - 1e-5 or centers[i, 0] > 1 - r + 1e-5 or 
                    centers[i, 1] < r - 1e-5 or centers[i, 1] > 1 - r + 1e-5):
                    valid = False
                    break
            if valid:
                for i in range(n):
                    for j in range(i+1, n):
                        if math.hypot(centers[i, 0]-centers[j, 0], centers[i, 1]-centers[j, 1]) < 2*r - 1e-5:
                            valid = False
                            break
                    if not valid: break
            if valid:
                best_r = r
            else:
                r = best_r # Revert if invalid to maintain stability
                
    # 4. Final strict adjustment to guarantee validity
    max_r = 1.0
    for i in range(n):
        max_r = min(max_r, centers[i, 0], 1-centers[i, 0], centers[i, 1], 1-centers[i, 1])
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(centers[i, 0]-centers[j, 0], centers[i, 1]-centers[j, 1])
            max_r = min(max_r, d / 2)
            
    radii = np.full(n, max_r)
    return centers, radii, float(np.sum(radii))
