# sol_000069 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 34f92e2c) state=4cb4f0b6 sum of radii=1.664780 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_forces(centers, radii, n):
    """Compute repulsive forces for boundary violations and circle overlaps."""
    forces = np.zeros((n, 2))
    
    # Boundary forces
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < 0: forces[i, 0] += (r - x) * 200.0
        if x + r > 1: forces[i, 0] -= (x + r - 1) * 200.0
        if y - r < 0: forces[i, 1] += (r - y) * 200.0
        if y + r > 1: forces[i, 1] -= (y + r - 1) * 200.0
        
    # Overlap forces
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[j, 0] - centers[i, 0]
            dy = centers[j, 1] - centers[i, 1]
            dist_sq = dx*dx + dy*dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-10 else 1e-7
            r_sum = radii[i] + radii[j]
            
            if dist < r_sum:
                overlap = r_sum - dist
                # Scale force by overlap and inverse distance to handle near-coincidence safely
                f = overlap * 150.0 / (dist + 1e-5)
                forces[i, 0] -= dx * f
                forces[i, 1] -= dy * f
                forces[j, 0] += dx * f
                forces[j, 1] += dy * f
            elif dist < 1e-6:
                # Random push if centers are numerically identical
                ang = np.random.rand() * 2 * np.pi
                fx, fy = np.cos(ang), np.sin(ang)
                forces[i] -= np.array([fx, fy]) * 0.1
                forces[j] += np.array([fx, fy]) * 0.1
                
    return forces

def check_valid(centers, radii, n):
    """Strict validation of non-overlap and boundary constraints."""
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if r < 0 or x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Hexagonal-like initialization for high initial density
    centers = np.zeros((n, 2))
    idx = 0
    row_counts = [6, 5, 6, 5, 4]  # Sums to 26
    for r_idx, count in enumerate(row_counts):
        y = (r_idx + 1) / 6.0 * 0.9 + 0.05
        x_step = 0.9 / (count + 1)
        for c_idx in range(count):
            x = x_step * (c_idx + 1) + 0.05
            centers[idx] = [x, y]
            idx += 1
            
    radii = np.full(n, 0.04)
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = 0.0
    
    lr = 0.00015      # Learning rate for center updates
    growth = 3e-5     # Radius expansion rate
    temp = 0.8        # Noise temperature
    num_steps = 4000  # Total optimization steps
    
    for step in range(num_steps):
        # Gradually expand circles
        radii += growth
        radii = np.clip(radii, 0, 0.5)
        
        # Compute physical forces
        forces = compute_forces(centers, radii, n)
        
        # Add exploration noise
        noise = np.random.randn(n, 2) * temp * 0.0005
        
        # Update positions and clamp to unit square
        centers += lr * (forces + noise)
        centers = np.clip(centers, 0, 1)
        
        # Periodically validate and track best valid configuration
        if step % 20 == 0:
            if check_valid(centers, radii, n):
                s = np.sum(radii)
                if s > best_sum:
                    best_sum = s
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
        # Cooling schedule: reduce noise, growth, and learning rate over time
        if step > 2500:
            temp *= 0.999
            growth *= 0.9995
            lr *= 0.9995
            if growth < 1e-7: 
                growth = 0  # Stop expansion if converged
                
    return best_centers, best_radii, best_sum
