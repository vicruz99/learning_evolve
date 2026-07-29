# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 4705e2a5) state=fce71a15 sum of radii=1.502458 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal initialization followed by an iterative force-based optimization
    to resolve overlaps and expand radii.
    """
    n = 26
    np.random.seed(42)
    
    # Initialize centers in a hexagonal pattern for a good starting density
    centers = np.zeros((n, 2))
    s = 0.22  # Initial spacing
    idx = 0
    row = 0
    
    # Generate hexagonal grid points
    while idx < n and row < 10:
        col = 0
        while idx < n and col < 10:
            # Hexagonal offset
            x = col * s + (row % 2) * (s / 2) + 0.05
            y = row * s * math.sqrt(3) / 2 + 0.05
            
            if 0 <= x <= 1 and 0 <= y <= 1:
                centers[idx] = [x, y]
                idx += 1
            col += 1
        row += 1
        
    # Fallback for any remaining circles (should not happen with these params)
    while idx < n:
        centers[idx] = [np.random.uniform(0.1, 0.9), np.random.uniform(0.1, 0.9)]
        idx += 1

    # Initial radii small enough to fit easily without overlap
    radii = np.full(n, 0.04)
    
    # Optimization parameters
    repulsion_strength = 10.0 
    growth_factor = 1.0001    # Slow growth to find precise limit
    max_iterations = 8000     
    
    # Main optimization loop
    for it in range(max_iterations):
        forces = np.zeros_like(centers)
        is_valid = True
        
        # 1. Pairwise interactions (Repulsion on overlap)
        for i in range(n):
            r_i = radii[i]
            ci = centers[i]
            for j in range(i + 1, n):
                r_j = radii[j]
                cj = centers[j]
                
                diff = ci - cj
                dist_sq = np.dot(diff, diff)
                dist = math.sqrt(dist_sq)
                
                if dist < r_i + r_j:
                    is_valid = False
                    overlap = r_i + r_j - dist
                    if dist > 1e-9:
                        # Repulsive force proportional to overlap
                        f_mag = overlap * repulsion_strength
                        dir_vec = diff / dist
                        forces[i] += dir_vec * f_mag
                        forces[j] -= dir_vec * f_mag
                    else:
                        # If centers coincide, apply random push
                        forces[i] += np.random.uniform(-1, 1, 2)
                        forces[j] -= np.random.uniform(-1, 1, 2)
        
        # 2. Boundary interactions (Repulsion from walls)
        for i in range(n):
            r_i = radii[i]
            x, y = centers[i]
            
            # Left boundary
            if x < r_i:
                is_valid = False
                forces[i, 0] += (r_i - x) * repulsion_strength
            # Right boundary
            elif x > 1 - r_i:
                is_valid = False
                forces[i, 0] -= (x - (1 - r_i)) * repulsion_strength
            
            # Bottom boundary
            if y < r_i:
                is_valid = False
                forces[i, 1] += (r_i - y) * repulsion_strength
            # Top boundary
            elif y > 1 - r_i:
                is_valid = False
                forces[i, 1] -= (y - (1 - r_i)) * repulsion_strength

        # 3. Update positions
        # Learning rate decay for stability
        lr = 1.0 / (1.0 + it * 0.0002)
        centers += forces * lr * 0.2
        
        # Clamp centers to valid range [0, 1]
        np.clip(centers, 0.0, 1.0, out=centers)
        
        # 4. Grow radii if configuration is valid
        if is_valid:
            radii *= growth_factor
        # If invalid, rely on position updates to resolve overlaps

    # Post-processing: Strict validity enforcement
    # Iteratively reduce radii to resolve any remaining overlaps with high precision
    for _ in range(500):
        changed = False
        
        # Enforce boundary constraints
        for i in range(n):
            x, y = centers[i]
            max_r = min(x, 1 - x, y, 1 - y)
            if radii[i] > max_r + 1e-10:
                radii[i] = max_r
                changed = True
        
        # Enforce pairwise non-overlap
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist = math.sqrt(np.dot(diff, diff))
                req_dist = radii[i] + radii[j]
                
                if dist < req_dist - 1e-10:
                    overlap = req_dist - dist
                    total_r = radii[i] + radii[j]
                    if total_r > 1e-12:
                        # Scale down radii proportionally to maintain relative sizes
                        scale = dist / total_r
                        radii[i] *= scale
                        radii[j] *= scale
                    else:
                        radii[i] = dist / 2
                        radii[j] = dist / 2
                    changed = True
        
        if not changed:
            break
            
    # Ensure non-negative radii
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, float(np.sum(radii))
