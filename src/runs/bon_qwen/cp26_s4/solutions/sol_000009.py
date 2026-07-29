# sol_000009 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state ca1ebfe6) state=3325542b sum of radii=0.448665 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing():
    # Number of circles
    N = 26
    
    # Initial centers setup: 6x5 grid
    cols, rows = 6, 5
    spacing_x = 1.0 / cols
    spacing_y = 1.0 / rows
    
    centers = []
    r_init = 0.001
    
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = spacing_x * (c + 0.5)
            y = spacing_y * (r + 0.5)
            centers.append([x, y])
            count += 1
        if count >= N:
            break
            
    centers = np.array(centers)
    radii = np.full(N, r_init)
    
    # Optimization parameters
    iterations = 3000
    dt = 0.5
    
    # Random seed for reproducibility
    np.random.seed(42)

    for i in range(iterations):
        # Cooling schedule
        if i < 1500:
            growth_rate = 1.0015
            jitter = 0.005
        elif i < 2500:
            growth_rate = 1.0005
            jitter = 0.002
        else:
            growth_rate = 1.0002
            jitter = 0.0005
            
        # Jitter centers
        centers += np.random.uniform(-jitter, jitter, centers.shape)
        
        # Grow radii
        radii *= growth_rate
        
        # Resolve overlaps and boundaries
        # Iterate multiple times per frame to ensure stability
        for _ in range(5):
            for j in range(N):
                x, y = centers[j]
                r = radii[j]
                
                # Boundary resolution
                if x - r < 0:
                    centers[j, 0] = r
                elif x + r > 1:
                    centers[j, 0] = 1 - r
                
                if y - r < 0:
                    centers[j, 1] = r
                elif y + r > 1:
                    centers[j, 1] = 1 - r
            
            # Pairwise resolution
            for j in range(N):
                for k in range(j + 1, N):
                    dx = centers[k, 0] - centers[j, 0]
                    dy = centers[k, 1] - centers[j, 1]
                    dist = math.hypot(dx, dy)
                    req_dist = radii[j] + radii[k]
                    
                    if dist < req_dist:
                        if dist == 0:
                            # Handle exact overlap (assign random direction)
                            dx = np.random.uniform(-1, 1)
                            dy = np.random.uniform(-1, 1)
                            dist = math.hypot(dx, dy)
                        
                        norm_x = dx / dist
                        norm_y = dy / dist
                        
                        separation = (req_dist - dist) / 2.0
                        centers[j, 0] -= norm_x * separation
                        centers[j, 1] -= norm_y * separation
                        centers[k, 0] += norm_x * separation
                        centers[k, 1] += norm_y * separation

    # Final cleanup to ensure strict validity
    for j in range(N):
        radii[j] = min(radii[j], centers[j, 0], 1 - centers[j, 0], 
                       centers[j, 1], 1 - centers[j, 1])
        
        for k in range(j + 1, N):
            dist = math.hypot(centers[k, 0] - centers[j, 0], 
                              centers[k, 1] - centers[j, 1])
            max_r = dist - radii[k]
            if max_r < radii[j]:
                radii[j] = max_r
                
    # Ensure non-negative radii
    radii = np.maximum(radii, 0)
    
    return centers, radii, np.sum(radii)
