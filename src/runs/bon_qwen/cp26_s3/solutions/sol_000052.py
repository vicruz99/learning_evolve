# sol_000052 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state baeb2167) state=b115df73 sum of radii=0.915850 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a hexagonal-like pattern
    centers = np.zeros((n, 2))
    idx = 0
    rows = [6, 5, 6, 5, 4]
    y = 0.12
    dy = 0.16
    for size in rows:
        x_start = (1.0 - (size - 1) * 0.17) / 2.0
        for i in range(size):
            centers[idx] = [x_start + i * 0.17, y]
            idx += 1
        y += dy
        if idx >= n:
            break
            
    # Add small random perturbation to break symmetry and avoid local traps
    centers += np.random.uniform(-0.02, 0.02, centers.shape)
    centers = np.clip(centers, 0.0, 1.0)
            
    # 2. Iterative expansion with force-directed relaxation
    r = 0.05
    lr = 1e-3
    
    for step in range(15000):
        forces = np.zeros_like(centers)
        has_overlap = False
        
        for i in range(n):
            # Wall repulsion forces
            for d in range(2):
                if centers[i, d] < r:
                    forces[i, d] += (r - centers[i, d]) * 15.0
                if centers[i, d] > 1.0 - r:
                    forces[i, d] -= (1.0 - r - centers[i, d]) * 15.0
                    
            # Pairwise repulsion forces
            for j in range(i + 1, n):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 2.0 * r and dist > 1e-8:
                    # Repulsion magnitude proportional to overlap depth
                    repulse = (2.0 * r - dist) / dist
                    forces[i, 0] += repulse * dx
                    forces[i, 1] += repulse * dy
                    forces[j, 0] -= repulse * dx
                    forces[j, 1] -= repulse * dy
                    has_overlap = True
                    
        # Update positions
        centers += lr * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Adaptive learning rate and radius growth
        if not has_overlap:
            r += 5e-5
            lr *= 0.999
        else:
            lr *= 0.997
        lr = max(lr, 5e-5)
            
    # 3. Compute exact maximum valid radius from final configuration
    min_sep = 1.0
    for i in range(n):
        # Minimum distance to walls
        min_sep = min(min_sep, centers[i, 0], 1.0 - centers[i, 0], 
                              centers[i, 1], 1.0 - centers[i, 1])
        # Minimum distance to other circles
        for j in range(i + 1, n):
            dist = np.sqrt((centers[i, 0] - centers[j, 0])**2 + 
                           (centers[i, 1] - centers[j, 1])**2)
            min_sep = min(min_sep, dist)
            
    # Apply small safety margin for numerical tolerance in validation
    r_final = 0.5 * min_sep - 1e-9
    radii = np.full(n, r_final)
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
