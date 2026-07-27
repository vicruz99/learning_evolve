# sol_000183 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9afef83a) state=48597ec9 sum of radii=2.022927 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    np.random.seed(42)
    N = 26
    
    # Initialize centers in a hexagonal pattern
    centers = np.zeros((N, 2))
    idx = 0
    counts = [6, 5, 6, 5, 4]
    for r_idx, count in enumerate(counts):
        y = r_idx * 0.18 + 0.15
        for c_idx in range(count):
            x = c_idx * 0.2 + 0.15 + (0.1 if r_idx % 2 == 1 else 0)
            if idx < N:
                centers[idx] = [x, y]
                idx += 1
                
    # Add small noise to break symmetry and avoid grid-locking
    centers += np.random.randn(N, 2) * 0.005
    centers = np.clip(centers, 0.05, 0.95)
    
    radii = np.full(N, 0.04)
    
    steps = 5000
    for step in range(steps):
        forces = np.zeros_like(centers)
        
        # Circle-circle repulsion
        for i in range(N):
            for j in range(i + 1, N):
                diff = centers[i] - centers[j]
                dist = np.hypot(diff[0], diff[1])
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    overlap = min_dist - dist
                    if dist > 1e-9:
                        factor = overlap / dist
                        forces[i, 0] += diff[0] * factor
                        forces[i, 1] += diff[1] * factor
                        forces[j, 0] -= diff[0] * factor
                        forces[j, 1] -= diff[1] * factor
                    else:
                        # Fallback for coincident centers
                        forces[i, 0] += np.random.randn() * 0.05
                        forces[i, 1] += np.random.randn() * 0.05
                        forces[j, 0] -= forces[i, 0]
                        forces[j, 1] -= forces[i, 1]
                        
            # Boundary repulsion
            x, y = centers[i]
            r = radii[i]
            if x < r: forces[i, 0] += (r - x) * 2.0
            if x > 1.0 - r: forces[i, 0] -= (x - (1.0 - r)) * 2.0
            if y < r: forces[i, 1] += (r - y) * 2.0
            if y > 1.0 - r: forces[i, 1] -= (y - (1.0 - r)) * 2.0
            
        # Adaptive step size (cooldown)
        alpha = 0.15 * (1.0 - step / steps) + 0.005
        
        centers += alpha * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Expand radii gradually
        if step < 4000:
            expansion = 1.0 + 0.004 / (1.0 + step / 300.0)
            for i in range(N):
                max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
                for j in range(N):
                    if i == j: continue
                    d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
                    if d / 2.0 < max_r:
                        max_r = d / 2.0
                radii[i] = min(radii[i] * expansion, max_r * 0.995)

    # Final exact radius calculation based on settled positions
    final_radii = np.zeros(N)
    for i in range(N):
        max_r = min(centers[i, 0], 1.0 - centers[i, 0], centers[i, 1], 1.0 - centers[i, 1])
        for j in range(N):
            if i == j: continue
            d = np.hypot(centers[i,0] - centers[j,0], centers[i,1] - centers[j,1])
            if d / 2.0 < max_r:
                max_r = d / 2.0
        final_radii[i] = max_r
        
    # Conservative adjustment to guarantee strict inequality in validation
    final_radii = final_radii * 0.9999999
    
    return centers, final_radii, np.sum(final_radii)
