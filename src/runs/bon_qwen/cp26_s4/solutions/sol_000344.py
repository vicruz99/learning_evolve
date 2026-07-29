# sol_000344 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2bd19375) state=f304079c sum of radii=1.766874 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    
    # 1. Initialize centers in a compact hexagonal-like pattern
    centers = np.zeros((n, 2))
    idx = 0
    spacing = 0.18
    for r in range(5):
        for c in range(5):
            if idx >= n: break
            cx = 0.08 + c * spacing + (r % 2) * (spacing / 2)
            cy = 0.08 + r * spacing
            centers[idx] = [cx, cy]
            idx += 1
    while idx < n:
        centers[idx] = [0.5, 0.5]
        idx += 1
        
    radii = np.full(n, 0.025)
    np.random.seed(42)
    
    # 2. Iterative expansion and conflict resolution
    for step in range(100000):
        # Slowly expand radii to push against constraints
        radii *= 1.00001
        
        # Compute pairwise distances and identify overlaps
        diff = centers[:, None] - centers[None, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        r_sum = radii[:, None] + radii[None, :]
        overlap_mask = dist < r_sum - 1e-9
        
        if np.any(overlap_mask):
            # Find the pair with maximum overlap to resolve first
            rows, cols = np.where(overlap_mask)
            ov_amounts = r_sum[rows, cols] - dist[rows, cols]
            best = np.argmax(ov_amounts)
            i, j = rows[best], cols[best]
            
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            d = np.hypot(dx, dy)
            if d < 1e-9:
                d = 1e-9
                dx, dy = 1.0, 0.0
                
            nx, ny = dx/d, dy/d
            shift = ov_amounts[best] / 2
            
            # Push centers apart exactly to touch
            centers[i, 0] += nx * shift
            centers[i, 1] += ny * shift
            centers[j, 0] -= nx * shift
            centers[j, 1] -= ny * shift
            
    # 3. Final boundary clamping
    for i in range(n):
        centers[i, 0] = np.clip(centers[i, 0], radii[i], 1-radii[i])
        centers[i, 1] = np.clip(centers[i, 1], radii[i], 1-radii[i])
        
    # 4. Final cleanup pass to ensure strict non-overlap compliance
    for _ in range(500):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                dx = centers[i,0] - centers[j,0]
                dy = centers[i,1] - centers[j,1]
                d = np.hypot(dx, dy)
                if d < radii[i] + radii[j] and d > 1e-9:
                    ov = radii[i] + radii[j] - d
                    nx, ny = dx/d, dy/d
                    centers[i,0] += nx*ov/2
                    centers[i,1] += ny*ov/2
                    centers[j,0] -= nx*ov/2
                    centers[j,1] -= ny*ov/2
                    changed = True
        if not changed: break
        for i in range(n):
            centers[i, 0] = np.clip(centers[i, 0], radii[i], 1-radii[i])
            centers[i, 1] = np.clip(centers[i, 1], radii[i], 1-radii[i])

    return centers, radii, float(np.sum(radii))
