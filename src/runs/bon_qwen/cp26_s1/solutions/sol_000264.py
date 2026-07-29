# sol_000264 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=4b5f1976 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    np.random.seed(42)
    n = 26
    
    # 1. Initialize centers in a hexagonal pattern for good starting density
    centers = np.zeros((n, 2))
    radii = np.ones(n) * 0.04
    idx = 0
    row, col = 0, 0
    while idx < n:
        x = 0.1 + col * 0.16
        y = 0.1 + row * 0.16 * np.sqrt(3) / 2
        if row % 2 == 1:
            x += 0.08
        if x <= 0.9 and y <= 0.9:
            centers[idx] = [x, y] + np.random.uniform(-0.02, 0.02, 2)
            idx += 1
        col += 1
        if col > 6:
            col = 0
            row += 1
            
    # 2. Parameters for gradient descent simulation
    lr = 0.005
    W = 500.0
    steps = 20000
    decay = 0.9998
    
    for step in range(steps):
        grad_c = np.zeros_like(centers)
        grad_r = -np.ones(n)  # Gradient of objective: -sum(r)
        
        for i in range(n):
            ci = centers[i]
            ri = radii[i]
            
            # Pairwise interaction gradients
            for j in range(i + 1, n):
                diff = ci - centers[j]
                d = np.sqrt(np.sum(diff**2))
                if d < ri + radii[j]:
                    d = max(d, 1e-7)
                    overlap = ri + radii[j] - d
                    factor = 2.0 * overlap / d
                    grad_c[i] -= factor * diff
                    grad_c[j] += factor * diff
                    grad_r[i] += 2.0 * overlap
                    grad_r[j] += 2.0 * overlap
                    
            # Boundary interaction gradients
            # Left: x >= r
            if ci[0] < ri:
                v = ri - ci[0]
                grad_c[i, 0] -= 2.0 * v
                grad_r[i] += 2.0 * v
            # Right: x + r <= 1
            if ci[0] + ri > 1.0:
                v = ci[0] + ri - 1.0
                grad_c[i, 0] += 2.0 * v
                grad_r[i] += 2.0 * v
            # Bottom: y >= r
            if ci[1] < ri:
                v = ri - ci[1]
                grad_c[i, 1] -= 2.0 * v
                grad_r[i] += 2.0 * v
            # Top: y + r <= 1
            if ci[1] + ri > 1.0:
                v = ci[1] + ri - 1.0
                grad_c[i, 1] += 2.0 * v
                grad_r[i] += 2.0 * v
                
        # Update variables
        centers -= lr * W * grad_c
        radii -= lr * W * grad_r
        
        # Clamp to keep physical validity during simulation
        radii = np.maximum(radii, 1e-4)
        centers = np.clip(centers, 0.0, 1.0)
        
        lr *= decay
        
    # 3. Final strict projection to guarantee checker passes
    # Enforce boundaries strictly
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        r = min(r, x, 1-x, y, 1-y)
        radii[i] = max(r, 0.0)
        
    # Enforce non-overlap strictly by shrinking overlapping pairs
    for _ in range(20):
        changed = False
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt(np.sum((centers[i] - centers[j])**2))
                r_sum = radii[i] + radii[j]
                if d < r_sum:
                    shrink = (r_sum - d) / 2.0 + 1e-6
                    radii[i] -= shrink
                    radii[j] -= shrink
                    radii[i] = max(radii[i], 0.0)
                    radii[j] = max(radii[j], 0.0)
                    changed = True
        if not changed:
            break
            
    # Final boundary safety check
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(radii[i], x, 1-x, y, 1-y)
        
    return centers, radii, np.sum(radii)
