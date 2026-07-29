# sol_000149 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3a06727e) state=f76ed554 sum of radii=1.342322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def jostling_optimize(init_centers, n, iterations=4000):
    """
    Iteratively expands radii and adjusts centers to maximize sum of radii.
    Maintains validity at all steps through projection.
    """
    centers = init_centers.copy()
    radii = np.full(n, 0.02)  # Start with small radii
    
    lr = 0.05
    expand_rate = 1.0015
    
    for t in range(iterations):
        # 1. Expand radii
        radii *= expand_rate
        
        # 2. Project radii to satisfy constraints given current centers
        # Using a snapshot of radii to avoid order-dependent updates
        current_radii = radii.copy()
        for i in range(n):
            x, y = centers[i]
            # Boundary limits
            max_r = min(x, 1.0 - x, y, 1.0 - y)
            
            # Neighbor limits
            for j in range(n):
                if i == j:
                    continue
                dx = x - centers[j, 0]
                dy = y - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                max_r = min(max_r, (dist - current_radii[j]) / 2.0)
            
            radii[i] = max(0.0, max_r)
            
        # 3. Compute forces to relieve tight constraints
        forces = np.zeros_like(centers)
        for i in range(n):
            x, y = centers[i]
            r_i = radii[i]
            
            # Boundary forces: push away from walls if touching
            if x - r_i < 1e-6:
                forces[i, 0] += lr
            if 1.0 - x - r_i < 1e-6:
                forces[i, 0] -= lr
            if y - r_i < 1e-6:
                forces[i, 1] += lr
            if 1.0 - y - r_i < 1e-6:
                forces[i, 1] -= lr
                
            # Neighbor forces: push apart if touching or nearly touching
            for j in range(n):
                if i == j:
                    continue
                dx = x - centers[j, 0]
                dy = y - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                
                # Buffer allows pre-emptive separation for expansion
                if dist < r_i + radii[j] + 1e-4:
                    if dist > 0:
                        fx = dx / dist
                        fy = dy / dist
                        forces[i, 0] += lr * fx
                        forces[i, 1] += lr * fy
                        
        # 4. Update centers and clamp
        centers += forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # 5. Cooling schedule
        if t % 500 == 0 and t > 0:
            expand_rate *= 0.96
            lr *= 0.96
            
    return centers, radii, np.sum(radii)

def run_packing():
    n = 26
    best_centers = None
    best_radii = None
    best_sum = 0.0
    
    # --- Initialization 1: Hexagonal Lattice ---
    # Pattern: 6, 5, 6, 5, 4 circles per row (sum = 26)
    c1 = np.zeros((n, 2))
    idx = 0
    row_counts = [6, 5, 6, 5, 4]
    for r, cnt in enumerate(row_counts):
        y = (r + 0.5) * 0.19
        x0 = 0.12 + (r % 2) * 0.09
        for c in range(cnt):
            if idx >= n:
                break
            x = x0 + c * 0.16
            if x < 1.0:
                c1[idx] = [x, y]
                idx += 1
                
    bc, br, bs = jostling_optimize(c1, n)
    best_centers, best_radii, best_sum = bc, br, bs
    
    # --- Initialization 2: Random Restarts ---
    np.random.seed(42)
    for _ in range(4):
        c_rand = np.random.rand(n, 2)
        bc2, br2, bs2 = jostling_optimize(c_rand, n)
        if bs2 > best_sum:
            best_centers, best_radii, best_sum = bc2, br2, bs2
            
    # --- Final Strict Projection ---
    # Ensures absolute validity before returning
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        max_r = min(x, 1.0 - x, y, 1.0 - y)
        for j in range(n):
            if i == j:
                continue
            dx = x - best_centers[j, 0]
            dy = y - best_centers[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            max_r = min(max_r, (dist - best_radii[j]) / 2.0)
        best_radii[i] = max(0.0, max_r)
        
    return best_centers, best_radii, float(np.sum(best_radii))
