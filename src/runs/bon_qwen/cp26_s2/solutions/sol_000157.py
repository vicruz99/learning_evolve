# sol_000157 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a0a8497a) state=88a900e0 sum of radii=1.881605 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26
    np.random.seed(42)

    # 1. Initialize centers in a scaled hexagonal lattice
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    k = 0
    y = 0
    r_init = 0.08
    # Generate rows
    while k < n:
        x = 0
        row_len = 0
        while x + 2 * r_init <= 1 and k < n:
            centers[k] = [x + r_init, y + r_init]
            radii[k] = r_init
            k += 1
            x += 2 * r_init
            row_len += 1
        y += np.sqrt(3) * r_init
        if y + 2 * r_init > 1:
            break

    # If we didn't fill 26, add them randomly
    if k < n:
        for i in range(k, n):
            centers[i] = np.random.rand(2) * 0.8 + 0.1
            radii[i] = 0.05

    # 2. Iterative Expansion
    step_sizes = np.ones(n) * 0.005
    perturbation_scale = 0.05
    
    for epoch in range(500):
        # Decay perturbation
        if epoch > 200:
            perturbation_scale *= 0.95

        for i in range(n):
            # Attempt to increase radius
            r_new = radii[i] + step_sizes[i]
            cx, cy = centers[i]
            
            # Check bounds
            if cx - r_new < 0 or cx + r_new > 1 or cy - r_new < 0 or cy + r_new > 1:
                continue
            
            # Check overlaps
            valid = True
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < r_new + radii[j] - 1e-9:
                    valid = False
                    break
            
            if valid:
                radii[i] = r_new
            else:
                # Repulsion / Push logic
                for j in range(n):
                    if i == j: continue
                    dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                    if dist < r_new + radii[j]:
                        if dist > 1e-9:
                            overlap = r_new + radii[j] - dist
                            # Move i away from j
                            centers[i] += (centers[i] - centers[j]) / dist * overlap * 0.5
                # Perturbation to avoid local minima
                centers[i] += np.random.randn(2) * perturbation_scale

        # Clamp centers and adjust radii to fit walls
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
            
            # Recalculate radius based on current position and neighbors
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                max_r = min(max_r, (dist - radii[j]))
            
            radii[i] = max(1e-5, max_r)

    # 3. Gradient Refinement
    # We refine positions to maximize sum of radii by resolving tight constraints
    current_sum = np.sum(radii)
    for _ in range(200):
        grad_sum = 0
        for i in range(n):
            grad_i = np.zeros(2)
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < 1e-9: continue
                
                # If tight, gradient points away
                gap = dist - radii[i] - radii[j]
                if gap < 0.002: # Active constraint
                    grad_i += (centers[i] - centers[j]) / dist
                    
                # Wall constraints
                for wall_idx in range(2):
                    if wall_idx == 0: # x
                        if centers[i, wall_idx] < radii[i] + 0.001:
                            grad_i[wall_idx] += 1
                        elif centers[i, wall_idx] > 1 - radii[i] - 0.001:
                            grad_i[wall_idx] -= 1
                    else: # y
                        if centers[i, wall_idx] < radii[i] + 0.001:
                            grad_i[wall_idx] += 1
                        elif centers[i, wall_idx] > 1 - radii[i] - 0.001:
                            grad_i[wall_idx] -= 1
            
            centers[i] += grad_i * 0.005

        # Recalculate radii after move
        for i in range(n):
            r = radii[i]
            centers[i, 0] = np.clip(centers[i, 0], r, 1 - r)
            centers[i, 1] = np.clip(centers[i, 1], r, 1 - r)
            
            max_r = min(centers[i, 0], 1 - centers[i, 0], centers[i, 1], 1 - centers[i, 1])
            for j in range(n):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                max_r = min(max_r, (dist - radii[j]))
            radii[i] = max(1e-5, max_r)

    return centers, radii, np.sum(radii)
