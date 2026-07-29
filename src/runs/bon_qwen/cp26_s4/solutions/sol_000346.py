# sol_000346 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 9cbd6fd8) state=0733de5d sum of radii=2.426803 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_radii(centers):
    """Compute the maximum possible radius for each circle given current centers."""
    n = centers.shape[0]
    radii = np.full(n, 1.0)
    for i in range(n):
        x, y = centers[i]
        # Distance to boundaries
        r = min(x, 1.0 - x, y, 1.0 - y)
        # Distance to other circles
        for j in range(n):
            if i != j:
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                if dist < r * 2.0:
                    r = dist * 0.5
        radii[i] = r
    return radii

def get_repulsion_forces(centers, target_dist):
    """Compute repulsive forces to keep circles separated by target_dist."""
    n = centers.shape[0]
    forces = np.zeros_like(centers)
    
    for i in range(n):
        # Boundary repulsion
        x, y = centers[i]
        if x < target_dist:
            forces[i, 0] += target_dist - x
        if x > 1.0 - target_dist:
            forces[i, 0] -= x - (1.0 - target_dist)
        if y < target_dist:
            forces[i, 1] += target_dist - y
        if y > 1.0 - target_dist:
            forces[i, 1] -= y - (1.0 - target_dist)
            
        # Neighbor repulsion
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx * dx + dy * dy
            dist = np.sqrt(dist_sq) if dist_sq > 1e-12 else 1e-9
            
            if dist < target_dist * 2.0:
                overlap = target_dist * 2.0 - dist
                f = overlap / dist
                forces[i, 0] += f * dx
                forces[i, 1] += f * dy
                forces[j, 0] -= f * dx
                forces[j, 1] -= f * dy
    return forces

def run_packing():
    n = 26
    centers = np.zeros((n, 2))
    
    # 1. Hexagonal grid initialization
    # 5 rows, staggered columns for high initial density
    idx = 0
    for row in range(5):
        y = 0.15 + row * 0.175
        num_cols = 5 if row % 2 == 0 else 6
        x_start = 0.12 if row % 2 == 0 else 0.185
        for col in range(num_cols):
            if idx >= n:
                break
            centers[idx, 0] = x_start + col * 0.175
            centers[idx, 1] = y
            idx += 1
            
    # 2. Force-directed expansion
    R = 0.05
    step_size = 0.15
    for iteration in range(8000):
        forces = get_repulsion_forces(centers, R)
        centers += step_size * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Annealing schedule
        if iteration % 200 == 0:
            R += 0.0008
            step_size *= 0.995
            
        # Reduce step size for fine tuning in later stages
        if iteration > 5000:
            step_size *= 0.9995

    # 3. Compute initial radii
    radii = compute_radii(centers)
    current_sum = np.sum(radii)
    
    # 4. Local hill-climbing optimization to maximize sum of radii
    delta = 0.002
    for _ in range(5000):
        improved = False
        for i in range(n):
            best_sum = current_sum
            best_c = centers[i].copy()
            
            # Try small perturbations
            for dx in [-delta, 0, delta]:
                for dy in [-delta, 0, delta]:
                    if dx == 0 and dy == 0:
                        continue
                    trial_c = best_c + np.array([dx, dy])
                    if trial_c[0] < 0 or trial_c[0] > 1 or trial_c[1] < 0 or trial_c[1] > 1:
                        continue
                    
                    old_c = centers[i]
                    centers[i] = trial_c
                    trial_radii = compute_radii(centers)
                    trial_sum = np.sum(trial_radii)
                    
                    if trial_sum > best_sum + 1e-9:
                        best_sum = trial_sum
                        best_c = trial_c
                        improved = True
                    centers[i] = old_c
            
            centers[i] = best_c
            
        if improved:
            radii = compute_radii(centers)
            current_sum = np.sum(radii)
            delta *= 0.998
        else:
            # Coarse to fine reduction if stuck
            delta *= 0.5
            if delta < 1e-6:
                break
                
    # Final radius computation
    radii = compute_radii(centers)
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
