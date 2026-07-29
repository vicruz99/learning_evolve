# sol_000021 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 88e7083c) state=312bc650 sum of radii=1.328830 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    np.random.seed(42)
    
    # Initialize centers in a hexagonal-like grid
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.035)
    
    idx = 0
    rows = [5, 4, 5, 4, 5, 3]
    y = 0.1
    dy = 0.18
    dx = 0.2
    for r_idx, count in enumerate(rows):
        x_start = (1.0 - (count - 1) * dx) / 2.0
        for c in range(count):
            if idx < n:
                centers[idx] = [x_start + c * dx, y]
                idx += 1
        y += dy
        
    # Simulation parameters
    dt = 0.008
    growth_rate = 0.00015
    k_rep = 800.0
    k_bound = 1000.0
    damping = 0.82
    velocities = np.zeros((n, 2))
    
    iterations = 12000
    for step in range(iterations):
        # Uniform growth of radii
        radii *= (1.0 + growth_rate)
        
        forces = np.zeros((n, 2))
        
        # Pairwise repulsion forces
        for i in range(n):
            for j in range(i + 1, n):
                diff = centers[i] - centers[j]
                dist_sq = diff[0]**2 + diff[1]**2
                dist = np.sqrt(dist_sq)
                min_dist = radii[i] + radii[j]
                
                if dist < min_dist and dist > 1e-7:
                    overlap = min_dist - dist
                    f_mag = overlap * k_rep
                    inv_dist = 1.0 / dist
                    forces[i, 0] += f_mag * diff[0] * inv_dist
                    forces[i, 1] += f_mag * diff[1] * inv_dist
                    forces[j, 0] -= f_mag * diff[0] * inv_dist
                    forces[j, 1] -= f_mag * diff[1] * inv_dist
        
        # Boundary repulsion forces
        for i in range(n):
            x, y_c = centers[i]
            r = radii[i]
            if x - r < 0:
                forces[i, 0] += (r - x) * k_bound
            if x + r > 1:
                forces[i, 0] -= (x + r - 1) * k_bound
            if y_c - r < 0:
                forces[i, 1] += (r - y_c) * k_bound
            if y_c + r > 1:
                forces[i, 1] -= (y_c + r - 1) * k_bound
                
        # Update velocities and positions
        velocities = velocities * damping + forces * dt
        centers += velocities * dt
        
        # Clamp positions to unit square
        centers[:, 0] = np.clip(centers[:, 0], 0, 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, 1)
        
        # Annealing schedule
        if step % 500 == 0:
            growth_rate *= 0.92
            k_rep *= 1.15
            damping *= 0.99
            
        # Periodic shrinkage to escape local minima (simulated annealing)
        if step > 4000 and step % 800 == 0:
            radii *= 0.96
            
    # Final rigorous overlap resolution
    for _ in range(500):
        max_ov = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < radii[i] + radii[j]:
                    ov = radii[i] + radii[j] - d
                    if ov > max_ov:
                        max_ov = ov
        if max_ov > 1e-8:
            radii -= max_ov / 2.0
        else:
            break
            
    # Final boundary projection
    for i in range(n):
        r = radii[i]
        r = max(0.0, r)
        radii[i] = r
        centers[i, 0] = max(r, min(1.0 - r, centers[i, 0]))
        centers[i, 1] = max(r, min(1.0 - r, centers[i, 1]))
        
    return centers, radii, float(np.sum(radii))
