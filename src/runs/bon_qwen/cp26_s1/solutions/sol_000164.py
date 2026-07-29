# sol_000164 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 3daa574a) state=9f0c4a8a sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_forces(centers, radii):
    """
    Compute repulsive forces to resolve overlaps and boundary violations.
    Forces are proportional to penetration depth.
    """
    N = centers.shape[0]
    forces = np.zeros((N, 2))
    
    # Pairwise distances and overlaps
    diff = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2) + 1e-12)
    np.fill_diagonal(dists, np.inf)
    
    sum_r = radii[:, None] + radii[None, :]
    overlap = np.maximum(0, sum_r - dists)
    
    idx_i, idx_j = np.where(overlap > 0)
    for ii, jj in zip(idx_i, idx_j):
        o = overlap[ii, jj]
        d = dists[ii, jj]
        dir_vec = diff[ii, jj] / d
        forces[ii] += o * dir_vec
        forces[jj] -= o * dir_vec
        
    # Boundary constraints
    for i in range(N):
        r = radii[i]
        x, y = centers[i]
        if x < r:
            forces[i, 0] += (r - x)
        if x > 1 - r:
            forces[i, 0] -= (x - (1 - r))
        if y < r:
            forces[i, 1] += (r - y)
        if y > 1 - r:
            forces[i, 1] -= (y - (1 - r))
            
    return forces

def run_packing():
    np.random.seed(42)
    N = 26
    best_sum = 0.0
    best_res = None
    
    # Run multiple trials with different perturbations to escape local minima
    for trial in range(5):
        centers = np.zeros((N, 2))
        idx = 0
        # Hexagonal lattice initialization for high packing density
        r_init = 0.096
        for i in range(6):
            for j in range(5):
                if idx >= N: 
                    break
                x = 0.13 + j * r_init * 2.0
                y = 0.13 + i * r_init * 1.7320508
                if i % 2 == 1:
                    x += r_init
                centers[idx] = [x, y]
                idx += 1
                
        # Add controlled randomness
        centers += np.random.uniform(-0.025, 0.025, centers.shape)
        centers = np.clip(centers, 0.05, 0.95)
        radii = np.full(N, r_init)
        
        lr = 0.035
        grow = 1.000015
        
        for step in range(6000):
            radii *= grow
            forces = compute_forces(centers, radii)
            
            # Clip forces to prevent instability during high-overlap phases
            max_force = np.max(np.abs(forces))
            if max_force > 0.8:
                forces = forces / max_force
                
            centers += lr * forces
            centers = np.clip(centers, 0.0, 1.0)
            
            # Random kicks to escape local minima
            if step % 300 == 0:
                centers += np.random.normal(0, 0.004, centers.shape)
                centers = np.clip(centers, 0.0, 1.0)
                
            # Cooling schedule
            lr *= 0.9995
            if step % 800 == 0:
                grow *= 0.97
                
        current_sum = np.sum(radii)
        if current_sum > best_sum:
            best_sum = current_sum
            best_res = (centers, radii, current_sum)
            
    return best_res
