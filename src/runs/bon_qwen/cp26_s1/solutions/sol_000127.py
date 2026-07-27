# sol_000127 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 92361807) state=5b6207b0 sum of radii=0.776245 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def compute_energy(centers, radii):
    """Compute quadratic overlap and boundary violation energy."""
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2) + 1e-8)
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = np.maximum(r_sum - dist, 0.0)
    energy = np.sum(overlap**2)
    
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    b_ol = np.maximum(r - x, 0.0)**2 + np.maximum(x + r - 1.0, 0.0)**2 + \
           np.maximum(r - y, 0.0)**2 + np.maximum(y + r - 1.0, 0.0)**2
    energy += np.sum(b_ol)
    return energy

def optimize_positions(centers, radii, n_iter=30):
    """Move circles to reduce overlaps and boundary violations."""
    lr = 0.05
    for _ in range(n_iter):
        grad = np.zeros_like(centers)
        
        # Inter-circle repulsion
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq + 1e-8)
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = r_sum - dist
        mask = overlap > 0
        overlap_rep = np.where(mask, overlap, 0.0)
        
        inv_dist = 1.0 / (dist + 1e-8)
        dir_force = diff * inv_dist[:, :, np.newaxis]
        force = dir_force * overlap_rep[:, :, np.newaxis]
        grad += np.sum(force, axis=1)
        
        # Boundary repulsion
        x, y = centers[:, 0], centers[:, 1]
        r = radii
        grad[:, 0] += np.where(x - r < 0, (x - r) * 30.0, 0.0)
        grad[:, 0] += np.where(x + r > 1, -(x + r - 1) * 30.0, 0.0)
        grad[:, 1] += np.where(y - r < 0, (y - r) * 30.0, 0.0)
        grad[:, 1] += np.where(y + r > 1, -(y + r - 1) * 30.0, 0.0)
        
        centers = centers + lr * grad
        centers = np.clip(centers, 0.0, 1.0)
        lr *= 0.96
    return centers

def run_packing():
    np.random.seed(42)
    n = 26
    
    # Initialize centers in a perturbed grid
    centers = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(6):
            if idx >= n:
                break
            centers[idx] = [0.1 + j * 0.15 + np.random.uniform(-0.01, 0.01),
                            0.1 + i * 0.20 + np.random.uniform(-0.01, 0.01)]
            idx += 1
    centers = np.clip(centers, 0.05, 0.95)
    radii = np.full(n, 0.02)
    
    # Iterative expansion
    growth = 1.0008
    for step in range(3000):
        radii *= growth
        centers = optimize_positions(centers, radii, n_iter=30)
        
        # Check feasibility and adjust growth
        if step > 500:
            if compute_energy(centers, radii) > 0.05:
                radii /= growth
                if growth > 1.0002:
                    growth *= 0.9995
                    
    # Final refinement
    centers = optimize_positions(centers, radii, n_iter=500)
    
    # Enforce strict boundary constraints
    r_max = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
    radii = np.minimum(radii, r_max)
    
    # Ensure no inter-circle overlap
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlap = r_sum - dist
    np.fill_diagonal(overlap, 0.0)
    max_ov = np.max(overlap)
    if max_ov > 1e-9:
        radii -= max_ov / 2.0 + 1e-7
        radii = np.maximum(radii, 0.0)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
