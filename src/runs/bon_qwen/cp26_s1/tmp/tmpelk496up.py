import numpy as np

def compute_forces(centers, current_r):
    """Compute repulsive forces between circles and boundaries."""
    n = centers.shape[0]
    # Vector differences and distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-10)
    dist = np.sqrt(dist_sq)
    
    # Overlap calculation
    min_dist = 2.0 * current_r
    overlap = np.maximum(min_dist - dist, 0.0)
    
    # Direction vectors (unit)
    safe_dist = np.where(dist < 1e-5, 1e-5, dist)
    dir_mat = diff / safe_dist[:, :, np.newaxis]
    
    # Repulsion forces proportional to overlap
    force_mat = overlap[:, :, np.newaxis] * dir_mat * 2.0
    forces = np.sum(force_mat, axis=1)
    
    # Boundary repulsion
    bnd_coeff = 2.0
    for i in range(n):
        for d in range(2):
            if centers[i, d] < current_r:
                forces[i, d] += bnd_coeff * (current_r - centers[i, d])
            if centers[i, d] > 1.0 - current_r:
                forces[i, d] -= bnd_coeff * (centers[i, d] - (1.0 - current_r))
                
    return forces

def run_packing():
    n = 26
    # Hexagonal grid initialization with slight perturbation to break symmetry
    centers = []
    count = 0
    for i in range(6):
        y = (i + 0.5) / 6.0
        for j in range(5):
            if count >= n:
                break
            x = (j + 0.5) / 4.0 + (0.5 if i % 2 == 1 else 0.0) / 4.0
            centers.append([x + 0.001 * (i % 2), y + 0.001 * (j % 2)])
            count += 1
    centers = np.array(centers)
    
    lr = 0.025
    decay = 0.996
    steps = 5000
    
    # Annealing-like optimization: gradually increase target radius to pack tighter
    for step in range(steps):
        current_r = 0.082 + 0.020 * (step / steps)
        forces = compute_forces(centers, current_r)
        
        centers += forces * lr
        centers = np.clip(centers, 1e-6, 1.0 - 1e-6)
        lr *= decay
        
    # Compute exact maximal valid radii for each circle based on final positions
    radii = np.zeros(n)
    for i in range(n):
        r_bound = min(centers[i, 0], 1.0 - centers[i, 0], 
                      centers[i, 1], 1.0 - centers[i, 1])
        dists = np.linalg.norm(centers - centers[i], axis=1)
        dists[i] = np.inf
        r_circ = np.min(dists) / 2.0
        radii[i] = min(r_bound, r_circ)
        
    # Ensure strict non-negativity
    radii = np.maximum(radii, 0.0)
    
    return centers, radii, np.sum(radii)