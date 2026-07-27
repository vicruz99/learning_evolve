import numpy as np

def compute_forces(centers, radii):
    """
    Compute repulsive forces between circles and from boundaries.
    Returns an (N, 2) array of forces.
    """
    n = len(radii)
    forces = np.zeros_like(centers)
    
    # Vectorized pairwise distances and differences
    diffs = centers[:, None, :] - centers[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    min_dists = radii[:, None] + radii[None, :]
    overlaps = np.maximum(0, min_dists - dists)
    
    # Avoid division by zero
    dists_safe = np.where(dists > 1e-9, dists, 1.0)
    directions = diffs / dists_safe[:, :, None]
    
    # Inter-circle repulsion
    force_matrix = overlaps[:, :, None] * directions
    forces += np.sum(force_matrix, axis=1)
    
    # Boundary repulsion
    forces[:, 0] += np.maximum(0, radii - centers[:, 0])
    forces[:, 0] -= np.maximum(0, centers[:, 0] - (1 - radii))
    forces[:, 1] += np.maximum(0, radii - centers[:, 1])
    forces[:, 1] -= np.maximum(0, centers[:, 1] - (1 - radii))
    
    return forces

def run_packing():
    np.random.seed(42)
    N = 26
    
    # Initialize with a structured hexagonal pattern plus small jitter
    centers = np.zeros((N, 2))
    idx = 0
    rows = [5, 6, 5, 6, 4]
    y_pos = 0.2
    dy = 0.18
    for i, cnt in enumerate(rows):
        x_start = 0.15 + (i % 2) * 0.08
        dx = 0.7 / (cnt - 1) if cnt > 1 else 0
        for j in range(cnt):
            centers[idx] = [x_start + j * dx, y_pos + i * dy]
            idx += 1
    centers += np.random.randn(N, 2) * 0.005
    
    radii = np.full(N, 0.02)
    
    lr = 0.08
    base_grow = 0.0005
    
    for step in range(12000):
        # Growth factor decays over time to fine-tune the packing
        current_grow = 1.0 + base_grow / (1.0 + step / 2000.0)
        radii *= current_grow
        
        # Resolve overlaps and push away from boundaries
        forces = compute_forces(centers, radii)
        centers += lr * forces
        
        # Enforce hard constraints
        centers[:, 0] = np.clip(centers[:, 0], radii, 1 - radii)
        centers[:, 1] = np.clip(centers[:, 1], radii, 1 - radii)
        
        # Cool down learning rate
        if step % 2000 == 0 and step > 0:
            lr *= 0.75
            
    # Final safety margin to strictly satisfy numerical tolerance
    radii *= 0.99998
    return centers, radii, float(np.sum(radii))