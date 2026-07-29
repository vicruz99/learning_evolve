# sol_000312 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 5cd869be) state=a4a38acf sum of radii=2.354675 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing():
    n = 26
    # Initialize centers on a perturbed grid for better convergence
    # A 6x5 grid provides 30 points; we take the first 26
    x = np.linspace(0.1, 0.9, 6)
    y = np.linspace(0.1, 0.9, 5)
    xx, yy = np.meshgrid(x, y)
    centers = np.vstack([xx.ravel(), yy.ravel()]).T[:n]
    
    # Break symmetry with small noise
    np.random.seed(42)
    centers += np.random.uniform(-0.01, 0.01, centers.shape)
    
    # Optimization parameters
    r = 0.05
    lr = 0.015
    growth = 0.0000018
    iters = 40000
    
    for step in range(iters):
        # Gradually increase target radius to pack tighter
        r += growth
        if r > 0.112:
            growth *= 0.995
            
        forces = np.zeros_like(centers)
        
        # Boundary repulsion forces
        mask_low = centers < r
        mask_high = centers > 1.0 - r
        forces[mask_low] += r - centers[mask_low]
        forces[mask_high] += (1.0 - r) - centers[mask_high]
        
        # Pairwise repulsion forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1.0)
        
        overlap = np.maximum(0.0, 2.0 * r - dist)
        inv_dist = 1.0 / np.maximum(dist, 1e-8)
        term = diff * overlap[:, :, np.newaxis] * inv_dist[:, :, np.newaxis]
        forces += 0.5 * np.sum(term, axis=1)
        
        # Update positions
        centers += forces * lr
        centers = np.clip(centers, r, 1.0 - r)
        
    # Compute exact feasible radii for the final configuration
    radii = np.zeros(n)
    for i in range(n):
        min_sep = 1.0
        # Distance to boundaries
        min_sep = min(min_sep, centers[i][0], 1.0 - centers[i][0], centers[i][1], 1.0 - centers[i][1])
        # Distance to other circles
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        dists = np.sqrt(np.sum((centers[mask] - centers[i])**2, axis=1))
        min_sep = min(min_sep, dists.min() * 0.5)
        radii[i] = max(0.0, min_sep - 1e-7)
        
    return centers, radii, float(np.sum(radii))
