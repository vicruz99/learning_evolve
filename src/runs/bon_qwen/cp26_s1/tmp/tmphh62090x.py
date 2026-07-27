import numpy as np

def run_packing():
    n = 26
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Initialization: Hexagonal lattice
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Initial radius
    
    count = 0
    r_init = 0.05
    y = r_init
    row = 0
    # Generate points in a hexagonal pattern
    while count < n:
        x = r_init
        # Shift odd rows by radius to create hexagonal packing
        if row % 2 == 1:
            x = r_init + r_init 
        else:
            x = r_init
        
        # Place circles in the row
        while x <= 1 - r_init and count < n:
            centers[count] = [x, y]
            radii[count] = r_init
            count += 1
            x += 2 * r_init
        
        # Move to next row
        y += r_init * np.sqrt(3)
        row += 1
        
    # Fill any remaining spots (should be rare with this density)
    while count < n:
        centers[count] = np.random.uniform(r_init, 1-r_init, 2)
        radii[count] = r_init
        count += 1

    # Optimization phases
    # Phase 1: Find arrangement
    # Phase 2: Tighten packing
    # Phase 3: Resolve overlaps with high penalty
    phases = [
        {'steps': 3000, 'alpha': 0.05, 'K': 50},
        {'steps': 3000, 'alpha': 0.01, 'K': 500},
        {'steps': 3000, 'alpha': 0.002, 'K': 5000},
    ]
    
    for phase in phases:
        steps = phase['steps']
        alpha = phase['alpha']
        K = phase['K']
        
        for step in range(steps):
            # Vectorized distance computation
            # diff[i, j] = centers[i] - centers[j]
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            
            # Set diagonal to infinity to ignore self-interaction
            np.fill_diagonal(dists, np.inf)
            
            # Overlap calculation: positive if overlapping
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            overlaps = np.maximum(0, r_sum - dists)
            
            # Compute forces
            # Direction from j to i: (c_i - c_j) / d_ij
            safe_dists = np.maximum(dists, 1e-9)
            directions = diff / safe_dists[:, :, np.newaxis]
            
            # Repulsive force proportional to overlap
            # F_ij = 2 * K * overlap_ij * direction_ij
            forces = np.sum(2 * K * overlaps[:, :, np.newaxis] * directions, axis=1)
            
            # Wall violations
            viol_left = np.maximum(0, radii - centers[:, 0])
            viol_right = np.maximum(0, centers[:, 0] + radii - 1)
            viol_bottom = np.maximum(0, radii - centers[:, 1])
            viol_top = np.maximum(0, centers[:, 1] + radii - 1)
            
            # Wall position gradients (pushing away from walls)
            # Gradient of penalty wrt x: 2K * (viol_right - viol_left)