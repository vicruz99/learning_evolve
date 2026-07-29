# sol_000309 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e3d19f45) state=138676ee sum of radii=2.397366 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def simulate(centers, radii, pairs, n, steps=100, rep_str=2000, bound_str=2000):
    """
    Simulate repulsive forces to resolve overlaps and boundary violations.
    
    Args:
        centers: np.array of shape (n, 2)
        radii: np.array of shape (n)
        pairs: np.array of shape (m, 2) containing indices of all circle pairs
        n: number of circles
        steps: number of simulation steps
        rep_str: repulsion strength multiplier
        bound_str: boundary force strength multiplier
        
    Returns:
        Updated centers
    """
    c = centers.copy()
    # Precompute sum of radii for each pair
    pair_radii_sum = radii[pairs[:, 0]] + radii[pairs[:, 1]]
    
    for _ in range(steps):
        forces = np.zeros((n, 2))
        
        # Vectorized force calculation for pairs
        c1 = c[pairs[:, 0]]
        c2 = c[pairs[:, 1]]
        diff = c1 - c2
        dist_sq = np.sum(diff**2, axis=1)
        dist = np.sqrt(dist_sq)
        
        # Avoid division by zero
        safe_dist = np.where(dist < 1e-10, 1e-10, dist)
        dir_vec = diff / safe_dist[:, np.newaxis]
        
        # Calculate overlap (positive if overlapping)
        overlap = pair_radii_sum - dist
        
        # Force magnitude proportional to overlap, capped for stability
        f_mag = np.clip(np.maximum(0, overlap) * rep_str, 0, 20.0)
        
        f1 = dir_vec * f_mag[:, np.newaxis]
        f2 = -f1
        
        # Accumulate forces
        np.add.at(forces, pairs[:, 0], f1)
        np.add.at(forces, pairs[:, 1], f2)
        
        # Boundary forces
        # Left boundary (x < r)
        d_l = c[:, 0] - radii
        forces[:, 0] += np.minimum(d_l, 0) * bound_str
        # Right boundary (x > 1-r)
        d_r = 1.0 - c[:, 0] - radii
        forces[:, 0] += np.minimum(d_r, 0) * bound_str
        # Bottom boundary (y < r)
        d_b = c[:, 1] - radii
        forces[:, 1] += np.minimum(d_b, 0) * bound_str
        # Top boundary (y > 1-r)
        d_t = 1.0 - c[:, 1] - radii
        forces[:, 1] += np.minimum(d_t, 0) * bound_str
        
        # Update positions
        c += forces * 0.0005 
        
        # Clip positions to strictly satisfy boundary constraints
        lower = radii[:, np.newaxis]
        upper = 1.0 - lower
        c = np.clip(c, lower, upper)
        
    return c

def run_packing() -> tuple:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
    """
    n = 26
    np.random.seed(42) # Fixed seed for reproducibility

    # Initial small radius
    r_init = 0.02
    radii = np.full(n, r_init)
    centers = np.zeros((n, 2))
    
    # Initialize positions in a perturbed 6x5 grid
    cols, rows = 6, 5
    idx = 0
    for r_idx in range(rows):
        for c_idx in range(cols):
            if idx >= n: 
                break
            # Center in cell
            cx = (c_idx + 0.5) / cols
            cy = (r_idx + 0.5) / rows
            # Add noise to break symmetry and encourage denser packing
            centers[idx, 0] = cx + np.random.uniform(-0.02, 0.02)
            centers[idx, 1] = cy + np.random.uniform(-0.02, 0.02)
            idx += 1
        if idx >= n:
            break
            
    # Ensure initial positions are valid
    lower = radii[:, np.newaxis]
    upper = 1.0 - lower
    centers = np.clip(centers, lower, upper)

    # Precompute all unique pairs of indices
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((i, j))
    pairs = np.array(pairs)
    
    growth_step = 0.001
    
    # Iterative growth loop
    # Try to increase radii, resolve overlaps, and check validity
    for _ in range(500):
        # Attempt to grow radii
        radii += growth_step
        
        # Run simulation to resolve overlaps caused by growth
        centers = simulate(centers, radii, pairs, n, steps=50, rep_str=2000, bound_str=2000)
        
        # Check validity
        valid = True
        
        # Check boundaries
        # x bounds
        if np.any(centers[:, 0] < radii - 1e-9) or np.any(centers[:, 0] > 1 - radii + 1e-9):
            valid = False
        # y bounds
        if np.any(centers[:, 1] < radii - 1e-9) or np.any(centers[:, 1] > 1 - radii + 1e-9):
            valid = False
            
        # Check overlaps
        if valid:
            c1 = centers[pairs[:, 0]]
            c2 = centers[pairs[:, 1]]
            dists = np.sqrt(np.sum((c1-c2)**2, axis=1))
            min_dists = radii[pairs[:, 0]] + radii[pairs[:, 1]]
            if np.any(dists < min_dists - 1e-9):
                valid = False
        
        if valid:
            # Configuration is valid, accept the growth
            pass
        else:
            # Configuration invalid, revert radii
            radii -= growth_step
            # Refine positions with smaller radii to find a better local arrangement
            centers = simulate(centers, radii, pairs, n, steps=100, rep_str=3000, bound_str=3000)
            # Reduce growth step to approach limit more precisely
            growth_step *= 0.9
            
        # If growth step is too small, stop and optimize final positions
        if growth_step < 1e-7:
            centers = simulate(centers, radii, pairs, n, steps=2000, rep_str=5000, bound_str=5000)
            break
            
    # Final high-precision simulation to ensure strict validity and optimal packing
    centers = simulate(centers, radii, pairs, n, steps=2000, rep_str=5000, bound_str=5000)
    
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
