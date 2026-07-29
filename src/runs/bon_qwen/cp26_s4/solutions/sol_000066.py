# sol_000066 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 10bf7585) state=2c3553aa sum of radii=0.957369 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def dist_sq(centers):
    """Computes pairwise squared distances."""
    return np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=2)

def optimize_positions_for_radius(r, seed, max_iter=500, step=0.01):
    """Local search to find valid positions for 26 circles of radius r."""
    rng = np.random.RandomState(seed)
    centers = rng.uniform(r, 1 - r, (26, 2))
    
    # Initial relaxation
    for _ in range(200):
        forces = np.zeros((26, 2))
        
        # Pairwise repulsion
        dists = dist_sq(centers)
        # Lower triangle indices
        i, j = np.triu_indices(26, 1)
        overlaps = r + r - np.sqrt(dists[i, j])
        overlaps = np.clip(overlaps, 0, None)
        
        if np.sum(overlaps) > 0:
            dir_vecs = centers[i] - centers[j]
            lengths = np.linalg.norm(dir_vecs, axis=1, keepdims=True) + 1e-8
            unit_vecs = dir_vecs / lengths
            pushes = overlaps[:, None] * unit_vecs
            np.add.at(forces, i, pushes)
            np.add.at(forces, j, -pushes)
            
        # Boundary repulsion
        forces[:, 0] += np.clip(r - centers[:, 0], 0, None)
        forces[:, 0] -= np.clip(centers[:, 0] - (1 - r), 0, None)
        forces[:, 1] += np.clip(r - centers[:, 1], 0, None)
        forces[:, 1] -= np.clip(centers[:, 1] - (1 - r), 0, None)
        
        centers += step * forces
        centers = np.clip(centers, r, 1 - r)
        
    # Return centers and max overlap
    overlaps = r + r - np.sqrt(dist_sq(centers)[np.triu_indices(26, 1)])
    overlaps = np.maximum(0, overlaps)
    return centers, np.max(overlaps)

def refine_unequal_radii(centers, radii, steps=2000, growth_rate=1e-4):
    """Refine a packing by allowing radii to grow and repelling circles."""
    for step in range(steps):
        # Grow radii slightly
        radii += growth_rate
        
        # Calculate forces
        forces = np.zeros((26, 2))
        
        # Pairwise repulsion
        dists = dist_sq(centers)
        i, j = np.triu_indices(26, 1)
        r_sum = radii[i] + radii[j]
        d = np.sqrt(dists[i, j]) + 1e-8
        overlap = r_sum - d
        push_strength = np.maximum(0, overlap)
        
        if np.sum(push_strength) > 0:
            dir_vecs = centers[i] - centers[j]
            unit_vecs = dir_vecs / d[:, None]
            pushes = push_strength[:, None] * unit_vecs
            np.add.at(forces, i, pushes)
            np.add.at(forces, j, -pushes)
            
        # Boundary repulsion
        for k in range(26):
            r_k = radii[k]
            # x-axis
            if centers[k, 0] < r_k:
                forces[k, 0] += (r_k - centers[k, 0])
            elif centers[k, 0] > 1 - r_k:
                forces[k, 0] -= (centers[k, 0] - (1 - r_k))
            # y-axis
            if centers[k, 1] < r_k:
                forces[k, 1] += (r_k - centers[k, 1])
            elif centers[k, 1] > 1 - r_k:
                forces[k, 1] -= (centers[k, 1] - (1 - r_k))
                
        # Update centers
        centers += 0.05 * forces
        centers = np.clip(centers, radii[:, None], 1 - radii[:, None])
        
        # If overlap is too high, slightly reduce radii to recover
        if np.any(overlap > 1e-4):
            radii -= 2 * growth_rate
            radii = np.maximum(radii, 1e-5)

    return centers, radii

def run_packing():
    # Stage 1: Find best equal radius
    best_r = 0.0
    best_centers = None
    
    # Binary search for r
    low, high = 0.0, 0.12
    for _ in range(15):
        mid = (low + high) / 2
        valid = False
        # Try multiple seeds to avoid local minima
        for seed in range(10):
            centers, max_ov = optimize_positions_for_radius(mid, seed)
            if max_ov < 1e-5:
                valid = True
                best_r = mid
                best_centers = centers
                break
        if valid:
            low = mid
        else:
            high = mid

    # Stage 2: Refine with unequal radii
    radii = np.full(26, best_r)
    final_centers, final_radii = refine_unequal_radii(best_centers, radii)
    
    # Final safety check and projection
    # Ensure circles are inside the square
    final_radii = np.minimum(final_radii, final_centers[:, 0])
    final_radii = np.minimum(final_radii, 1 - final_centers[:, 0])
    final_radii = np.minimum(final_radii, final_centers[:, 1])
    final_radii = np.minimum(final_radii, 1 - final_centers[:, 1])
    final_radii = np.maximum(final_radii, 1e-6)

    sum_radii = np.sum(final_radii)
    
    return final_centers, final_radii, sum_radii
