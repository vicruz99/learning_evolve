# sol_000170 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 8150d860) state=b42db275 sum of radii=0.000000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize sum of radii.
    Strategy: 
    1. Initialize centers in a hexagonal grid pattern.
    2. Iteratively expand radii and use force-directed relaxation to resolve overlaps.
    3. Post-process to ensure strict validity.
    """
    n = 26
    np.random.seed(42) # Deterministic randomness

    # 1. Initialization
    centers = np.zeros((n, 2))
    idx = 0
    r_init = 0.08 # Initial radius for grid placement
    
    # Hexagonal packing setup
    # Row height = r * sqrt(3)
    row_h = r_init * np.sqrt(3)
    row = 0
    
    while idx < n:
        y = r_init + row * row_h
        if y + r_init > 1.0:
            # Cannot fit more rows with this radius, stop grid filling
            break
        
        # Odd rows shifted by r_init (actually 2*r_init start x)
        # Row 0 starts at r. Row 1 starts at 2r.
        x_start = r_init if row % 2 == 0 else 2 * r_init
        x = x_start
        while x + r_init <= 1.0:
            centers[idx] = [x, y]
            idx += 1
            x += 2 * r_init
            if idx >= n: break
        row += 1
        
    # Fill remaining circles randomly if grid didn't fill all 26
    while idx < n:
        best_c = None
        best_d = 0
        # Try to find a spot far from existing circles
        for _ in range(200):
            c = np.random.rand(2) * 0.9 + 0.05
            min_d = 1.0
            for k in range(n):
                if k < idx:
                    d = np.linalg.norm(centers[k] - c)
                    if d < min_d: min_d = d
            if min_d > best_d:
                best_d = min_d
                best_c = c
        if best_c is not None:
            centers[idx] = best_c
            idx += 1
        else:
            # Fallback random
            centers[idx] = np.random.rand(2) * 0.8 + 0.1
            idx += 1

    # 2. Optimization Loop
    radii = np.ones(n) * 0.05 # Start with small radius
    
    dt = 0.05
    repulsion_k = 15.0
    wall_k = 15.0
    expansion_rate = 1.0002 
    max_steps = 5000
    
    for step in range(max_steps):
        # Expand radii
        radii *= expansion_rate
        
        # Calculate forces
        forces = np.zeros_like(centers)
        
        # Vectorized distance calculation
        # diff[i, j] = centers[i] - centers[j]
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] 
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Set diagonal to inf to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        # Overlap calculation
        # r_i + r_j - dist
        radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        overlap = radii_sum - dists
        
        # We only apply repulsion if overlap > 0
        active_overlap = np.maximum(0, overlap)
        
        # Direction vectors (normalized)
        safe_dists = np.where(dists < 1e-9, 1e-9, dists)
        norms = diff / safe_dists[:, :, np.newaxis]
        
        # Force contribution from each pair
        # F_ij = overlap * k * direction
        force_contribs = active_overlap[:, :, np.newaxis] * norms * repulsion_k
        
        # Sum forces on each circle i
        forces += np.sum(force_contribs, axis=1)
        
        # Wall repulsion forces
        # Left wall (x < r)
        mask = centers[:, 0] < radii
        forces[mask, 0] += wall_k * (radii[mask] - centers[mask, 0])
        # Right wall (x > 1-r)
        mask = centers[:, 0] > 1.0 - radii
        forces[mask, 0] -= wall_k * (centers[mask, 0] - (1.0 - radii[mask]))
        # Bottom wall (y < r)
        mask = centers[:, 1] < radii
        forces[mask, 1] += wall_k * (radii[mask] - centers[mask, 1])
        # Top wall (y > 1-r)
        mask = centers[:, 1] > 1.0 - radii
        forces[mask, 1] -= wall_k * (centers[mask, 1] - (1.0 - radii[mask]))
        
        # Limit force magnitude to prevent instability
        force_norms = np.linalg.norm(forces, axis=1)
        max_f = 10.0
        scale = np.where(force_norms > max_f, max_f / force_norms, 1.0)
        forces *= scale[:, np.newaxis]
        
        # Update centers
        centers += dt * forces
        centers = np.clip(centers, 0.0, 1.0)
        
        # Add small random noise to escape local minima
        if step % 100 == 0:
            centers += np.random.randn(*centers.shape) * 0.005
            centers = np.clip(centers, 0.0, 1.0)

    # 3. Final Correction to ensure validity
    # Check overlaps and boundary violations
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dists, np.inf)
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    overlaps = radii_sum - dists
    max_ovl = np.max(overlaps)
    
    bound_ovls = np.zeros(n)
    bound_ovls += np.maximum(0, radii - centers[:, 0])
    bound_ovls += np.maximum(0, radii - (1.0 - centers[:, 0]))
    bound_ovls += np.maximum(0, radii - centers[:, 1])
    bound_ovls += np.maximum(0, radii - (1.0 - centers[:, 1]))
    max_bound = np.max(bound_ovls)
    
    max_violation = max(max_ovl, max_bound)
    
    if max_violation > 1e-7:
        # Reduce radii to resolve violations
        # For overlap, delta = overlap / 2
        # For boundary, delta = violation
        delta = max(max_ovl / 2.0, max_bound)
        radii -= delta
        radii = np.maximum(radii, 0.0)
        
    sum_radii = np.sum(radii)
    return centers, radii, sum_radii
