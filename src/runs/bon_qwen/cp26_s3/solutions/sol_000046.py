# sol_000046 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state adcd3d40) state=b7abe1da sum of radii=0.339040 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Pack 26 circles in a unit square to maximize sum of radii.
    Uses a force-directed expanding circle packing algorithm with hexagonal initialization.
    """
    np.random.seed(42)
    n = 26
    
    # Initialize centers on a hexagonal grid to encourage dense packing
    s = 0.12 # Initial spacing
    row_h = s * math.sqrt(3) / 2
    candidates = []
    
    # Generate grid points covering [0,1]x[0,1]
    start_x, start_y = 0.05, 0.05
    
    # Determine grid dimensions
    cols = int((1.0 - 2*start_x) / s) + 2
    rows = int((1.0 - 2*start_y) / row_h) + 2
    
    for j in range(rows):
        y = start_y + j * row_h
        for i in range(cols):
            x = start_x + i * s + (j % 2) * (s / 2.0)
            if 0 <= x <= 1 and 0 <= y <= 1:
                candidates.append([x, y])
    
    candidates = np.array(candidates)
    
    if len(candidates) >= n:
        idx = np.random.choice(len(candidates), n, replace=False)
        centers = candidates[idx].copy()
    else:
        centers = np.random.rand(n, 2) * 0.8 + 0.1

    radii = np.ones(n) * 0.005 # Start with very small radii
    
    max_iter = 10000
    growth_rate = 4e-5 
    repulsion_strength = 1000.0 
    boundary_strength = 1000.0
    damping = 0.8
    velocity = np.zeros_like(centers)
    
    best_centers = centers.copy()
    best_radii = radii.copy()
    best_sum = 0.0
    
    # Preallocate arrays for performance
    diff = np.zeros((n, n, 2))
    dist_sq = np.zeros((n, n))
    dist = np.zeros((n, n))
    sum_radii = np.zeros((n, n))
    mask = ~np.eye(n, dtype=bool)
    
    # Check initial validity
    is_valid_initial = True
    if np.any(centers[:, 0] + radii > 1.0 + 1e-9) or \
       np.any(centers[:, 0] - radii < -1e-9) or \
       np.any(centers[:, 1] + radii > 1.0 + 1e-9) or \
       np.any(centers[:, 1] - radii < -1e-9):
        is_valid_initial = False
    
    if is_valid_initial:
        d = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        d2 = np.sum(d**2, axis=2)
        sr = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Check if dist < r1 + r2 - 1e-12 (invalid)
        # Equivalent to dist + 1e-12 < r1 + r2
        if np.any(np.sqrt(d2 + 1e-12) + 1e-12 < sr):
            is_valid_initial = False
            
    if is_valid_initial:
        best_sum = np.sum(radii)
    
    for t in range(max_iter):
        # 1. Compute differences between centers
        diff[:] = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        
        # 2. Compute distances
        dist_sq[:] = np.sum(diff**2, axis=2)
        dist[:] = np.sqrt(dist_sq + 1e-12)
        
        # 3. Sum of radii matrix
        sum_radii[:] = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # 4. Compute repulsive forces
        inv_dist = np.where(dist > 1e-12, 1.0 / dist, 0.0)
        dir_vec = diff * inv_dist[:, :, np.newaxis]
        
        # Overlap amount: positive if r1 + r2 > dist
        overlap = sum_radii - dist
        overlap_mag = np.maximum(0.0, overlap)
        
        # Force magnitude proportional to overlap
        force_mag = repulsion_strength * overlap_mag
        
        # Sum forces: force on i is sum of forces from all j
        forces = np.sum(force_mag[:, :, np.newaxis] * dir_vec, axis=1)
        
        # 5. Compute boundary forces
        # Push circles away from walls if they protrude
        forces[:, 0] += boundary_strength * np.maximum(0.0, radii - centers[:, 0])
        forces[:, 0] -= boundary_strength * np.maximum(0.0, centers[:, 0] + radii - 1.0)
        forces[:, 1] += boundary_strength * np.maximum(0.0, radii - centers[:, 1])
        forces[:, 1] -= boundary_strength * np.maximum(0.0, centers[:, 1] + radii - 1.0)
        
        # 6. Update velocity and position
        velocity = damping * velocity + forces * 0.0005
        centers += velocity
        
        # 7. Grow radii
        # If max overlap is high, pause growth to let forces resolve collisions
        max_ov = np.max(overlap_mag[mask])
        
        if max_ov > 0.02:
            current_growth = 0.0
        else:
            # Adaptive growth: slower as we approach collisions
            current_growth = growth_rate * (1.0 - max_ov / 0.02)
            if current_growth < 1e-7:
                current_growth = 1e-7 # Keep growing slowly to search
        
        radii += current_growth
        
        # 8. Perturbation to escape local minima
        if t % 2000 == 0 and t > 0:
            centers += np.random.normal(0, 0.005, centers.shape)
            
        # 9. Validation and tracking best solution
        if t % 100 == 0:
            valid = True
            # Check boundary constraints
            if np.any(centers[:, 0] + radii > 1.0 + 1e-9) or \
               np.any(centers[:, 0] - radii < -1e-9) or \
               np.any(centers[:, 1] + radii > 1.0 + 1e-9) or \
               np.any(centers[:, 1] - radii < -1e-9):
                valid = False
            
            if valid:
                # Check overlap constraints
                # dist >= r1 + r2 - 1e-12 is required for validity
                # i.e., dist + 1e-12 >= r1 + r2
                # If dist + 1e-12 < r1 + r2, then invalid
                if np.any(dist[mask] + 1e-12 < sum_radii[mask]):
                    valid = False
            
            if valid:
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()

    return best_centers, best_radii, best_sum
