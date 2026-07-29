# sol_000189 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 083f9270) state=3429aee2 sum of radii=2.081911 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses a hexagonal lattice initialization followed by a local optimization.
    """
    n = 26
    sqrt3 = np.sqrt(3)
    
    # 1. Initialization: Hexagonal Lattice
    # We arrange circles in rows. 
    # A staggered arrangement (hexagonal) is denser.
    # Layout: 6 rows with counts 5, 4, 5, 4, 5, 3 (Total 26)
    
    layout_x = []
    layout_y = []
    
    # Row 0: 5 circles (unoffset)
    for i in range(5):
        layout_x.append(1 + 2*i) 
        layout_y.append(1)
    # Row 1: 4 circles (offset)
    for i in range(4):
        layout_x.append(2 + 2*i)
        layout_y.append(1 + sqrt3)
    # Row 2: 5 circles (unoffset)
    for i in range(5):
        layout_x.append(1 + 2*i)
        layout_y.append(1 + 2*sqrt3)
    # Row 3: 4 circles (offset)
    for i in range(4):
        layout_x.append(2 + 2*i)
        layout_y.append(1 + 3*sqrt3)
    # Row 4: 5 circles (unoffset)
    for i in range(5):
        layout_x.append(1 + 2*i)
        layout_y.append(1 + 4*sqrt3)
    # Row 5: 3 circles (offset)
    for i in range(3):
        layout_x.append(2 + 2*i)
        layout_y.append(1 + 5*sqrt3)
        
    # Initial radius estimate based on bounding box of this layout
    # Width ~ 10r, Height ~ 10.66r. Limiting factor is height.
    r_initial = 0.0938
    
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i, 0] = layout_x[i] * r_initial
        centers[i, 1] = layout_y[i] * r_initial
        
    r = r_initial
    
    # 2. Optimization Loop
    # Iteratively increase radius and relax positions to resolve overlaps.
    
    step_r = 0.0002
    max_outer_iters = 600
    
    for _ in range(max_outer_iters):
        r_new = r + step_r
        
        # Relax centers to fit r_new
        temp_centers = centers.copy()
        success = False
        
        # Run relaxation steps
        for step in range(50):
            # Compute pairwise distances
            diff = temp_centers[:, np.newaxis, :] - temp_centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            dist = np.sqrt(dist_sq)
            
            # Overlap detection
            req_dist = 2 * r_new
            overlaps = req_dist - dist
            overlaps[overlaps < 0] = 0
            np.fill_diagonal(overlaps, 0)
            
            # Compute repulsive forces
            safe_dist = np.where(dist > 1e-9, dist, 1e-9)
            norm_diff = diff / safe_dist[:, :, np.newaxis]
            forces = np.sum(overlaps[:, :, np.newaxis] * norm_diff * 50.0, axis=1)
            
            # Apply forces
            temp_centers += forces * 1e-4
            
            # Wall constraints (push inside if touching/past wall)
            pen_left = r_new - temp_centers[:, 0]
            pen_left[pen_left < 0] = 0
            pen_right = temp_centers[:, 0] - (1 - r_new)
            pen_right[pen_right < 0] = 0
            pen_bottom = r_new - temp_centers[:, 1]
            pen_bottom[pen_bottom < 0] = 0
            pen_top = temp_centers[:, 1] - (1 - r_new)
            pen_top[pen_top < 0] = 0
            
            temp_centers[:, 0] += (pen_left - pen_right) * 50.0 * 1e-4
            temp_centers[:, 1] += (pen_bottom - pen_top) * 50.0 * 1e-4
            
            # Hard clamp
            temp_centers[:, 0] = np.clip(temp_centers[:, 0], r_new, 1 - r_new)
            temp_centers[:, 1] = np.clip(temp_centers[:, 1], r_new, 1 - r_new)
            
            # Check validity periodically
            if step % 5 == 0:
                min_dist_sq = np.min(dist_sq[~np.eye(n, dtype=bool)])
                min_dist = np.sqrt(min_dist_sq)
                if min_dist >= 2 * r_new - 1e-7:
                    # Check walls
                    if (np.all(temp_centers[:, 0] >= r_new - 1e-7) and 
                        np.all(temp_centers[:, 0] <= 1 - r_new + 1e-7) and
                        np.all(temp_centers[:, 1] >= r_new - 1e-7) and 
                        np.all(temp_centers[:, 1] <= 1 - r_new + 1e-7)):
                        success = True
                        break
        
        if success:
            r = r_new
            centers = temp_centers
        else:
            # If stuck, perturb centers to escape local optimum
            centers += np.random.normal(0, 0.002, size=centers.shape)
            centers[:, 0] = np.clip(centers[:, 0], r, 1-r)
            centers[:, 1] = np.clip(centers[:, 1], r, 1-r)

    # 3. Final Relaxation to ensure strict validity
    for _ in range(200):
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist = np.sqrt(dist_sq)
        overlaps = 2 * r - dist
        overlaps[overlaps < 0] = 0
        np.fill_diagonal(overlaps, 0)
        safe_dist = np.where(dist > 1e-9, dist, 1e-9)
        norm_diff = diff / safe_dist[:, :, np.newaxis]
        forces = np.sum(overlaps[:, :, np.newaxis] * norm_diff * 100.0, axis=1)
        centers += forces * 1e-4
        centers[:, 0] = np.clip(centers[:, 0], r, 1-r)
        centers[:, 1] = np.clip(centers[:, 1], r, 1-r)
        
    # 4. Adjust radius to be strictly valid based on final positions
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    min_dist_sq = np.min(dist_sq)
    min_dist = np.sqrt(min_dist_sq)
    
    # Radius limited by circle separation
    r_opt = min_dist / 2.0
    # Radius limited by walls
    r_wall = min(np.min(centers[:, 0]), 1 - np.max(centers[:, 0]), 
                 np.min(centers[:, 1]), 1 - np.max(centers[:, 1]))
    
    r_final = min(r_opt, r_wall)
    # Small safety margin
    r_final = max(0, r_final - 1e-9)
    
    radii = np.full(n, r_final)
    sum_radii = float(np.sum(radii))
    
    return centers, radii, sum_radii
