# sol_000267 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state fd8f28d8) state=a06f0869 sum of radii=2.541000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize

# Global constant for number of circles
N_CIRCLES = 26

def objective(params):
    """
    Objective function to minimize.
    We want to maximize sum of radii, so we minimize -sum(radii).
    We add a penalty for constraint violations (overlaps and boundary).
    
    Args:
        params: 1D numpy array of length 3*N_CIRCLES [x0, y0, r0, x1, y1, r1, ...]
    """
    n = N_CIRCLES
    # Reshape to (n, 3) matrix
    pts = params.reshape(n, 3)
    centers = pts[:, :2]
    radii = pts[:, 2]
    
    # Term to maximize (so we minimize negative)
    sum_radii = np.sum(radii)
    
    penalty = 0.0
    
    # --- Boundary Penalties ---
    # Constraints: r <= x <= 1-r  and  r <= y <= 1-r
    # Equivalent to: x >= r, 1-x >= r, y >= r, 1-y >= r
    
    # x >= r  => r - x <= 0. Violation if r - x > 0.
    val = radii - centers[:, 0]
    penalty += np.sum(np.maximum(0, val)**2)
    
    # 1-x >= r => 1 - x - r >= 0. Violation if 1 - x - r < 0.
    val = 1.0 - centers[:, 0] - radii
    penalty += np.sum(np.maximum(0, -val)**2)
    
    # y >= r
    val = radii - centers[:, 1]
    penalty += np.sum(np.maximum(0, val)**2)
    
    # 1-y >= r
    val = 1.0 - centers[:, 1] - radii
    penalty += np.sum(np.maximum(0, -val)**2)
    
    # --- Overlap Penalties ---
    # Constraint: distance(i, j) >= r_i + r_j
    # Violation if (r_i + r_j) - distance > 0
    
    # Compute pairwise distances
    # centers shape (n, 2)
    # diff shape (n, n, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    # Ensure non-negative due to numerical errors
    dist = np.sqrt(np.maximum(dist_sq, 0))
    
    # Sum of radii for all pairs
    r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Violation amount
    violation = r_sum - dist
    
    # We only care about unique pairs (i < j)
    # Create a mask for upper triangle
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    v = violation[mask]
    penalty += np.sum(np.maximum(0, v)**2)
    
    # Large weight for penalty to enforce constraints strictly
    weight = 100000.0
    return -sum_radii + weight * penalty

def run_packing():
    n = N_CIRCLES
    
    # Bounds for variables: x in [0,1], y in [0,1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    best_valid_sum = -1.0
    best_valid_params = None

    # List of initial parameter configurations to try
    candidates = []

    # Initialization 1: Hexagonal Lattice
    # A hexagonal packing is dense. We start with a valid small radius and let optimizer expand.
    r_init = 0.085
    points = []
    # Generate enough points in hexagonal grid
    for i in range(20):
        y = r_init + i * np.sqrt(3) * r_init
        if y > 1.0 - r_init: break
        # Offset x for alternating rows
        start_x = 2 * r_init if (i % 2 == 1) else r_init
        x = start_x
        while x <= 1.0 - r_init:
            points.append([x, y])
            x += 2 * r_init
    
    pts_arr = np.array(points[:n])
    if len(pts_arr) < n:
        needed = n - len(pts_arr)
        pts_arr = np.vstack([pts_arr, np.random.uniform(0.1, 0.9, (needed, 2))])
    
    # Initialize radii
    radii_hex = np.full(n, r_init)
    params_hex = np.hstack([pts_arr.ravel(), radii_hex])
    candidates.append(params_hex)

    # Initialization 2: Random positions
    centers_rand = np.random.uniform(0.1, 0.9, (n, 2))
    radii_rand = np.full(n, 0.05)
    candidates.append(np.hstack([centers_rand.ravel(), radii_rand]))

    # Initialization 3: Grid perturbation (5x5 + 1)
    centers_grid = []
    for i in range(5):
        for j in range(5):
            centers_grid.append([0.1 + i*0.2, 0.1 + j*0.2])
    centers_grid.append([0.5, 0.5])
    radii_grid = np.full(n, 0.05)
    candidates.append(np.hstack([np.array(centers_grid).ravel(), radii_grid]))

    # Run optimization for each candidate
    for p0 in candidates:
        try:
            # Use L-BFGS-B which supports bounds
            res = scipy.optimize.minimize(objective, p0, method='L-BFGS-B', 
                                          bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-12})
            params = res.x
        except Exception:
            continue
            
        # Decode parameters
        pts = params.reshape(n, 3)
        centers = pts[:, :2]
        radii = pts[:, 2]
        
        # Strict Validation Check
        valid = True
        
        # Check boundaries
        for i in range(n):
            x, y = centers[i]
            r = radii[i]
            # Tolerance 1e-6
            if x < r - 1e-6 or x > 1 - r + 1e-6 or y < r - 1e-6 or y > 1 - r + 1e-6:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(n):
                for j in range(i + 1, n):
                    d = np.linalg.norm(centers[i] - centers[j])
                    if d < radii[i] + radii[j] - 1e-6:
                        valid = False
                        break
                if not valid: break
        
        if valid:
            s = np.sum(radii)
            if s > best_valid_sum:
                best_valid_sum = s
                best_valid_params = params

    if best_valid_params is not None:
        pts = best_valid_params.reshape(n, 3)
        return pts[:, :2], pts[:, 2], best_valid_sum
    else:
        # Fallback to a known valid packing
        # 25 circles in 5x5 grid radius 0.1, and 1 small circle in a gap
        centers = []
        radii = []
        for i in range(5):
            for j in range(5):
                centers.append([0.1 + i*0.2, 0.1 + j*0.2])
                radii.append(0.1)
        # Place 26th circle in the center of the square formed by (0.1,0.1), (0.3,0.1), (0.1,0.3), (0.3,0.3)
        # Center at (0.2, 0.2). Distance to corners is sqrt(0.02) approx 0.1414.
        # Max radius = 0.1414 - 0.1 = 0.0414. Use 0.041 for safety.
        centers.append([0.2, 0.2])
        radii.append(0.041)
        
        return np.array(centers), np.array(radii), np.sum(radii)
