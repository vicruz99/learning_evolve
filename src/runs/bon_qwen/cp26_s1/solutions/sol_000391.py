# sol_000391 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 19a68663) state=25063632 sum of radii=2.620408 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization ---
    # We start with a hexagonal packing pattern to give the optimizer a good head start.
    # We place circles with a small initial radius to ensure validity.
    
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.05) # Start with small valid radii
    
    # Generate hexagonal grid points
    # Row spacing and column spacing depend on desired density.
    # We want to fit 26 points. A 5x5 grid is 25.
    # Let's create a dense grid and select/perturb points.
    
    # Heuristic for hex grid:
    # Spacing s. Vertical dist = s * sqrt(3) / 2.
    # Let's try to fit points in [0.1, 0.9] roughly.
    
    points = []
    # Row 0
    for i in range(6): # Try 6 points
        x = 0.1 + i * 0.15
        y = 0.1
        if x <= 0.9:
            points.append([x, y])
            
    # Row 1 (shifted)
    for i in range(5):
        x = 0.2 + i * 0.15
        y = 0.1 + 0.15 * np.sqrt(3) / 2
        if x <= 0.9:
            points.append([x, y])

    # Row 2
    for i in range(6):
        x = 0.1 + i * 0.15
        y = 0.1 + 0.15 * np.sqrt(3)
        if x <= 0.9 and y <= 0.9:
            points.append([x, y])

    # Row 3 (shifted)
    for i in range(5):
        x = 0.2 + i * 0.15
        y = 0.1 + 1.5 * 0.15 * np.sqrt(3)
        if x <= 0.9 and y <= 0.9:
            points.append([x, y])

    # Row 4
    for i in range(6):
        x = 0.1 + i * 0.15
        y = 0.1 + 2 * 0.15 * np.sqrt(3)
        if x <= 0.9 and y <= 0.9:
            points.append([x, y])
            
    # We might have more or less than 26 points.
    # Let's just fill the array with these points, and if we have fewer, 
    # add random points in the center, or if more, pick first 26.
    # Actually, let's just generate a grid that definitely has > 26 points 
    # and pick a subset, or just overwrite.
    
    # Better approach: Fill a grid of size roughly sqrt(26) x sqrt(26) with offset
    pts_list = []
    # Grid generation
    row_idx = 0
    while len(pts_list) < n:
        y_base = 0.08 + row_idx * 0.18 # spacing y
        if y_base > 0.9: break
        
        # Offset for hexagonal pattern
        x_offset = 0.0 if row_idx % 2 == 0 else 0.09
        
        col_idx = 0
        while True:
            x_val = 0.08 + col_idx * 0.18 + x_offset
            if x_val > 0.92: break
            pts_list.append([x_val, y_base])
            col_idx += 1
            if len(pts_list) >= n: break
        row_idx += 1

    # If we still don't have 26, add random ones (shouldn't happen with this density)
    while len(pts_list) < n:
        pts_list.append([np.random.rand(), np.random.rand()])
        
    centers = np.array(pts_list[:n])
    
    # --- 2. Optimization Setup ---
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (upper bound 0.5 is safe, radius can't be > 0.5 in unit square)

    # Objective: Minimize -sum(radii)
    def objective(vars_flat):
        radii_vec = vars_flat[2::3]
        return -np.sum(radii_vec)

    # Constraints:
    # We will define a function that returns all constraint values >= 0.
    # Constraints:
    # 1. Boundary: x - r >= 0, 1 - x - r >= 0, y - r >= 0, 1 - y - r >= 0
    # 2. Overlap: dist(i, j) - (r_i + r_j) >= 0
    
    def constraints(vars_flat):
        constraints_vec = []
        
        # Extract positions and radii
        centers = vars_flat.reshape(n, 3)[:, :2]
        r = vars_flat[2::3]
        
        # Boundary constraints (4 per circle)
        # x - r >= 0
        constraints_vec.append(centers[:, 0] - r)
        # 1 - x - r >= 0
        constraints_vec.append(1.0 - centers[:, 0] - r)
        # y - r >= 0
        constraints_vec.append(centers[:, 1] - r)
        # 1 - y - r >= 0
        constraints_vec.append(1.0 - centers[:, 1] - r)
        
        # Overlap constraints
        # Vectorized distance calculation might be heavy for N=26 in a single vector return?
        # N=26 -> 325 pairs. It's fine.
        
        # Compute pairwise distances
        # dist_matrix[i, j] = ||c_i - c_j||
        # We only need i < j
        
        # Efficient calculation
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (n, n, 2)
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # To save memory/time, iterate or use broadcasting carefully.
        # Given N=26, (26x26) array is tiny.
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # r_sum[i, j] = r_i + r_j
        r_sum = r[:, np.newaxis] + r[np.newaxis, :]
        
        # Constraint: dists[i, j] - r_sum[i, j] >= 0
        # We only need upper triangle
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        overlaps = dists[mask] - r_sum[mask]
        
        constraints_vec.append(overlaps)
        
        # Flatten the list of arrays
        return np.concatenate([arr.flatten() for arr in constraints_vec])

    cons = {'type': 'ineq', 'fun': constraints}

    # --- 3. Run Optimization ---
    
    # Use SLSQP
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 1000, 'ftol': 1e-12, 'disp': False})
    
    # --- 4. Post-processing ---
    
    best_vars = res.x
    best_centers = best_vars.reshape(n, 3)[:, :2]
    best_radii = best_vars[2::3]
    
    # Numerical safety: slightly reduce radii if they are extremely close to boundaries/overlaps
    # to ensure strict validation passes (though SLSQP should handle it).
    # We check for any potential violations and shrink radii slightly.
    
    # A robust way: Calculate max valid radii for the found centers.
    # But moving centers might be better. However, let's just ensure validity.
    
    # Check boundary constraints
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        # Max radius allowed by boundary
        r_max_bound = min(x, 1-x, y, 1-y)
        if r > r_max_bound:
            best_radii[i] = r_max_bound
            
    # Check overlap constraints
    # If overlaps exist, we must reduce radii.
    # We can iteratively reduce radii.
    for _ in range(10): # A few iterations to resolve overlaps
        overlap_found = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((best_centers[i] - best_centers[j]) ** 2))
                sum_r = best_radii[i] + best_radii[j]
                if dist < sum_r - 1e-9: # Strict overlap
                    # Reduce both radii equally? Or the one that hurts less?
                    # Simple approach: scale down both to fit.
                    needed_reduction = sum_r - dist + 1e-9
                    # Distribute reduction
                    best_radii[i] -= needed_reduction / 2
                    best_radii[j] -= needed_reduction / 2
                    overlap_found = True
                    # Ensure non-negative
                    if best_radii[i] < 0: best_radii[i] = 0
                    if best_radii[j] < 0: best_radii[j] = 0
        if not overlap_found:
            break
            
    # Re-check boundary after radius reduction
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        r_max_bound = min(x, 1-x, y, 1-y)
        if r > r_max_bound:
            best_radii[i] = r_max_bound

    sum_radii = np.sum(best_radii)
    
    # Final validation check (debugging)
    # validate_packing(best_centers, best_radii) 
    
    return best_centers, best_radii, sum_radii

# To comply with the prompt structure, we define run_packing. 
# The code above is inside the function scope logic, but needs to be self-contained.
# I will structure it properly below.
