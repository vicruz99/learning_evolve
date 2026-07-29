# sol_000158 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2296af5d) state=4526bae6 sum of radii=0.260000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import math
from scipy.optimize import minimize

def run_packing() -> tuple:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        tuple: (centers, radii, sum_radii)
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n_circles = 26
    
    # Helper function to calculate penalty for a given state
    # State vector: [x0, y0, r0, x1, y1, r1, ...]
    # Actually, separating variables might be cleaner, but let's pack them.
    # Let's keep centers and radii separate for logic, but optimizer needs 1D array.
    # We will use bounds for x, y in [0, 1] and r in [0, 0.5].
    
    # We will define the objective function that takes a flattened array
    # [x_0, y_0, r_0, x_1, y_1, r_1, ..., x_25, y_25, r_25]
    
    def objective(vars):
        # vars shape: (n_circles * 3,)
        centers = vars[:2 * n_circles].reshape(n_circles, 2)
        radii = vars[2 * n_circles:]
        
        # Objective: maximize sum of radii -> minimize -sum
        score = -np.sum(radii)
        
        # Penalty coefficient. Needs to be large enough to enforce constraints.
        # Since radii are around 0.1, penalty should be strong.
        # Let's use a large constant.
        penalty = 0.0
        penalty_weight = 1000.0
        
        # Check boundaries
        # x - r >= 0  => r - x <= 0
        # x + r <= 1  => x + r - 1 <= 0
        # Same for y
        
        # Boundary penalties
        # If x < r, penalty for (r - x)
        # If x > 1-r, penalty for (x + r - 1)
        
        # We can vectorize this
        # x coordinates
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        
        # Left boundary violation: r - x > 0
        left_violation = np.maximum(0, r - x)
        # Right boundary violation: x + r - 1 > 0
        right_violation = np.maximum(0, x + r - 1)
        # Bottom
        bottom_violation = np.maximum(0, r - y)
        # Top
        top_violation = np.maximum(0, y + r - 1)
        
        boundary_penalty = np.sum(left_violation**2 + right_violation**2 + 
                                  bottom_violation**2 + top_violation**2)
        
        penalty += penalty_weight * boundary_penalty
        
        # Overlap penalties
        # For all pairs i < j
        # dist^2 = (xi-xj)^2 + (yi-yj)^2
        # condition: dist >= ri + rj  => dist - (ri+rj) >= 0
        # violation: ri + rj - dist
        
        # Efficient calculation of distances
        # centers shape (N, 2)
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
        # dists = np.sqrt(np.sum(diff**2, axis=2))
        # But this creates N^2 matrix. 26^2 is small.
        
        # Calculate all pairwise distances
        # Using broadcasting
        c1 = centers[:, np.newaxis, :] # (N, 1, 2)
        c2 = centers[np.newaxis, :, :] # (1, N, 2)
        diffs = c1 - c2 # (N, N, 2)
        dists = np.sqrt(np.sum(diffs**2, axis=2)) # (N, N)
        
        # Radii sums
        r1 = r[:, np.newaxis]
        r2 = r[np.newaxis, :]
        r_sums = r1 + r2 # (N, N)
        
        # Violation
        # We only care about i < j, but symmetry handles it if we sum all and divide by 2?
        # Actually, violation is max(0, r_sum - dist)
        violations = np.maximum(0, r_sums - dists)
        
        # Sum of squares of violations. Diagonal is 0 (dist=0, r_sum=2r, violation 2r? No, dist 0).
        # Diagonal elements: dist=0, r_sum=2r. violation = 2r. This is bad.
        # We must ignore diagonal or set dist on diagonal to infinity?
        # Or just mask diagonal.
        np.fill_diagonal(dists, np.inf) # Effectively
        # Recompute violations or just ignore diagonal in sum
        # Let's recompute properly avoiding self
        violations = np.maximum(0, r_sums - dists)
        np.fill_diagonal(violations, 0)
        
        overlap_penalty = np.sum(violations**2)
        
        penalty += penalty_weight * overlap_penalty
        
        return score + penalty

    # Initialization
    # Try to place circles in a grid-like pattern
    # 26 circles. Maybe 6 rows, some 5, some 4?
    # Or just random valid positions.
    
    # Let's try a dense grid initialization
    # 5x5 grid gives 25 spots. 26th one?
    # Let's create a 6x5 grid (30 spots) and pick 26?
    # Or just space them out.
    
    # A good heuristic for N circles is to place them on a grid of size ceil(sqrt(N))
    grid_size = 6 # 6x6 grid has 36 points, we pick 26
    # Or 5x6
    points = []
    # Let's use a hexagonal lattice pattern if possible, but grid is easier
    # Let's try to fit 26 points in [0.1, 0.9] roughly
    # Step size approx 0.2
    # Rows
    for r_idx in range(6):
        for c_idx in range(5): # 30 points
            if len(points) < 26:
                x = 0.1 + c_idx * 0.2
                y = 0.1 + r_idx * 0.2
                # Adjust to center better?
                # Center of square is 0.5
                # Maybe shift
                x = x 
                y = y
                points.append([x, y])
    
    # If we don't have enough, fill with random
    while len(points) < 26:
        points.append([np.random.rand(), np.random.rand()])
        
    points = np.array(points[:26])
    
    # Initial radii small
    radii_init = np.full(26, 0.01)
    
    # Construct initial vector
    x_init = points[:, 0]
    y_init = points[:, 1]
    
    x0 = np.concatenate([x_init, y_init, radii_init])
    
    # Bounds
    # x in [0, 1], y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n_circles):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
    
    # Optimization
    # We might need multiple restarts or a robust method.
    # SLSQP is also an option but L-BFGS-B is fast for large penalty.
    # However, penalty method can be sensitive.
    # Let's try minimizing with L-BFGS-B
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                   options={'maxiter': 2000, 'ftol': 1e-9})
    
    # Extract result
    best_vars = res.x
    centers = best_vars[:2 * n_circles].reshape(n_circles, 2)
    radii = best_vars[2 * n_circles:]
    
    # Ensure radii are non-negative (optimizer should handle bounds)
    radii = np.maximum(radii, 0)
    
    sum_radii = np.sum(radii)
    
    # Validation check (internal)
    # The problem asks to return valid packing.
    # If the optimizer got stuck with overlaps, we might need to shrink radii.
    # But with high penalty, it should be valid.
    # Let's do a quick fix: if any overlap, reduce radii slightly.
    # Or just trust the penalty.
    # Actually, let's implement a simple shrinking loop if invalid.
    
    valid = False
    # Simple validation loop
    for _ in range(10):
        # Check overlaps
        dists = np.linalg.norm(centers[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        # Check diagonal
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        max_r_sum = np.max(r_sums) # Not quite right, need pairwise
        
        # Check pairwise
        overlap_found = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                d = np.sqrt((centers[i,0]-centers[j,0])**2 + (centers[i,1]-centers[j,1])**2)
                if d < radii[i] + radii[j] - 1e-7:
                    overlap_found = True
                    break
            if overlap_found: break
            
            # Check boundary
            if (centers[i,0] < radii[i] - 1e-7 or centers[i,0] > 1 - radii[i] + 1e-7 or
                centers[i,1] < radii[i] - 1e-7 or centers[i,1] > 1 - radii[i] + 1e-7):
                overlap_found = True # Treat as violation
                break
        
        if not overlap_found:
            valid = True
            break
        else:
            # Reduce radii slightly
            radii *= 0.99
            
    return centers, radii, float(sum_radii)

# Note: The function above might be slow if run repeatedly or if optimization takes long.
# For the purpose of the task, we rely on the optimizer finding a good local max.
# Given the constraints and the nature of L-BFGS-B, it should find a decent packing.
# To improve, one could run multiple random restarts.

def run_packing_optimized() -> tuple:
    """
    A more robust version with multiple restarts.
    """
    n_circles = 26
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    def objective(vars, penalty_weight=1000.0):
        centers = vars[:2 * n_circles].reshape(n_circles, 2)
        radii = vars[2 * n_circles:]
        score = -np.sum(radii)
        
        # Penalties
        penalty = 0.0
        
        # Boundaries
        x = centers[:, 0]
        y = centers[:, 1]
        r = radii
        
        # Left/Right
        left_v = np.maximum(0, r - x)
        right_v = np.maximum(0, x + r - 1)
        # Bottom/Top
        bottom_v = np.maximum(0, r - y)
        top_v = np.maximum(0, y + r - 1)
        
        penalty += penalty_weight * np.sum(left_v**2 + right_v**2 + bottom_v**2 + top_v**2)
        
        # Overlaps
        # Vectorized pairwise distance
        # dist_sq = sum((c_i - c_j)^2)
        # Use broadcasting
        c_diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(c_diff**2, axis=2))
        
        r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Mask diagonal
        dists[np.arange(n_circles), np.arange(n_circles)] = np.inf
        
        violations = np.maximum(0, r_sums - dists)
        penalty += penalty_weight * np.sum(violations**2)
        
        return score + penalty

    bounds_list = []
    for _ in range(n_circles):
        bounds_list.extend([(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)])

    # Try a few random initializations
    # Also try a grid initialization
    
    # 1. Grid Init
    points = []
    # Hexagonal packing approximation
    # Rows
    row_height = 0.15 # Approx
    # Let's just place them in a 5x5 grid plus some
    # 5x5 grid points
    grid_pts = []
    for i in range(5):
        for j in range(5):
            grid_pts.append([0.1 + i*0.2, 0.1 + j*0.2])
    # Add 1 more point in the center or somewhere
    grid_pts.append([0.5, 0.5]) # Overlaps, but optimizer will fix
    # We need 26 points. Grid has 25 + 1 = 26.
    # Actually 0.1 to 0.9 step 0.2 is 0.1, 0.3, 0.5, 0.7, 0.9 (5 points).
    # So 5x5 = 25.
    # The 26th point at (0.5, 0.5) overlaps with center circle.
    
    init_configs = []
    
    # Config 1: Grid + center
    x_init = np.array([p[0] for p in grid_pts])
    y_init = np.array([p[1] for p in grid_pts])
    r_init = np.full(26, 0.05)
    x0_1 = np.concatenate([x_init, y_init, r_init])
    init_configs.append(x0_1)
    
    # Config 2: Random valid
    for _ in range(3):
        centers_r = np.random.rand(26, 2) * 0.8 + 0.1 # Keep inside
        # Check for overlaps, if overlap move?
        # Just random
        r_init_r = np.full(26, 0.01)
        x0_r = np.concatenate([centers_r.flatten(), r_init_r])
        init_configs.append(x0_r)
        
    # Config 3: Hexagonal lattice
    # 6 rows, 5 cols alternating
    pts_hex = []
    for r_idx in range(6):
        offset = 0.1 if r_idx % 2 == 1 else 0.0
        # Actually shift by 0.1 (radius)
        # Let's try to pack tightly
        # Row spacing ~ 0.17
        y = 0.1 + r_idx * 0.17
        if y > 0.9: continue
        # X positions
        # If even row: 0.1, 0.3, 0.5, 0.7, 0.9 (5 circles)
        # If odd row: 0.2, 0.4, 0.6, 0.8 (4 circles)
        # Total 5+4+5+4+5+4 = 27?
        # We need 26.
        x_start = 0.1 if r_idx % 2 == 0 else 0.2
        count = 5 if r_idx % 2 == 0 else 4
        # But maybe we can fit 5 in odd row if shifted less?
        # Let's just generate points
        for c in range(5):
            x = x_start + c * 0.2
            if x <= 0.9: # Check bound
                if len(pts_hex) < 26:
                    pts_hex.append([x, y])
    
    if len(pts_hex) >= 26:
        pts_hex = pts_hex[:26]
    else:
        # Fill remainder
        while len(pts_hex) < 26:
            pts_hex.append([np.random.rand(), np.random.rand()])
            
    pts_hex = np.array(pts_hex)
    x0_hex = np.concatenate([pts_hex.flatten(), np.full(26, 0.05)])
    init_configs.append(x0_hex)

    best_res = None
    
    for i, x0 in enumerate(init_configs):
        try:
            res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds_list,
                           options={'maxiter': 1000, 'ftol': 1e-10})
            if res.fun < best_sum + 1e-6: # We minimize negative sum, so smaller is better?
                # Wait, objective is -sum + penalty.
                # So smaller is better.
                # But we want to track best valid packing.
                # Let's store the result and check validity later.
                pass
            
            # Extract
            centers = res.x[:2*n_circles].reshape(n_circles, 2)
            radii = res.x[2*n_circles:]
            # Clean radii
            radii = np.maximum(radii, 0)
            
            # Calculate valid sum
            # Check validity
            is_valid = True
            # Boundary
            for k in range(n_circles):
                if radii[k] < 0: is_valid = False; break
                if centers[k,0] < radii[k] or centers[k,0] > 1-radii[k] or \
                   centers[k,1] < radii[k] or centers[k,1] > 1-radii[k]:
                    is_valid = False; break
            
            if is_valid:
                # Check overlaps
                for k in range(n_circles):
                    for m in range(k+1, n_circles):
                        d = np.hypot(centers[k,0]-centers[m,0], centers[k,1]-centers[m,1])
                        if d < radii[k] + radii[m] - 1e-9:
                            is_valid = False
                            break
                    if not is_valid: break
            
            if is_valid:
                current_sum = np.sum(radii)
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                # If invalid but high score, maybe shrink radii to make valid?
                # But let's stick to valid ones found.
                # If we have an invalid one with very high -sum (high radii), we can try to shrink.
                # But for now, let's just keep best valid.
        
        except Exception:
            continue

    # If no valid packing found (unlikely with small radii init), fallback to small radii grid
    if best_centers is None:
        # Fallback: 5x5 grid with small radii
        pts = []
        for i in range(5):
            for j in range(5):
                pts.append([0.1 + i*0.2, 0.1 + j*0.2])
        pts.append([0.05, 0.5]) # 26th
        best_centers = np.array(pts[:26])
        best_radii = np.full(26, 0.01)
        best_sum = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum)

# We should export run_packing as the main function
run_packing = run_packing_optimized
