# sol_000108 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 82d73ba2) state=4f238205 sum of radii=0.003260 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def get_initial_configurations(N, num_configs=3):
    """
    Generate multiple initial configurations for N circles.
    Uses hexagonal grids and perturbations.
    """
    configs = []
    
    # Configuration 1: Hexagonal Grid
    # Estimate radius for equal packing to guide grid density
    # Approx area per circle ~ 1/N. Radius ~ sqrt(1/(N*pi)).
    # For N=26, r ~ 0.1. Spacing ~ 0.2.
    # Let's create a grid dense enough to hold 26 points.
    
    def hex_grid(start_r=0.05):
        centers = []
        radii = []
        # Try to fill the square with a hexagonal pattern
        # Row height sqrt(3)/2 * spacing
        # Spacing = 2 * start_r
        spacing = 2 * start_r
        row_height = spacing * math.sqrt(3) / 2
        
        y = start_r
        row_idx = 0
        count = 0
        
        while y <= 1 - start_r and count < N:
            # Offset for even/odd rows
            offset = spacing / 2.0 if row_idx % 2 == 1 else 0.0
            
            # Determine x range
            x_start = start_r - offset
            x_end = 1 - start_r
            
            # If offset pushes x_start < start_r, clamp? 
            # Actually, just iterate x
            x = start_r
            if row_idx % 2 == 1:
                x = start_r + spacing / 2.0
            
            while x <= 1 - start_r and count < N:
                centers.append([x, y])
                radii.append(start_r)
                count += 1
                x += spacing
            
            y += row_height
            row_idx += 1
        
        # If we didn't reach N, fill remaining randomly? 
        # With start_r=0.05, we should fit plenty.
        # If count < N, add random valid points
        while count < N:
            # Simple random placement avoiding immediate overlap is hard without check
            # Just place at empty spots or perturb
            # For initialization, validity isn't strictly required if penalty handles it,
            # but valid is better.
            # Let's just duplicate last point? No.
            # Let's just add points in a secondary pass.
            # But for N=26, start_r=0.05 fits ~100 circles.
            break 
            
        return np.array(centers[:N]), np.array(radii[:N])

    # Config 1: Grid
    c1, r1 = hex_grid(start_r=0.05)
    configs.append((c1, r1))
    
    # Config 2: Grid with slightly different radius/offset
    c2, r2 = hex_grid(start_r=0.04) # More space
    if len(c2) < N:
        # Fill missing with random valid positions
        current_centers = c2
        current_radii = r2
        count = N - len(current_centers)
        # Just pick random points in [0.1, 0.9]
        for _ in range(count):
            pt = np.random.uniform(0.1, 0.9, 2)
            current_centers = np.vstack([current_centers, pt])
            current_radii = np.append(current_radii, 0.04)
        c2 = current_centers[:N]
        r2 = current_radii[:N]
    configs.append((c2, r2))
    
    # Config 3: Random valid
    centers = np.zeros((N, 2))
    radii = np.zeros(N)
    # Place first at center
    centers[0] = [0.5, 0.5]
    radii[0] = 0.05
    for i in range(1, N):
        # Find a spot far from existing
        placed = False
        attempts = 0
        while not placed and attempts < 100:
            x, y = np.random.uniform(0.1, 0.9, 2)
            r = 0.05
            ok = True
            for j in range(i):
                dist = np.hypot(centers[j,0]-x, centers[j,1]-y)
                if dist < r + radii[j] + 1e-5: # strict separation
                    ok = False
                    break
            if ok and x-r >= 0 and x+r <= 1 and y-r >= 0 and y+r <= 1:
                centers[i] = [x, y]
                radii[i] = r
                placed = True
            attempts += 1
        if not placed:
            # Fallback: place at random, let optimizer fix
            centers[i] = np.random.uniform(0, 1, 2)
            radii[i] = 0.02
    configs.append((centers, radii))
    
    return configs

def objective_function(vars, N, centers_init=None, radii_init=None):
    """
    Objective function to minimize: -sum(radii) + penalty
    vars: [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = vars[:2*N].reshape(N, 2)
    radii = vars[2*N:]
    
    # Ensure radii are non-negative (though bounds handle this, penalty helps)
    # Actually bounds handle it.
    
    sum_radii = np.sum(radii)
    
    penalty = 0.0
    lambda_overlap = 1000.0
    lambda_boundary = 1000.0
    
    # Overlap penalties
    # Vectorized computation
    # Centers shape (N, 2)
    # Radii shape (N,)
    
    # Compute pairwise distances
    # Using broadcasting: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    dists = np.sqrt(dists_sq + 1e-12) # Avoid 0
    
    radii_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    
    # Overlap amount
    overlap = radii_sum - dists
    
    # Only positive overlaps count
    overlap[overlap < 0] = 0
    penalty += lambda_overlap * np.sum(overlap**2)
    
    # Boundary penalties
    # Left: x - r >= 0 -> violation if r - x > 0
    violation_left = radii - centers[:, 0]
    violation_left[violation_left < 0] = 0
    penalty += lambda_boundary * np.sum(violation_left**2)
    
    # Right: 1 - x - r >= 0 -> violation if x + r - 1 > 0
    violation_right = centers[:, 0] + radii - 1.0
    violation_right[violation_right < 0] = 0
    penalty += lambda_boundary * np.sum(violation_right**2)
    
    # Bottom: y - r >= 0
    violation_bottom = radii - centers[:, 1]
    violation_bottom[violation_bottom < 0] = 0
    penalty += lambda_boundary * np.sum(violation_bottom**2)
    
    # Top: 1 - y - r >= 0
    violation_top = centers[:, 1] + radii - 1.0
    violation_top[violation_top < 0] = 0
    penalty += lambda_boundary * np.sum(violation_top**2)
    
    return -sum_radii + penalty

def clean_solution(centers, radii, N):
    """
    Post-process solution to ensure strict validity.
    If small overlaps exist, reduce radii slightly.
    """
    # Check overlaps
    max_overlap = 0
    dists = np.sqrt(np.sum((centers[:, np.newaxis, :] - centers[np.newaxis, :, :])**2, axis=2))
    r_sums = radii[:, np.newaxis] + radii[np.newaxis, :]
    # dist < r1 + r2
    overlap = r_sums - dists
    # Ignore diagonal
    np.fill_diagonal(overlap, 0)
    
    # Check boundary
    x, y = centers[:, 0], centers[:, 1]
    r = radii
    b_viol = np.maximum(0, r - x) + np.maximum(0, x + r - 1) + \
             np.maximum(0, r - y) + np.maximum(0, y + r - 1)
    max_b_viol = np.max(b_viol)
    
    max_overlap_val = np.max(overlap)
    
    correction = max(max_overlap_val, max_b_viol) + 1e-7
    
    if correction > 0:
        radii = radii - correction
        radii = np.maximum(radii, 0)
        
    return centers, radii

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    configs = get_initial_configurations(N)
    
    best_centers = None
    best_radii = None
    best_sum = -1.0
    best_score = float('inf') # We minimize -sum + penalty
    
    bounds = []
    for i in range(N):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r (max radius 0.5)
        
    for init_c, init_r in configs:
        # Flatten
        x0 = np.concatenate([init_c.flatten(), init_r])
        
        # Optimize
        # L-BFGS-B is good for bounds
        res = minimize(objective_function, x0, args=(N,), method='L-BFGS-B', 
                       bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9, 'gtol': 1e-6})
        
        opt_vars = res.x
        opt_centers = opt_vars[:2*N].reshape(N, 2)
        opt_radii = opt_vars[2*N:]
        
        # Calculate score (negative of objective if penalty was 0, but objective includes penalty)
        # We want to check validity and sum
        # Let's clean it first to see if it's valid
        c_clean, r_clean = clean_solution(opt_centers, opt_radii, N)
        
        current_sum = np.sum(r_clean)
        
        # Check validity explicitly
        valid = True
        # Boundary
        for i in range(N):
            xc, yc = c_clean[i]
            rc = r_clean[i]
            if xc < rc or xc > 1-rc or yc < rc or yc > 1-rc:
                valid = False
                break
        
        if valid:
            # Check overlaps
            for i in range(N):
                for j in range(i+1, N):
                    dist = np.sqrt((c_clean[i,0]-c_clean[j,0])**2 + (c_clean[i,1]-c_clean[j,1])**2)
                    if dist < r_clean[i] + r_clean[j] - 1e-12:
                        valid = False
                        break
                if not valid: break
        
        # We prefer valid solutions. If invalid, the sum might be artificially high due to overlapping.
        # But clean_solution reduces radii to make it valid.
        # So current_sum is the sum of a valid packing.
        
        # However, clean_solution might have reduced radii significantly if the optimizer failed.
        # We should track the best valid sum.
        
        # Let's verify validity again on the cleaned version just to be sure
        # (clean_solution logic assumes simple reduction works, which is safe)
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = c_clean
            best_radii = r_clean
            best_score = res.fun
            
    # Fallback if best_centers is None (should not happen)
    if best_centers is None:
        best_centers = np.zeros((N, 2))
        best_radii = np.zeros(N)
        best_sum = 0.0
        
    return best_centers, best_radii, best_sum
