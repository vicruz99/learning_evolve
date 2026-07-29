# sol_000123 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 40ff4175) state=07459d96 sum of radii=2.584015 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    N = 26
    
    def calculate_score(centers, radii, penalty_weight):
        """
        Calculates the objective value to minimize.
        Objective: -sum(radii) + penalty for violations.
        """
        sum_radii = np.sum(radii)
        penalty = 0.0
        
        # Boundary penalties
        # x - r >= 0  => max(0, r - x)^2
        # 1 - x - r >= 0 => max(0, r - (1 - x))^2
        # y - r >= 0
        # 1 - y - r >= 0
        
        # Vectorized calculations
        # centers shape (N, 2), radii shape (N)
        
        # Boundary checks
        # Left: r <= x  => r - x <= 0. Violation max(0, r - x)
        # Right: r <= 1 - x => x + r <= 1. Violation max(0, x + r - 1)
        # Bottom: r <= y
        # Top: r <= 1 - y => y + r <= 1. Violation max(0, y + r - 1)
        
        violations_boundary = np.zeros(N)
        
        # Left
        diff = radii - centers[:, 0]
        violations_boundary += np.maximum(0, diff)**2
        
        # Right
        diff = centers[:, 0] + radii - 1.0
        violations_boundary += np.maximum(0, diff)**2
        
        # Bottom
        diff = radii - centers[:, 1]
        violations_boundary += np.maximum(0, diff)**2
        
        # Top
        diff = centers[:, 1] + radii - 1.0
        violations_boundary += np.maximum(0, diff)**2
        
        penalty += np.sum(violations_boundary)
        
        # Overlap penalties
        # dist(i, j) >= r_i + r_j
        # Violation: max(0, r_i + r_j - dist)
        
        # Compute pairwise distances
        # Using broadcasting: (N, 1, 2) - (1, N, 2)
        diff_centers = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (N, N, 2)
        dists = np.sqrt(np.sum(diff_centers**2, axis=2)) # (N, N)
        
        # Sum of radii matrix
        rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :] # (N, N)
        
        # Violations where r_i + r_j > dist
        # We only care about i < j to avoid double counting and self
        # Create a mask for upper triangle
        mask = np.triu(np.ones((N, N), dtype=bool), k=1)
        
        violation_dist = np.maximum(0, rad_sum - dists)
        # Zero out diagonal and lower triangle just in case
        violation_dist = np.where(mask, violation_dist, 0)
        
        penalty += np.sum(violation_dist**2)
        
        return -sum_radii + penalty_weight * penalty

    def objective_function(params):
        # params is flattened [x1, y1, r1, x2, y2, r2, ...]
        centers = params[0::3].reshape(-1, 2) # x, y
        radii = params[2::3]
        # Wait, params structure: [x1, x2... xN, y1...yN, r1...rN]? 
        # Or [x1, y1, r1, x2, y2, r2]? 
        # Let's use [x1, y1, r1, ...] for easier indexing if we want, 
        # but separating is often cleaner. 
        # Let's stick to [x_0, y_0, r_0, x_1, y_1, r_1, ...]
        # Actually, for vectorization, [x_coords, y_coords, r_coords] is best.
        
        # Reshape based on my chosen layout in the main block
        # I will pass params as [x_0, x_1, ..., x_N, y_0, ..., y_N, r_0, ..., r_N]
        
        cx = params[:N]
        cy = params[N:2*N]
        r = params[2*N:3*N]
        
        centers = np.column_stack((cx, cy))
        
        return calculate_score(centers, r, penalty_weight=100.0)

    # Bounds for variables
    # x, y in [0, 1], r in [0, 0.5]
    bounds_opt = [(0.0, 1.0)] * N + [(0.0, 1.0)] * N + [(0.0, 0.5)] * N

    best_score = float('inf')
    best_params = None

    # Multiple restarts to find global optimum
    num_restarts = 5
    
    for k in range(num_restarts):
        # Initialization
        # Try a grid-based initialization with jitter
        if k == 0:
            # 5x5 grid + 1
            pts_x = []
            pts_y = []
            # 5x5 grid
            for i in range(5):
                for j in range(5):
                    pts_x.append(0.1 + 0.2 * i)
                    pts_y.append(0.1 + 0.2 * j)
            # 26th point in a gap
            pts_x.append(0.2)
            pts_y.append(0.8) # arbitrary gap
            
            cx_init = np.array(pts_x)
            cy_init = np.array(pts_y)
        else:
            # Random jitter or random placement
            # Random points
            cx_init = np.random.uniform(0.05, 0.95, N)
            cy_init = np.random.uniform(0.05, 0.95, N)
        
        r_init = np.full(N, 0.05) # Small radius
        
        # Flatten
        x0 = np.concatenate([cx_init, cy_init, r_init])
        
        # Optimize
        # Use L-BFGS-B with bounds
        try:
            res = opt.minimize(objective_function, x0, method='L-BFGS-B', bounds=bounds_opt, 
                               options={'maxiter': 2000, 'ftol': 1e-12})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback to simple grid if optimization failed
        pts_x = []
        pts_y = []
        for i in range(5):
            for j in range(5):
                pts_x.append(0.1 + 0.2 * i)
                pts_y.append(0.1 + 0.2 * j)
        pts_x.append(0.2)
        pts_y.append(0.8)
        best_params = np.concatenate([np.array(pts_x), np.array(pts_y), np.full(26, 0.05)])

    # Extract results
    cx = best_params[:N]
    cy = best_params[N:2*N]
    r = best_params[2*N:3*N]
    
    centers = np.column_stack((cx, cy))
    
    # Validate and clamp if necessary (though optimizer should have done it)
    # Just to be safe against numerical errors
    r = np.clip(r, 0, None)
    
    # Verify validity locally
    valid = True
    # Check boundaries
    for i in range(N):
        if cx[i] - r[i] < -1e-9 or cx[i] + r[i] > 1 + 1e-9 or \
           cy[i] - r[i] < -1e-9 or cy[i] + r[i] > 1 + 1e-9:
            valid = False
            # Try to fix?
            # For now just flag
    
    # Check overlaps
    dists = np.sqrt(((centers[:, None] - centers[None, :])**2).sum(axis=2))
    rad_sums = r[:, None] + r[None, :]
    overlaps = (dists < rad_sums - 1e-9)
    np.fill_diagonal(overlaps, False)
    if overlaps.any():
        valid = False

    # If not valid, we might need to shrink radii. 
    # But with high penalty, it should be valid.
    # If invalid, we can run a correction pass.
    if not valid:
        # Simple correction: reduce radii to satisfy constraints
        # This is a rough fix
        max_iter = 10
        for _ in range(max_iter):
            changed = False
            # Fix boundaries
            for i in range(N):
                r[i] = min(r[i], cx[i], 1-cx[i], cy[i], 1-cy[i])
                if r[i] < 0: r[i] = 0
            
            # Fix overlaps
            # Iterate over pairs
            for i in range(N):
                for j in range(i+1, N):
                    d = np.sqrt((cx[i]-cx[j])**2 + (cy[i]-cy[j])**2)
                    if d < r[i] + r[j] - 1e-9:
                        # Reduce radii proportionally or equally
                        # Reduce both by half the overlap
                        overlap = r[i] + r[j] - d
                        reduction = overlap / 2 + 1e-9
                        r[i] -= reduction
                        r[j] -= reduction
                        changed = True
                        if r[i] < 0: r[i] = 0
                        if r[j] < 0: r[j] = 0
        
        # Re-check
        # If still invalid, it's a bad run, but we return best effort.

    sum_radii = np.sum(r)
    
    return centers, r, sum_radii
