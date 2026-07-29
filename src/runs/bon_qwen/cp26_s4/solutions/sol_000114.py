# sol_000114 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e52471dd) state=1b63ec3c sum of radii=0.000002 correctness=1.0
# stdout(first 200): Restart 0: Found valid packing with sum=0.00000 Restart 4: Found valid packing with sum=0.00000 Restart 5: Found valid packing with sum=0.00000
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import random

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    Uses a penalty method with scipy.optimize.minimize.
    """
    n = 26
    best_centers = None
    best_radii = None
    best_sum = -1.0
    best_score = -1.0 # Higher is better (sum_radii)

    # Run multiple times with different seeds to avoid local minima
    n_restarts = 10
    
    for restart_idx in range(n_restarts):
        # --- Initialization ---
        # Use a grid-like initialization to spread circles out, then perturb
        # Or random. Random is good for diversity.
        
        # Strategy: Place circles on a hexagonal-ish grid or random
        # Let's try random positions in the center to avoid boundary issues initially
        centers = np.random.rand(n, 2) * 0.6 + 0.2 # Random in [0.2, 0.8]
        radii = np.ones(n) * 0.01 # Small initial radius
        
        # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]
            
        # Bounds
        # x, y in [0, 1], r in [0, 0.5] (r cannot be > 0.5)
        bounds = [(0, 1), (0, 1), (0, 0.5)] * n
        
        # Penalty weight
        mu = 100.0
        
        # --- Optimization Loop (Annealing mu) ---
        # We perform a few steps of optimization, increasing mu to enforce constraints strictly
        for step in range(15):
            
            # Define the objective function
            def objective(vars_arr):
                pts = vars_arr.reshape(-1, 3)
                xs = pts[:, 0]
                ys = pts[:, 1]
                rs = pts[:, 2]
                
                # Objective: Maximize sum of radii -> Minimize negative sum
                obj_val = -np.sum(rs)
                
                penalty = 0.0
                
                # 1. Boundary Penalties
                # Constraint: r <= x <= 1-r  =>  x >= r  AND  x <= 1-r
                # Violation 1: r - x > 0
                v_left = np.maximum(0, rs - xs)
                penalty += np.sum(v_left**2)
                
                # Violation 2: x - (1-r) > 0 => x + r - 1 > 0
                v_right = np.maximum(0, xs + rs - 1.0)
                penalty += np.sum(v_right**2)
                
                # Constraint: r <= y <= 1-r
                v_bottom = np.maximum(0, rs - ys)
                penalty += np.sum(v_bottom**2)
                
                v_top = np.maximum(0, ys + rs - 1.0)
                penalty += np.sum(v_top**2)
                
                # 2. Overlap Penalties
                # Constraint: dist(i,j) >= r_i + r_j
                # Violation: r_i + r_j - dist > 0
                
                # Compute pairwise distances efficiently
                # pts[:, :2] contains (x, y)
                # Using broadcasting for distance matrix
                # dist_matrix[i, j] = distance between i and j
                
                # To save memory/time, we can compute upper triangle, 
                # but for N=26, full matrix is fine.
                
                # Centers (N, 2)
                c2d = pts[:, :2]
                
                # Diff (N, N, 2)
                # diff = c2d[:, np.newaxis, :] - c2d[np.newaxis, :, :] 
                # This creates a large array. Let's use einsum or loop for safety?
                # Actually, 26x26 is small.
                
                # Optimized distance calculation
                # sum_sq = np.sum((c2d[:, None, :] - c2d[None, :, :])**2, axis=2)
                # dists = np.sqrt(sum_sq)
                
                # Avoiding large temporary arrays if possible, but 26^2 is tiny.
                diff = c2d[:, np.newaxis, :] - c2d[np.newaxis, :, :]
                dists = np.sqrt(np.sum(diff**2, axis=2))
                
                # Radius sum matrix (N, N)
                r_sum = rs[:, np.newaxis] + rs[np.newaxis, :]
                
                # Overlap amount
                # ovlp[i, j] = max(0, r_i + r_j - dist_ij)
                ovlp = np.maximum(0, r_sum - dists)
                
                # Sum of squared violations
                # We sum over all i, j. Since matrix is symmetric and diagonal is 0,
                # this counts each pair twice. That's fine for penalty.
                penalty += np.sum(ovlp**2)
                
                return obj_val + mu * penalty

            # Run minimizer
            try:
                res = minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                               options={'ftol': 1e-12, 'gtol': 1e-8, 'maxiter': 500})
                x0 = res.x
            except:
                pass # If optimization fails, continue with current x0

            # Increase penalty weight to force feasibility
            mu *= 2.0
            
            # Optional: Add small random perturbation to escape local minima if stuck
            # But usually increasing mu is enough.
            # If we want to be aggressive:
            if step == 10:
                # Perturb radii slightly to help
                pts = x0.reshape(-1, 3)
                pts[:, 2] += np.random.randn(n) * 0.001
                pts[:, 2] = np.clip(pts[:, 2], 0.001, 0.5)
                x0 = pts.flatten()

        # --- Evaluate Result ---
        pts = x0.reshape(-1, 3)
        centers_cand = pts[:, :2]
        radii_cand = pts[:, 2]
        
        # Calculate sum of radii
        sum_r = np.sum(radii_cand)
        
        # Check validity (approximate)
        # We trust the optimizer but let's check constraint violations
        is_valid = True
        
        # Check boundaries
        for i in range(n):
            x, y, r = centers_cand[i, 0], centers_cand[i, 1], radii_cand[i]
            if x < r - 1e-7 or x > 1 - r + 1e-7 or y < r - 1e-7 or y > 1 - r + 1e-7:
                is_valid = False
                break
        
        # Check overlaps
        if is_valid:
            for i in range(n):
                for j in range(i + 1, n):
                    dx = centers_cand[i, 0] - centers_cand[j, 0]
                    dy = centers_cand[i, 1] - centers_cand[j, 1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist < radii_cand[i] + radii_cand[j] - 1e-7:
                        is_valid = False
                        break
                if not is_valid:
                    break

        if is_valid and sum_r > best_sum:
            best_sum = sum_r
            best_centers = centers_cand.copy()
            best_radii = radii_cand.copy()
            best_score = sum_r
            print(f"Restart {restart_idx}: Found valid packing with sum={sum_r:.5f}")
        elif not is_valid:
             # Even if invalid according to strict check, maybe penalty was low?
             # But we need to return valid.
             pass

    # If no valid packing found (unlikely with good init), return something safe
    if best_centers is None:
        # Fallback: small circles in grid
        cols = 6
        rows = 5
        x = np.linspace(0.1, 0.9, cols)
        y = np.linspace(0.1, 0.9, rows)
        centers_fallback = np.array([(xi, yi) for xi in x for yi in y])[:26]
        radii_fallback = np.ones(26) * 0.05
        best_centers = centers_fallback
        best_radii = radii_fallback
        best_sum = np.sum(radii_fallback)

    return best_centers, best_radii, best_sum
