# sol_000080 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e5887d00) state=626d9970 sum of radii=2.604709 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
import math

def generate_initial_guess(n, seed=None):
    """
    Generates an initial valid configuration of n circles.
    Tries a hexagonal lattice first, then fills remaining with random valid placements.
    """
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    idx = 0
    
    # Try hexagonal lattice
    r_est = 0.15 # Slightly larger than expected to ensure tight packing later
    row = 0
    placed_lattice = 0
    
    # We might need to scan a bit or just fill
    # Let's try to place points in a grid pattern
    x_start = 0.1
    y_start = 0.1
    step = 0.15
    
    # Better: specific hex grid generation
    # y = r + k * sqrt(3)*r
    r_init = 0.1
    y_curr = r_init
    row_idx = 0
    
    while idx < n and y_curr + r_init <= 1.0:
        if row_idx % 2 == 0:
            x_curr = r_init
        else:
            x_curr = 2 * r_init
        
        while idx < n and x_curr + r_init <= 1.0:
            centers[idx] = [x_curr, y_curr]
            radii[idx] = r_init
            idx += 1
            x_curr += 2 * r_init
            placed_lattice += 1
        
        y_curr += np.sqrt(3) * r_init
        row_idx += 1

    # If not all placed, fill with random valid spots (small radius)
    if idx < n:
        # Simple strategy: place in grid of 10x10 empty spots?
        # Just place at center with tiny radius to start
        # Or spread them out
        # We will rely on optimizer to expand them
        # Place them in a loose grid
        grid_size = int(np.ceil(np.sqrt(n - idx)))
        spacing = 1.0 / (grid_size + 1)
        count = 0
        for r in range(grid_size):
            for c in range(grid_size):
                if count >= n - idx:
                    break
                centers[idx] = [spacing * (c + 1), spacing * (r + 1)]
                radii[idx] = 0.01 # Small initial radius
                idx += 1
                count += 1
            if count >= n - idx:
                break
                
    return centers, radii

def run_packing():
    n = 26
    
    # We will try multiple random seeds to find the best packing
    best_sum_radii = 0.0
    best_centers = None
    best_radii = None
    
    # Number of restarts
    restarts = 5
    
    for seed in range(restarts):
        # Generate initial guess
        np.random.seed(seed + 42)
        
        # For some restarts, randomize positions slightly
        centers, radii = generate_initial_guess(n)
        
        # Add small random perturbation to centers to break symmetry and explore
        if seed > 0:
            perturbation = np.random.uniform(-0.02, 0.02, (n, 2))
            centers = np.clip(centers + perturbation, 0.05, 0.95)
            # Ensure radii are small enough for perturbed centers
            # Recompute max possible radius for current centers to be safe
            for i in range(n):
                x, y = centers[i]
                # Distance to boundary
                dist_bound = min(x, 1-x, y, 1-y)
                # Distance to other centers
                min_dist_other = 1.0
                for j in range(n):
                    if i == j: continue
                    d = np.sqrt((centers[i]-centers[j])**2).sum() # Vectorized dist
                    # manual dist
                    dx = centers[i,0] - centers[j,0]
                    dy = centers[i,1] - centers[j,1]
                    d = math.sqrt(dx*dx + dy*dy)
                    if d < min_dist_other:
                        min_dist_other = d
                
                # Safe radius
                radii[i] = min(0.05, dist_bound * 0.5, min_dist_other * 0.45)
        
        # Flatten variables: x1, y1, r1, x2, y2, r2, ...
        x0 = np.zeros(3 * n)
        for i in range(n):
            x0[3*i] = centers[i, 0]
            x0[3*i+1] = centers[i, 1]
            x0[3*i+2] = radii[i]

        # Bounds: x,y in [0,1], r >= 0
        # Tighter bounds for r can help, but 0.5 is safe upper bound
        bounds = [(0, 1)] * n + [(0, 1)] * n + [(0, 0.5)] * n
        # Reshape bounds list
        bounds = []
        for _ in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])

        def objective(vars):
            # Maximize sum of radii -> Minimize negative sum
            return -np.sum(vars[2::3])

        def constraints(vars):
            cons = []
            
            # Extract variables
            X = vars[0::3]
            Y = vars[1::3]
            R = vars[2::3]
            
            # 1. Boundary constraints: r <= x <= 1-r  =>  x-r >= 0, 1-x-r >= 0
            # Similarly for y
            for i in range(n):
                cons.append(X[i] - R[i])
                cons.append(1.0 - X[i] - R[i])
                cons.append(Y[i] - R[i])
                cons.append(1.0 - Y[i] - R[i])
            
            # 2. Overlap constraints: dist(i,j) >= r_i + r_j
            # dist^2 >= (r_i + r_j)^2
            # (xi-xj)^2 + (yi-yj)^2 - (ri+rj)^2 >= 0
            
            # Vectorized computation for speed
            # Create matrices for broadcasting
            # X_col is shape (n, 1)
            X_col = X[:, None]
            Y_col = Y[:, None]
            R_col = R[:, None]
            
            # Difference matrices
            diff_x = X_col - X_col.T
            diff_y = Y_col - Y_col.T
            
            # Distance squared
            dist_sq = diff_x**2 + diff_y**2
            
            # Sum of radii squared
            sum_r = R_col + R_col.T
            sum_r_sq = sum_r**2
            
            # Constraint value
            # We only need upper triangle (i < j)
            # dist_sq[i,j] - sum_r_sq[i,j] >= 0
            # np.triu_indices with k=1
            i_idx, j_idx = np.triu_indices(n, k=1)
            
            vals = dist_sq[i_idx, j_idx] - sum_r_sq[i_idx, j_idx]
            cons.extend(vals)
            
            return np.array(cons)

        cons_dict = {'type': 'ineq', 'fun': constraints}
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons_dict,
                           options={'maxiter': 1000, 'ftol': 1e-10, 'disp': False})
            
            if res.success or (res.status > 0 and np.isfinite(res.fun)):
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    # Extract best vars
                    best_vars = res.x
                    best_centers = np.zeros((n, 2))
                    best_radii = np.zeros(n)
                    for i in range(n):
                        best_centers[i, 0] = best_vars[3*i]
                        best_centers[i, 1] = best_vars[3*i+1]
                        best_radii[i] = best_vars[3*i+2]
                        
        except Exception:
            continue
            
    # Fallback if optimization failed completely
    if best_centers is None:
        centers, radii = generate_initial_guess(n)
        best_centers = centers
        best_radii = radii
        best_sum_radii = np.sum(radii)

    # Final cleanup: ensure non-negative radii (clipping small negatives if any due to precision)
    best_radii = np.maximum(best_radii, 0.0)
    
    return best_centers, best_radii, float(np.sum(best_radii))
