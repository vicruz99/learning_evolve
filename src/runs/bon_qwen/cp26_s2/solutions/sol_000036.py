# sol_000036 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 16d9c155) state=eda1d042 sum of radii=1.597778 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing():
    """
    Optimizes the packing of 26 circles in a unit square to maximize sum of radii.
    """
    n_circles = 26
    np.random.seed(42)
    
    best_sum_radii = 0
    best_centers = None
    best_radii = None
    
    # --- Phase 1: Optimize Centers for Equal Radii ---
    # We want to find centers that allow the largest possible equal radius r.
    # Objective: maximize r s.t. dist(c_i, boundary) >= r and dist(c_i, c_j) >= 2r
    
    n_restarts = 200
    max_iter = 150
    step_size = 0.005
    
    for _ in range(n_restarts):
        # Initialize centers randomly inside a smaller box to avoid immediate boundary violations
        # or in a grid pattern
        if np.random.random() < 0.5:
            # Random initialization with padding
            centers = np.random.rand(n_circles, 2) * 0.8 + 0.1
        else:
            # Grid-like initialization
            centers = np.zeros((n_circles, 2))
            for i in range(n_circles):
                r_idx = int(np.sqrt(n_circles))
                cx = (i % r_idx) * (0.8 / (r_idx - 1 if r_idx > 1 else 1)) + 0.1
                cy = (i // r_idx) * (0.8 / (r_idx - 1 if r_idx > 1 else 1)) + 0.1
                centers[i] = [cx, cy]
        
        # Initial valid radius for this configuration
        current_r = 0.05 
        
        # Hill climbing to expand radius
        for _ in range(max_iter):
            # Calculate max valid radius for current centers
            # Distance to boundary
            dist_boundary = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]), 
                                       np.minimum(centers[:, 1], 1 - centers[:, 1]))
            
            # Distance to other circles (half distance)
            # Vectorized distance computation
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dists, np.inf)
            min_dist_to_others = np.min(dists, axis=1) / 2.0
            
            # The limiting radius is the minimum of boundary dist and half inter-circle dist
            r_limit = np.minimum(dist_boundary, min_dist_to_others)
            current_r = np.min(r_limit)
            
            # If we can't increase r, we are stuck in a local optimum for equal radii
            # We want to perturb centers to allow larger r.
            # We identify "tight" constraints and move circles away.
            
            # Find circles that are constrained (limiting r)
            constrained = (r_limit - current_r < 1e-6)
            
            # Try to move constrained circles in random directions
            moved = False
            for i in range(n_circles):
                if constrained[i]:
                    # Check which constraint is active
                    active_boundary = (dist_boundary[i] - current_r < 1e-6)
                    # Find nearest neighbor
                    nearest_idx = np.argmin(dists[i])
                    active_neighbor = (dists[i, nearest_idx]/2.0 - current_r < 1e-6)
                    
                    # Generate a small random displacement
                    delta = np.random.randn(2) * step_size
                    new_pos = centers[i] + delta
                    
                    # Check if new position is better (allows larger radius for this circle)
                    # We only check local improvement to speed up
                    nb = np.minimum(np.minimum(new_pos[0], 1 - new_pos[0]), 
                                   np.minimum(new_pos[1], 1 - new_pos[1]))
                    # Dist to nearest neighbor
                    d_nn = np.sqrt(np.sum((new_pos - centers[nearest_idx])**2)) / 2.0
                    
                    # New local limit
                    new_r_limit = np.minimum(nb, d_nn)
                    
                    if new_r_limit > r_limit[i] - 1e-7:
                        centers[i] = new_pos
                        moved = True
            
            if not moved:
                break # Stuck, stop optimizing this restart
        
        # Calculate sum of radii for this configuration (assuming equal radii)
        current_sum = current_r * n_circles
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            
    # --- Phase 2: Optimize Radii for Fixed Centers (Linear Programming) ---
    # Maximize sum(r_i)
    # Subject to:
    # r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    # r_i + r_j <= dist(i, j)
    # r_i >= 0
    
    if best_centers is not None:
        c = best_centers
        
        # Calculate distances between all pairs
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Calculate distances to boundaries for each circle
        x = c[:, 0]
        y = c[:, 1]
        dist_x_min = x
        dist_x_max = 1 - x
        dist_y_min = y
        dist_y_max = 1 - y
        
        # LP Setup
        # Variables: r_0, r_1, ..., r_25 (26 variables)
        # Objective: Maximize sum(r_i) => Minimize -sum(r_i)
        c_obj = np.ones(n_circles) * -1
        
        # Constraints matrix A_ub @ x <= b_ub
        # 1. Boundary constraints: r_i <= dist_to_boundary
        #    r_i <= x_i  => r_i - x_i <= 0? No, x_i is constant.
        #    r_i <= x_i
        #    r_i <= 1-x_i
        #    r_i <= y_i
        #    r_i <= 1-y_i
        
        # We can stack these.
        # A_ub will be (4*n + n*(n-1)/2) x n
        
        # Actually, scipy linprog handles bounds better.
        # r_i >= 0 is handled by bounds.
        # r_i <= dist_boundary can be handled by upper bounds in bounds argument?
        # But dist_boundary depends on center, which is fixed now. So yes.
        
        bounds_r = []
        for i in range(n_circles):
            max_r = min(x[i], 1-x[i], y[i], 1-y[i])
            bounds_r.append((0, max_r))
            
        # Overlap constraints: r_i + r_j <= dist(i, j)
        # r_i + r_j - dist(i, j) <= 0
        # This is A_ub @ r <= b_ub
        
        constraints_matrix = []
        constraints_vector = []
        
        # Only consider pairs with distance < 2*max_possible_radius to save time?
        # But for correctness, we include all.
        # Optimization: if dist(i,j) is very large, constraint is loose.
        
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                d = dists[i, j]
                # Constraint: r_i + r_j <= d
                row = np.zeros(n_circles)
                row[i] = 1.0
                row[j] = 1.0
                constraints_matrix.append(row)
                constraints_vector.append(d)
                
        if constraints_matrix:
            A_ub = np.array(constraints_matrix)
            b_ub = np.array(constraints_vector)
            
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds_r, method='highs')
            
            if res.success:
                opt_radii = res.x
                final_sum = np.sum(opt_radii)
                
                # Verify validity just in case of numerical issues
                # (The validation function will be called externally, but good to check)
                valid = True
                for i in range(n_circles):
                    r = opt_radii[i]
                    cx, cy = c[i]
                    if cx - r < -1e-9 or cx + r > 1 + 1e-9 or cy - r < -1e-9 or cy + r > 1 + 1e-9:
                        valid = False
                        break
                if valid:
                    for i in range(n_circles):
                        for j in range(i+1, n_circles):
                            d = np.sqrt((c[i,0]-c[j,0])**2 + (c[i,1]-c[j,1])**2)
                            if d < opt_radii[i] + opt_radii[j] - 1e-9:
                                valid = False
                                break
                        if not valid: break
                
                if valid and final_sum > best_sum_radii:
                    best_sum_radii = final_sum
                    best_radii = opt_radii
                else:
                    # If LP failed or didn't improve, fallback to equal radii from Phase 1
                    best_radii = np.full(n_circles, best_sum_radii / n_circles)
            else:
                best_radii = np.full(n_circles, best_sum_radii / n_circles)
        else:
            best_radii = np.full(n_circles, best_sum_radii / n_circles)

    if best_radii is None:
        best_radii = np.full(n_circles, 0.05)

    return best_centers, best_radii, best_sum_radii
