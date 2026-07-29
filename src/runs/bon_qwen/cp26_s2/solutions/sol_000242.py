# sol_000242 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 6efaf445) state=45c7d839 sum of radii=0.499627 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimizes the packing of 26 circles in a unit square to maximize the sum of radii.
    """
    n_circles = 26
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Helper to generate hexagonal lattice points
    def generate_hex_lattice(target_n, square_size=1.0, padding=0.02):
        """
        Generates points in a hexagonal grid pattern inside the square.
        Returns a subset of points if more are generated than needed, 
        or adds random points if fewer.
        """
        points = []
        # Hexagonal spacing logic: vertical dist = sqrt(3)/2 * horizontal_dist
        # We iterate possible spacings to fit points
        
        # Heuristic: Try to fit rows
        # Estimate spacing s. Area per point ~ s^2 * sqrt(3)/2
        # n * s^2 * sqrt(3)/2 ~ (1-2p)^2
        # s ~ sqrt( (1-2p)^2 * 2 / (n * sqrt(3)) )
        effective_size = square_size - 2 * padding
        s = np.sqrt((effective_size**2) * 2 / (target_n * np.sqrt(3)))
        
        # Generate points
        row_y = padding
        while row_y <= square_size - padding:
            col_x = padding
            # Determine row parity for offset
            row_idx = int(round((row_y - padding) / (s * np.sqrt(3)/2)))
            offset = (s / 2) if (row_idx % 2 == 1) else 0
            
            current_x = padding + offset
            while current_x <= square_size - padding:
                points.append([current_x, row_y])
                current_x += s
            row_y += s * np.sqrt(3) / 2
            
        points = np.array(points)
        
        # Select or pad to match target_n
        if len(points) >= target_n:
            # Pick the best subset? Or just first N?
            # For optimization, just taking first N is usually fine if grid is dense
            # To be more robust, maybe pick random subset?
            # But deterministic is better for reproducibility.
            # Let's just take the first N.
            return points[:target_n]
        else:
            # Pad with random points if lattice was too sparse (shouldn't happen with formula)
            extra_needed = target_n - len(points)
            extra_pts = np.random.rand(extra_needed, 2) * (square_size - 2*padding) + padding
            return np.vstack([points, extra_pts])

    def get_initial_radii(centers):
        """
        Calculates a valid initial radius for each circle based on minimum distances.
        """
        radii = np.ones(n_circles) * 0.01
        for i in range(n_circles):
            min_dist = float('inf')
            # Distance to boundaries
            x, y = centers[i]
            min_dist = min(x, 1-x, y, 1-y)
            
            # Distance to other centers
            for j in range(n_circles):
                if i == j: continue
                dist = np.sqrt(np.sum((centers[i] - centers[j])**2))
                if dist < min_dist:
                    min_dist = dist
                    
            # Radius must be small enough so 2*r <= min_dist (roughly)
            # But we share space with others. 
            # A safe radius is min_dist / (2 * sqrt(N_local))? 
            # Just take a fraction of min_dist.
            # If min_dist is dist to neighbor, r_i + r_j <= dist.
            # Assuming equal radii, 2r <= dist => r <= dist/2.
            radii[i] = min_dist / 2.5 # Slightly smaller to be safe and allow movement
            
        return radii

    def objective(x_flat):
        # x_flat contains [x1, y1, r1, x2, y2, r2, ...]
        # We want to maximize sum(r), so minimize -sum(r)
        radii = x_flat[2::3]
        return -np.sum(radii)

    def constraints(x_flat):
        # Returns a vector of constraint values. All must be >= 0.
        # Format: x_flat[3*i] = xi, x_flat[3*i+1] = yi, x_flat[3*i+2] = ri
        
        centers = x_flat[:2*n_circles].reshape(n_circles, 2)
        radii = x_flat[2*n_circles:]
        
        cons_list = []
        
        # 1. Boundary constraints: center +/- radius in [0, 1]
        # x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        
        cons_list.extend(centers[:, 0] - radii)
        cons_list.extend(1 - centers[:, 0] - radii)
        cons_list.extend(centers[:, 1] - radii)
        cons_list.extend(1 - centers[:, 1] - radii)
        
        # 2. Overlap constraints: ||ci - cj||^2 >= (ri + rj)^2
        # (xi - xj)^2 + (yi - yj)^2 - (ri + rj)^2 >= 0
        
        # Vectorized calculation for overlaps is tricky with broadcasting to avoid N^2 memory spike if N large,
        # but N=26 is small.
        # Create matrices of coordinates
        X = centers[:, 0]
        Y = centers[:, 1]
        R = radii
        
        # Diff matrices
        # DiffX[i, j] = Xi - Xj
        DiffX = X[:, np.newaxis] - X[np.newaxis, :]
        DiffY = Y[:, np.newaxis] - Y[np.newaxis, :]
        DiffR_sum = R[:, np.newaxis] + R[np.newaxis, :]
        
        # Squared distances
        dist_sq = DiffX**2 + DiffY**2
        rad_sum_sq = DiffR_sum**2
        
        # Constraint: dist_sq - rad_sum_sq >= 0
        # We only need upper triangle (i < j) to avoid duplicates and zeros on diagonal
        mask = np.triu(np.ones((n_circles, n_circles), dtype=bool), k=1)
        cons_overlaps = (dist_sq - rad_sum_sq)[mask]
        
        cons_list.extend(cons_overlaps)
        
        return np.array(cons_list)

    def bounds_definition():
        # Bounds for [x1, y1, r1, ...]
        # x, y in [0, 1]
        # r in [0, 1] (loose upper bound, constrained by logic)
        b = []
        for _ in range(n_circles):
            b.append((0, 1)) # x
            b.append((0, 1)) # y
            b.append((0, 1)) # r
        return b

    bounds = bounds_definition()

    # Run multiple optimizations to escape local minima
    num_attempts = 10
    
    for attempt in range(num_attempts):
        # 1. Generate initial centers
        if attempt == 0:
            # Try hexagonal lattice first
            initial_centers = generate_hex_lattice(n_circles)
            # Sort to have deterministic order
            initial_centers = initial_centers[initial_centers[:, 0].argsort()]
        else:
            # Random perturbation or new random
            initial_centers = np.random.rand(n_circles, 2) * 0.8 + 0.1 # Keep away from edges initially
            
        # 2. Calculate valid initial radii
        initial_radii = get_initial_radii(initial_centers)
        
        # 3. Flatten variables
        x0 = np.concatenate([initial_centers.flatten(), initial_radii])
        
        # 4. Optimize
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints={'type': 'ineq', 'fun': constraints}, 
                           options={'ftol': 1e-9, 'maxiter': 500, 'disp': False})
            
            if res.success:
                current_sum = np.sum(res.x[2*n_circles:])
                # Validate result manually to be sure (though solver should respect constraints)
                # But solver constraints are inequalities >= 0, so it should be valid.
                # However, numerical noise might exist.
                
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = res.x[:2*n_circles].reshape(n_circles, 2)
                    best_radii = res.x[2*n_circles:]
        except Exception as e:
            print(f"Optimization attempt {attempt} failed: {e}")
            continue

    # Fallback if optimization failed completely
    if best_centers is None:
        # Generate a simple valid grid packing
        centers = np.zeros((n_circles, 2))
        radii = np.ones(n_circles) * 0.05
        idx = 0
        for r in range(5):
            for c in range(5):
                if idx < n_circles:
                    centers[idx] = [0.2 + c*0.2, 0.2 + r*0.2]
                    idx += 1
        if idx < n_circles:
             centers[idx] = [0.5, 0.5]
             idx += 1
        best_centers = centers[:n_circles]
        best_radii = radii[:n_circles]
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii

# Run the packing function to get the solution
centers, radii, total_sum = run_packing()
