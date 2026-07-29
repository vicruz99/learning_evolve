# sol_000299 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bdf692b1) state=b9b59a56 sum of radii=1.333311 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Returns (centers, radii, sum_radii).
    """
    n_circles = 26
    
    # 1. Initialization: Create a perturbed grid or hexagonal layout
    # A 5x5 grid gives 25 points. We need 26.
    # Let's try to distribute them somewhat evenly.
    # We can use a random perturbation of a grid or a specific pattern.
    # A hexagonal packing is denser.
    
    centers = np.zeros((n_circles, 2))
    
    # Simple initialization: 5 rows, distributing 26 points
    # Row counts: 6, 5, 6, 5, 4 -> 26? No.
    # Let's just place them on a grid and add jitter.
    # 26 points. Maybe 5 columns, 5 rows = 25. Plus 1 in middle?
    # Or 6 columns, 4 rows + 2?
    # Let's use a simple grid with jitter.
    
    # Grid coordinates
    # We want to cover [0, 1] x [0, 1]
    # Let's try 6 points in X, 5 in Y? 30 points. Remove 4?
    # Or just scatter.
    
    # Better: Use a hexagonal lattice pattern clipped to square
    # Side length of hex cell approx 1/sqrt(26 * sqrt(3)/2) ?
    # Area per point approx 1/26 ~ 0.038.
    # r_hex ~ sqrt(0.038 / (sqrt(3)/2)) ~ 0.21? No, this is for density.
    # Let's just use a grid with spacing 0.2
    points = []
    # 5x5 grid
    for i in range(5):
        for j in range(5):
            points.append([0.1 + i*0.2, 0.1 + j*0.2])
    # We have 25 points. Add one at (0.5, 0.5) or somewhere?
    # (0.5, 0.5) is already there (i=2, j=2).
    # Let's shift the grid or use a different pattern.
    # Let's try to place 26 points in a roughly optimal equal packing config?
    # Or just random start is often fine with good optimizer.
    # But let's try a "blue noise" like distribution or just random.
    
    np.random.seed(42) # For reproducibility
    centers = np.random.uniform(0.05, 0.95, size=(n_circles, 2))
    
    # Run optimization loop
    # We want to maximize sum(r_i).
    # We will iteratively update centers based on forces derived from LP duals.
    
    # LP formulation:
    # Maximize sum(r_i)
    # s.t.
    #   r_i + r_j <= dist(c_i, c_j)  for all i < j
    #   r_i <= x_i
    #   r_i <= 1 - x_i
    #   r_i <= y_i
    #   r_i <= 1 - y_i
    #   r_i >= 0
    
    # This is equivalent to:
    # Minimize -sum(r_i)
    # s.t.
    #   r_i + r_j <= dist_ij
    #   r_i <= limit_i  (where limit_i is dist to boundary)
    #   r_i >= 0
    
    # In linprog:
    # c = -1 (for minimization of negative sum)
    # A_ub * r <= b_ub
    # A_ub rows correspond to pairs (i, j). 1 at i, 1 at j.
    # b_ub values are distances.
    # bounds for r_i: (0, limit_i)
    
    # Optimization parameters
    learning_rate = 0.05
    max_iter = 500
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Pre-allocate matrix for constraints (dense, but N=26 is small)
    n_pairs = n_circles * (n_circles - 1) // 2
    A_ub = np.zeros((n_pairs, n_circles))
    pairs_list = []
    
    idx = 0
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            pairs_list.append((i, j))
            idx += 1
            
    c_obj = -np.ones(n_circles)
    
    for iteration in range(max_iter):
        # Compute distances and boundary limits
        dists = np.zeros(n_pairs)
        limits = np.zeros(n_circles)
        
        # Compute pairwise distances
        # Vectorized distance computation
        # centers shape (26, 2)
        # diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :] # (26, 26, 2)
        # dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        # But we only need upper triangle
        
        # Efficient distance calculation
        # Using broadcasting
        c1 = centers[:, np.newaxis, :] # (26, 1, 2)
        c2 = centers[np.newaxis, :, :] # (1, 26, 2)
        diff = c1 - c2
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2)) # (26, 26)
        
        # Extract distances for pairs
        for k, (i, j) in enumerate(pairs_list):
            dists[k] = dist_matrix[i, j]
            
        # Compute boundary limits
        x = centers[:, 0]
        y = centers[:, 1]
        limits = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Solve LP
        # Bounds for r: (0, limit)
        bounds = [(0, lim) for lim in limits]
        
        # If limits are negative (centers outside), LP might fail or be infeasible.
        # Ensure limits are at least 0 (though centers should be inside).
        limits = np.maximum(limits, 0)
        bounds = [(0, lim) for lim in limits]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=dists, bounds=bounds, method='highs')
            
            if res.success:
                radii = res.x
                current_sum = -res.fun # res.fun is min of -sum, so -res.fun is max sum
                
                # Check if this is better
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
                # Compute Gradient to update centers
                # Gradient of objective w.r.t b_ub (distances) is given by duals (marginals)
                # For minimization problem min c^T x s.t. Ax <= b, dual y >= 0.
                # Sensitivity: d(opt_val)/d(b) = -y ? Or y?
                # Let's check sign.
                # If we relax constraint (increase b), optimal value (min) should decrease (or stay same).
                # So derivative should be <= 0.
                # Duals y are usually >= 0.
                # So d(opt_val)/d(b) = -y.
                # But our objective is sum(r) = -opt_val.
                # So d(sum)/d(b) = -(-y) = y.
                # So gradient of sum w.r.t distance d_ij is dual variable y_k (for constraint k).
                
                # Extract duals
                # 'highs' method stores marginals in result.marginals?
                # Or result.ineqlin.marginals?
                # In recent scipy, result object has 'marginals' dict or 'ineqlin.marginals'
                
                # Accessing marginals safely
                marginals = None
                if hasattr(res, 'marginals') and isinstance(res.marginals, dict):
                    marginals = res.marginals.get('ineqlin', None)
                elif hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals'):
                    marginals = res.ineqlin.marginals
                else:
                    # Fallback if attribute not found or None
                    marginals = np.zeros(n_pairs)
                
                if marginals is not None:
                    duals = np.array(marginals)
                else:
                    duals = np.zeros(n_pairs)
                
                # Compute forces on centers
                # Force on i from j is dual_k * (c_i - c_j) / dist_ij
                # Sum over j
                forces = np.zeros((n_circles, 2))
                
                # We need to map duals back to pairs
                # duals[k] corresponds to pair pairs_list[k] = (i, j)
                for k, (i, j) in enumerate(pairs_list):
                    d = dists[k]
                    if d > 1e-12:
                        force_mag = duals[k]
                        # Unit vector from j to i
                        u = (centers[i] - centers[j]) / d
                        forces[i] += force_mag * u
                        forces[j] -= force_mag * u
                    else:
                        # If overlap (dist=0), strong repulsion
                        # Random push? Or just 0
                        pass
                
                # Also consider boundary forces?
                # If r_i is limited by boundary, moving center away from wall helps.
                # The LP bounds handle this, but we don't have explicit duals for bounds in simple access sometimes.
                # However, the gradient from distances is usually sufficient to spread circles.
                # To keep circles inside, we can just project.
                
                # Update centers
                # Step size might need adaptation.
                # A constant learning rate with decay?
                step_size = learning_rate / (1 + iteration * 0.01)
                
                centers += step_size * forces
                
                # Project centers back to [0, 1]
                # But wait, if we project, we might snap them to boundary.
                # Maybe clamp with a small margin?
                centers = np.clip(centers, 0.001, 0.999)
                
            else:
                # If LP fails, maybe centers are too close or bad config.
                # Perturb centers slightly
                centers += np.random.normal(0, 0.01, size=centers.shape)
                centers = np.clip(centers, 0.0, 1.0)
                
        except Exception:
            # Safety catch
            centers += np.random.normal(0, 0.05, size=centers.shape)
            centers = np.clip(centers, 0.0, 1.0)

    # Final check and cleanup
    # Ensure we use the best found configuration
    if best_centers is not None:
        centers = best_centers
        # Recompute radii for best centers to be sure
        # (In case the last step was worse)
        c1 = centers[:, np.newaxis, :]
        c2 = centers[np.newaxis, :, :]
        diff = c1 - c2
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        dists = np.zeros(n_pairs)
        limits = np.zeros(n_circles)
        for k, (i, j) in enumerate(pairs_list):
            dists[k] = dist_matrix[i, j]
            
        x = centers[:, 0]
        y = centers[:, 1]
        limits = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        limits = np.maximum(limits, 0)
        bounds = [(0, lim) for lim in limits]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=dists, bounds=bounds, method='highs')
            if res.success:
                radii = res.x
                sum_radii = -res.fun
            else:
                # Fallback
                radii = best_radii
                sum_radii = best_sum
        except:
            radii = best_radii
            sum_radii = best_sum
    else:
        # Should not happen if loop runs
        radii = np.zeros(n_circles)
        sum_radii = 0.0

    return centers, radii, sum_radii

# For validation/debugging locally (not required for solution but good practice)
if __name__ == "__main__":
    # Just to ensure it runs
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
