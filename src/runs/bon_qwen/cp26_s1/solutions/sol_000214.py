# sol_000214 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0926bf18) state=6d10fca4 sum of radii=1.750141 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import linprog

def run_packing():
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Strategy: Optimize centers to maximize the sum of radii found by solving an LP
    for the radii constraints given those centers.
    """
    n_circles = 26
    
    def compute_radii_and_sum(centers):
        """
        Given centers, solves LP to find max sum of radii.
        Maximize sum(r_i) subject to:
          1. r_i + r_j <= distance(c_i, c_j) for all i < j
          2. 0 <= r_i <= distance(c_i, boundary)
        """
        n = centers.shape[0]
        x = centers[:, 0]
        y = centers[:, 1]
        
        # Distance to boundaries: min(x, 1-x, y, 1-y)
        dist_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        
        # Pairwise distances
        # Vectorized difference: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diff**2, axis=2))
        
        # LP Setup
        # Objective: maximize sum(r) -> minimize -sum(r)
        c_obj = -np.ones(n)
        
        # Bounds for r_i: [0, dist_boundary_i]
        # If dist_boundary is negative (center outside), bounds will be invalid, 
        # but we handle centers within [0,1] mostly.
        bounds = [(0.0, max(0.0, db)) for db in dist_boundary]
        
        # Inequality constraints: r_i + r_j <= dist_ij
        # A_ub @ r <= b_ub
        # We only need upper triangular part (i < j)
        rows, cols = np.triu_indices(n, k=1)
        m = len(rows)
        
        # Construct A_ub matrix
        A_ub = np.zeros((m, n))
        b_ub = dist_matrix[rows, cols]
        
        # Set coefficients 1 for pairs
        A_ub[np.arange(m), rows] = 1.0
        A_ub[np.arange(m), cols] = 1.0
        
        # Solve LP
        try:
            # 'highs' is the default in recent scipy, but we specify for clarity if available
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return res.x, np.sum(res.x)
        except Exception:
            pass
        
        # Fallback if LP fails (e.g. infeasible bounds)
        # Return small valid radii
        r_safe = np.full(n, 0.0001)
        return r_safe, 0.0

    def objective(centers_flat):
        """Objective function to minimize (negative sum of radii)."""
        centers = centers_flat.reshape(-1, 2)
        _, s = compute_radii_and_sum(centers)
        return -s

    best_sum = 0.0
    best_centers = None
    
    # Generate initial configurations
    starts = []
    rng = np.random.RandomState(42)
    
    # 1. Hexagonal Grid Initialization
    # Attempt to place circles in a dense hexagonal pattern
    grid_centers = []
    count = 0
    r_idx = 0
    # Heuristic steps
    y_step = 0.18
    x_step = 0.16
    
    while count < n_circles:
        c_idx = 0
        # Allow variable number of columns per row for hex packing
        cols_in_row = 6 if r_idx % 2 == 0 else 5 
        while count < n_circles and c_idx < cols_in_row:
            x = 0.05 + c_idx * x_step + (r_idx % 2) * (x_step / 2)
            y = 0.05 + r_idx * y_step
            if 0 <= x <= 1 and 0 <= y <= 1:
                grid_centers.append([x, y])
                count += 1
            c_idx += 1
        r_idx += 1
    
    if len(grid_centers) >= n_circles:
        h_centers = np.array(grid_centers[:n_circles])
    else:
        # Fallback to random if grid generation is weird
        h_centers = rng.uniform(0.1, 0.9, size=(n_circles, 2))
        
    # Add random jitter to escape symmetry
    h_centers += rng.uniform(-0.02, 0.02, size=h_centers.shape)
    starts.append(h_centers.flatten())
    
    # 2. Random Initializations
    for _ in range(5):
        r_centers = rng.uniform(0.05, 0.95, size=(n_circles, 2))
        starts.append(r_centers.flatten())
        
    # Bounds for centers: [0, 1]
    bounds = [(0.0, 1.0)] * (n_circles * 2)
    
    # Optimization Loop
    for x0 in starts:
        try:
            # Powell method is derivative-free and good for this type of problem
            res = minimize(objective, x0, method='Powell', bounds=bounds, 
                           options={'maxiter': 2000, 'ftol': 1e-12})
            
            # Check if we improved
            if -res.fun > best_sum:
                best_sum = -res.fun
                best_centers = res.x.reshape(n_circles, 2)
        except Exception:
            pass

    # If optimization failed completely, use the first start
    if best_centers is None:
        best_centers = starts[0].reshape(n_circles, 2)
        
    # Final computation of radii
    radii, final_sum = compute_radii_and_sum(best_centers)
    
    # Safety: Clip centers to [0, 1] just in case
    best_centers = np.clip(best_centers, 0.0, 1.0)
    
    # Recompute radii after clipping (centers might have shifted slightly)
    radii, final_sum = compute_radii_and_sum(best_centers)
    
    # Ensure radii are non-negative
    radii = np.maximum(radii, 0.0)
    
    return best_centers, radii, float(np.sum(radii))
