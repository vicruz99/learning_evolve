# sol_000220 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state df9a626f) state=8492293c sum of radii=2.524400 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False

    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False

    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            if np.sqrt(dist_sq) < radii[i] + radii[j] - 1e-12:
                return False
    return True

def generate_initial_guess(n, seed=None):
    """Generates a feasible initial configuration for n circles."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create a dense grid (6x5 = 30 points)
    x_grid = np.linspace(0.1, 0.9, 6)
    y_grid = np.linspace(0.1, 0.9, 5)
    
    centers = []
    for x in x_grid:
        for y in y_grid:
            centers.append([x, y])
    centers = np.array(centers)
    
    # Select n random points from the grid
    indices = np.random.choice(len(centers), n, replace=False)
    selected_centers = centers[indices]
    
    # Initial radius is small to ensure no overlap
    r_init = 0.04
    radii = np.full(n, r_init)
    
    return selected_centers, radii

def objective_and_constraints(X, n):
    """
    Calculates objective and constraints for the optimizer.
    X: flattened array of [x1, y1, r1, ..., x26, y26, r26]
    """
    coords = X[:2*n] # x1..x26, y1..y26
    x = coords[0::2]
    y = coords[1::2]
    radii = X[2*n:]
    
    # Objective: Maximize sum of radii -> Minimize negative sum
    obj = -np.sum(radii)
    
    # Constraints: must be >= 0
    # 1. Boundary constraints (4 per circle)
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    wall_c1 = x - radii
    wall_c2 = 1.0 - x - radii
    wall_c3 = y - radii
    wall_c4 = 1.0 - y - radii
    
    # 2. Non-overlap constraints
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # Vectorized calculation
    x_mat = x.reshape(n, 1)
    y_mat = y.reshape(n, 1)
    r_mat = radii.reshape(n, 1)
    
    dist_sq = (x_mat - x_mat.T)**2 + (y_mat - y_mat.T)**2
    rad_sum_sq = (r_mat + r_mat.T)**2
    
    # We only need upper triangle (i < j)
    triu_idx = np.triu_indices(n, k=1)
    pair_c = dist_sq[triu_idx] - rad_sum_sq[triu_idx]
    
    # Combine all constraints
    all_constraints = np.concatenate([wall_c1, wall_c2, wall_c3, wall_c4, pair_c])
    
    return obj, all_constraints

def get_constraints_dict(n):
    """Returns a constraint dictionary for scipy.optimize.minimize."""
    def con_fun(X):
        _, cons = objective_and_constraints(X, n)
        return cons
    
    return {'type': 'ineq', 'fun': con_fun}

def run_packing():
    n = 26
    bounds = [(0.0, 1.0) for _ in range(2*n)] + [(0.0, 0.5) for _ in range(n)]
    
    best_sum_r = -1.0
    best_X = None
    
    # Multi-start optimization
    num_starts = 10
    for i in range(num_starts):
        centers, radii = generate_initial_guess(n, seed=i*42)
        
        # Flatten to optimization vector
        X0 = np.zeros(3 * n)
        X0[0::3] = centers[:, 0]
        X0[1::3] = centers[:, 1]
        X0[2::3] = radii
        
        # Re-interleave to [x1, y1, r1, x2, y2, r2, ...]
        X_flat = np.zeros(3 * n)
        X_flat[0::3] = centers[:, 0]
        X_flat[1::3] = centers[:, 1]
        X_flat[2::3] = radii
        
        # We need to adjust the variable order in the objective/constraints function
        # The previous function assumed [x1...xn, y1...yn, r1...rn]
        # Let's just use the [x1, y1, r1...] order for simplicity in the solver
        X_flat = np.zeros(3*n)
        X_flat[0::3] = centers[:, 0]
        X_flat[1::3] = centers[:, 1]
        X_flat[2::3] = radii

        # Redefine objective/constraints for [x1, y1, r1, ...] order
        def obj(X):
            return -np.sum(X[2::3])
        
        def con(X):
            x = X[0::3]
            y = X[1::3]
            r = X[2::3]
            
            wall = np.concatenate([x-r, 1-x-r, y-r, 1-y-r])
            
            x_m = x.reshape(n,1)
            y_m = y.reshape(n,1)
            r_m = r.reshape(n,1)
            
            dist = (x_m - x_m.T)**2 + (y_m - y_m.T)**2
            r_sum = (r_m + r_m.T)**2
            triu = np.triu_indices(n, k=1)
            pairs = dist[triu] - r_sum[triu]
            
            return np.concatenate([wall, pairs])

        constraints = {'type': 'ineq', 'fun': con}
        
        try:
            res = minimize(obj, X_flat, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 500, 'ftol': 1e-9})
            
            if res.success:
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_X = res.x.copy()
        except Exception:
            continue
            
    if best_X is None:
        # Fallback if optimization fails
        centers, radii = generate_initial_guess(n)
        best_sum_r = np.sum(radii)
        best_X = np.zeros(3*n)
        best_X[0::3] = centers[:,0]
        best_X[1::3] = centers[:,1]
        best_X[2::3] = radii

    # Extract final solution
    centers_out = np.column_stack((best_X[0::3], best_X[1::3]))
    radii_out = best_X[2::3]
    final_sum = -np.sum(-radii_out) # Should be sum(radii_out)
    
    # Final validation
    if not validate_packing(centers_out, radii_out):
        # If invalid, try to shrink slightly to fix boundary/overlap errors
        factor = 0.99
        radii_out *= factor
        centers_out *= 1.0 # Centers stay, just radii shrink
        
    return centers_out, radii_out, np.sum(radii_out)
