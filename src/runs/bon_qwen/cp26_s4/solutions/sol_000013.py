# sol_000013 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 04e92922) state=f232a016 sum of radii=1.097087 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Solves the circle packing problem for 26 circles in a unit square
    by maximizing the sum of radii using LP duality and gradient ascent.
    """
    n = 26
    centers = np.random.rand(n, 2)
    
    # Optimization parameters
    max_iter = 2000
    alpha_init = 0.1
    alpha_min = 1e-6
    decay = 0.995
    
    alpha = alpha_init

    # Precompute indices for constraints to avoid rebuilding lists
    # Pairwise constraints: r_i + r_j <= dist_ij
    pair_indices = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_pairs = len(pair_indices)
    
    # Boundary constraints indices mapping: 
    # 0: x - r >= 0  -> r <= x
    # 1: 1 - x - r >= 0 -> r <= 1 - x
    # 2: y - r >= 0  -> r <= y
    # 3: 1 - y - r >= 0 -> r <= 1 - y
    # Total boundary constraints per circle = 4
    # Total constraints = num_pairs + 4*n

    for iteration in range(max_iter):
        # 1. Compute distances and boundary distances
        # Distance matrix
        diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_matrix = np.sqrt(np.sum(diffs**2, axis=2))
        dist_matrix[np.eye(n, dtype=bool)] = np.inf # Self distance infinity
        
        # Boundary distances for each circle
        # b[i, 0] = x_i, b[i, 1] = 1-x_i, b[i, 2] = y_i, b[i, 3] = 1-y_i
        boundary_dists = np.zeros((n, 4))
        boundary_dists[:, 0] = centers[:, 0]
        boundary_dists[:, 1] = 1.0 - centers[:, 0]
        boundary_dists[:, 2] = centers[:, 1]
        boundary_dists[:, 3] = 1.0 - centers[:, 1]

        # 2. Setup LP: Minimize -sum(r_i)
        # Variables: r_0, ..., r_25
        c = -np.ones(n)
        
        # Constraints A_ub * r <= b_ub
        # We construct A_ub and b_ub
        # Rows 0 to num_pairs-1: Pairwise
        # Rows num_pairs to end: Boundary
        
        total_constraints = num_pairs + 4 * n
        A_ub = np.zeros((total_constraints, n))
        b_ub = np.zeros(total_constraints)
        
        # Fill Pairwise constraints
        # For each pair (i, j), row has 1 at i and 1 at j
        row_idx = 0
        for i, j in pair_indices:
            A_ub[row_idx, i] = 1.0
            A_ub[row_idx, j] = 1.0
            b_ub[row_idx] = dist_matrix[i, j]
            row_idx += 1
            
        # Fill Boundary constraints
        # For each circle i, 4 constraints
        for i in range(n):
            # r_i <= x_i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = boundary_dists[i, 0]
            row_idx += 1
            # r_i <= 1-x_i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = boundary_dists[i, 1]
            row_idx += 1
            # r_i <= y_i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = boundary_dists[i, 2]
            row_idx += 1
            # r_i <= 1-y_i
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = boundary_dists[i, 3]
            row_idx += 1

        # Bounds for r_i: [0, inf)
        bounds = [(0, None) for _ in range(n)]
        
        # Solve LP
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            
            if not res.success:
                # If LP fails, reduce step size and try again or break
                alpha *= 0.5
                continue

            radii = res.x
            duals = res.ineqlin.marginals
            
            # 3. Compute Gradient on Centers
            grad_centers = np.zeros_like(centers)
            
            # Duals corresponding to pairwise constraints
            # duals[0:num_pairs]
            k = 0
            for i, j in pair_indices:
                lam = duals[k]
                if lam > 1e-9: # Only consider active constraints
                    dist = dist_matrix[i, j]
                    if dist > 1e-12:
                        # Force direction: i pushes away from j, j pushes away from i
                        # Gradient of dist_ij w.r.t x_i is (x_i - x_j) / dist
                        force_vec = (centers[i] - centers[j]) / dist
                        grad_centers[i] += lam * force_vec
                        grad_centers[j] -= lam * force_vec
                k += 1
            
            # Duals corresponding to boundary constraints
            # duals[num_pairs : num_pairs + 4*n]
            # Order: for each i: 0:x, 1:1-x, 2:y, 3:1-y
            k = num_pairs
            for i in range(n):
                # Constraint r_i <= x_i (b = x_i). deriv b/deriv x_i = 1.
                # Gradient contribution: dual * 1
                grad_centers[i, 0] += duals[k] * 1.0
                k += 1
                
                # Constraint r_i <= 1-x_i (b = 1-x_i). deriv b/deriv x_i = -1.
                # Gradient contribution: dual * (-1)
                grad_centers[i, 0] += duals[k] * (-1.0)
                k += 1
                
                # Constraint r_i <= y_i (b = y_i). deriv b/deriv y_i = 1.
                grad_centers[i, 1] += duals[k] * 1.0
                k += 1
                
                # Constraint r_i <= 1-y_i (b = 1-y_i). deriv b/deriv y_i = -1.
                grad_centers[i, 1] += duals[k] * (-1.0)
                k += 1

            # 4. Update Centers
            centers += alpha * grad_centers
            
            # Project back to [0, 1]
            centers = np.clip(centers, 0.0, 1.0)
            
            # Decay step size
            alpha = max(alpha_min, alpha * decay)
            
        except Exception as e:
            alpha *= 0.5
            continue

    # Final cleanup and validation
    # Re-run LP one last time with final centers to get consistent radii
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diffs**2, axis=2))
    dist_matrix[np.eye(n, dtype=bool)] = np.inf
    
    boundary_dists = np.zeros((n, 4))
    boundary_dists[:, 0] = centers[:, 0]
    boundary_dists[:, 1] = 1.0 - centers[:, 0]
    boundary_dists[:, 2] = centers[:, 1]
    boundary_dists[:, 3] = 1.0 - centers[:, 1]
    
    total_constraints = num_pairs + 4 * n
    A_ub = np.zeros((total_constraints, n))
    b_ub = np.zeros(total_constraints)
    
    row_idx = 0
    for i, j in pair_indices:
        A_ub[row_idx, i] = 1.0
        A_ub[row_idx, j] = 1.0
        b_ub[row_idx] = dist_matrix[i, j]
        row_idx += 1
        
    for i in range(n):
        for dim in range(4):
            A_ub[row_idx, i] = 1.0
            b_ub[row_idx] = boundary_dists[i, dim]
            row_idx += 1
            
    bounds = [(0, None) for _ in range(n)]
    res = linprog(-np.ones(n), A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    final_radii = res.x
    sum_radii = np.sum(final_radii)
    
    return centers, final_radii, sum_radii
