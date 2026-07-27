# sol_000099 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state bf51a1cd) state=0ded05bb sum of radii=1.750260 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog
from scipy.spatial.distance import pdist

def compute_objective(vars_flat, N, K):
    """Objective function for center optimization: maximize sum of radii with penalty for overlaps/boundaries."""
    centers = vars_flat[:2 * N].reshape(N, 2)
    radii = vars_flat[2 * N:]
    
    penalty = 0.0
    
    # Boundary violations: x-r >= 0, x+r <= 1, y-r >= 0, y+r <= 1
    # Equivalent to: r <= x, r <= 1-x, r <= y, r <= 1-y
    viol_x = np.maximum(0, radii - centers[:, 0]) + np.maximum(0, radii - (1 - centers[:, 0]))
    viol_y = np.maximum(0, radii - centers[:, 1]) + np.maximum(0, radii - (1 - centers[:, 1]))
    penalty += np.sum(viol_x**2 + viol_y**2)
    
    # Pairwise overlap violations: dist >= r_i + r_j
    dists = pdist(centers)
    i_idx, j_idx = np.triu_indices(N, k=1)
    sum_radii_pairs = radii[i_idx] + radii[j_idx]
    overlap = np.maximum(0, sum_radii_pairs - dists)
    penalty += np.sum(overlap**2)
    
    # We minimize negative sum of radii + penalty
    return -np.sum(radii) + K * penalty

def solve_radii_lp(centers, N):
    """Given fixed centers, solve LP to find optimal radii maximizing sum."""
    # Bounds from walls
    bounds_r = np.minimum(np.minimum(centers[:, 0], 1 - centers[:, 0]),
                          np.minimum(centers[:, 1], 1 - centers[:, 1]))
    bounds_r = np.maximum(bounds_r, 1e-9) # Ensure positive upper bound
    
    # Pairwise constraints: r_i + r_j <= d_ij
    dists = pdist(centers)
    i_idx, j_idx = np.triu_indices(N, k=1)
    
    num_constraints = len(dists)
    A_ub = np.zeros((num_constraints, N))
    A_ub[np.arange(num_constraints), i_idx] = 1.0
    A_ub[np.arange(num_constraints), j_idx] = 1.0
    b_ub = dists
    
    # LP: minimize -sum(r_i) s.t. A_ub @ r <= b_ub, 0 <= r <= bounds_r
    c_obj = -np.ones(N)
    bounds = [(0.0, b) for b in bounds_r]
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            return res.x
    except Exception:
        pass
    return None

def run_packing():
    N = 26
    K = 1.0e6  # Penalty weight
    best_sum = -1.0
    best_centers = None
    best_radii = None
    
    # Bounds for variables: x,y in [0,1], r in [0, 0.5]
    bounds = [(0.0, 1.0)] * (2 * N) + [(1e-5, 0.5)] * N
    
    # Run multiple trials with different initializations
    for seed in range(20):
        np.random.seed(seed)
        # Initialize centers in a perturbed grid/hex pattern for faster convergence
        init_centers = np.random.rand(N, 2) * 0.8 + 0.1
        init_radii = np.ones(N) * 0.08
        init_vars = np.concatenate([init_centers.flatten(), init_radii])
        
        res = minimize(
            compute_objective,
            init_vars,
            args=(N, K),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12}
        )
        
        if not res.success:
            continue
            
        current_centers = res.x[:2 * N].reshape(N, 2)
        
        # Solve LP for exact optimal radii given these centers
        opt_radii = solve_radii_lp(current_centers, N)
        if opt_radii is not None:
            # Clamp tiny radii to zero if needed, though LP handles it
            opt_radii = np.maximum(opt_radii, 0.0)
            current_sum = np.sum(opt_radii)
            
            # Final strict feasibility check & slight shrink if necessary due to float precision
            valid = True
            # Check boundaries
            for i in range(N):
                x, y = current_centers[i]
                r = opt_radii[i]
                if x - r < -1e-13 or x + r > 1 + 1e-13 or y - r < -1e-13 or y + r > 1 + 1e-13:
                    valid = False
                    break
            if valid:
                # Check pairwise
                dists = pdist(current_centers)
                i_idx, j_idx = np.triu_indices(N, k=1)
                min_gap = np.min(dists - (opt_radii[i_idx] + opt_radii[j_idx]))
                if min_gap < -1e-13:
                    valid = False
            
            if valid and current_sum > best_sum:
                best_sum = current_sum
                best_centers = current_centers.copy()
                best_radii = opt_radii.copy()
                
    # Fallback if nothing found (should not happen)
    if best_centers is None:
        best_centers = np.random.rand(N, 2)
        best_radii = np.ones(N) * 0.01
        
    return best_centers, best_radii, np.sum(best_radii)
