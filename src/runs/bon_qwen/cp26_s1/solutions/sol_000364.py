# sol_000364 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state b037cf31) state=32acfccf sum of radii=0.520000 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses L-BFGS-B optimization with a penalty method for constraints.
    """
    n = 26
    
    # We use a penalty method. The objective is to maximize sum(radii), 
    # so we minimize -sum(radii) + penalty.
    
    def compute_cost_and_validity(params):
        # params structure: [x0, x1, ..., xn-1, y0, ..., yn-1, r0, ..., rn-1]
        # Length = 3 * n
        
        xs = params[:n]
        ys = params[n:2*n]
        rs = params[2*n:3*n]
        
        centers = np.column_stack((xs, ys))
        radii = rs
        
        # Calculate sum of radii (we want to maximize this, so objective is negative)
        sum_radii = np.sum(radii)
        
        # Check validity for the return check
        is_valid = True
        tolerance = 1e-9
        
        # 1. Check non-negative radii
        if np.any(radii < -tolerance):
            is_valid = False
            
        # 2. Check boundary constraints
        # r <= x <= 1-r  =>  x - r >= 0  AND  x + r <= 1
        if np.any(xs - radii < -tolerance): is_valid = False
        if np.any(xs + radii > 1 + tolerance): is_valid = False
        if np.any(ys - radii < -tolerance): is_valid = False
        if np.any(ys + radii > 1 + tolerance): is_valid = False
        
        # 3. Check overlaps
        # dist >= r_i + r_j
        # Vectorized distance computation
        # diff shape (n, n, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Set diagonal to infinity to ignore self-distance
        np.fill_diagonal(dists, np.inf)
        
        r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        # Check if any distance is less than sum of radii (minus tolerance)
        # We want dist >= r_sum. Violation if dist < r_sum - tol
        overlaps = dists < (r_sum - tolerance)
        if np.any(overlaps):
            is_valid = False
            
        # Calculate penalty for optimization
        # We use squared penalties for smoothness (C1)
        # Penalty weight needs to be high enough to enforce constraints
        
        penalty_weight = 5000.0
        
        # Boundary penalties
        # max(0, violation)^2
        viol_x_min = np.maximum(0, radii - xs) # x < r
        viol_x_max = np.maximum(0, xs + radii - 1) # x > 1-r
        viol_y_min = np.maximum(0, radii - ys) # y < r
        viol_y_max = np.maximum(0, ys + radii - 1) # y > 1-r
        
        boundary_penalty = (np.sum(viol_x_min**2) + np.sum(viol_x_max**2) +
                            np.sum(viol_y_min**2) + np.sum(viol_y_max**2))
        
        # Overlap penalties
        # max(0, r_i + r_j - dist)^2
        # We only need to sum over unique pairs or all pairs (factor of 2 doesn't matter for gradient direction)
        # To save compute, we can just sum all entries of the matrix, but diagonal is 0.
        overlap_violations = np.maximum(0, r_sum - dists)
        overlap_penalty = np.sum(overlap_violations**2)
        
        total_penalty = penalty_weight * (boundary_penalty + overlap_penalty)
        
        # Objective to minimize
        objective = -sum_radii + total_penalty
        
        return objective, is_valid, centers, radii, sum_radii

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (radius cannot be > 0.5 in unit square)
    bounds = [(0.0, 1.0)] * n + [(0.0, 1.0)] * n + [(0.0, 0.5)] * n
    
    best_sum_radii = -1.0
    best_centers = None
    best_radii = None
    
    # Run multiple restarts to escape local minima
    n_restarts = 15
    rng = np.random.RandomState(42)
    
    for _ in range(n_restarts):
        # Initialization
        # Start with small radii to ensure feasibility, then grow
        # Spread centers randomly but avoiding immediate boundaries
        xs_init = rng.uniform(0.15, 0.85, n)
        ys_init = rng.uniform(0.15, 0.85, n)
        rs_init = rng.uniform(0.02, 0.05, n)
        
        params0 = np.concatenate([xs_init, ys_init, rs_init])
        
        try:
            # Run optimization
            res = minimize(
                compute_cost_and_validity, # We need to adapt this to return just scalar for minimize
                # Actually minimize expects a function returning scalar. 
                # Let's wrap it.
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-8
                }
            )
            
            # The minimize function passes params to our function. 
            # But our function returns tuple. We need a wrapper.
            # Redefine objective function for minimize
            
            def obj_for_scipy(p):
                cost, _, _, _, _ = compute_cost_and_validity(p)
                return cost

            res = minimize(
                obj_for_scipy,
                params0,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': 2000,
                    'ftol': 1e-10,
                    'gtol': 1e-8
                }
            )
            
            # Retrieve result
            final_params = res.x
            _, is_valid, centers, radii, sum_radii = compute_cost_and_validity(final_params)
            
            if is_valid:
                if sum_radii > best_sum_radii:
                    best_sum_radii = sum_radii
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                    
        except Exception:
            continue

    # Fallback if no valid solution found (should not happen with good weight)
    if best_centers is None:
        # Simple grid fallback
        best_centers = np.zeros((n, 2))
        best_radii = np.zeros(n)
        idx = 0
        for i in range(5):
            for j in range(6):
                if idx < n:
                    best_centers[idx] = [0.15 + j * 0.14, 0.15 + i * 0.18]
                    best_radii[idx] = 0.02
                    idx += 1
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum_radii)
