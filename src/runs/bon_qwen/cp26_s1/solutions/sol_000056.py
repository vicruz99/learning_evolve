# sol_000056 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2252d37f) state=afd38cd3 sum of radii=2.342095 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize, linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # Phase 1: Find centers for dense equal-circle packing
    n_circles = 26
    
    # Initial placement on a 5x5 grid plus extra points
    # 5x5 grid gives 25 points. We add one at (0.5, 0.5) roughly, 
    # or just perturb to make 26 distinct points.
    # A 6x5 grid (30 points) is safer to start with.
    x_grid = np.linspace(0.08, 0.92, 6)
    y_grid = np.linspace(0.08, 0.92, 5)
    
    centers = []
    for x in x_grid:
        for y in y_grid:
            centers.append([x, y])
            if len(centers) == 26:
                break
        if len(centers) == 26:
            break
            
    centers = np.array(centers)
    
    # Function to calculate overlap penalty for equal radius rho
    def overlap_energy(centers_flat, rho):
        centers = centers_flat.reshape(-1, 2)
        energy = 0.0
        
        # Boundary penalty (soft constraint inside optimizer, but we use bounds too)
        # We rely on bounds for hard boundary, but this helps if bounds are loose
        # Actually, we pass bounds to minimize, so we just check overlaps.
        
        # Calculate all pairwise distances
        # Vectorized distance matrix
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # Triangular part (i < j)
        # Upper triangle indices
        triu_indices = np.triu_indices(n_circles, k=1)
        pair_dists = dists[triu_indices]
        
        # Penalty: max(0, 2*rho - dist)^2
        violation = 2 * rho - pair_dists
        penalty = np.maximum(violation, 0.0)**2
        energy = np.sum(penalty)
        
        return energy

    # Optimization loop to find max rho
    rho = 0.05
    step = 0.002
    max_rho = 0.12 # Heuristic upper bound
    rho_step_increase = 1.05
    rho_step_decrease = 0.5
    
    # Bounds for centers: [rho, 1-rho]
    # We will update bounds every iteration
    
    # Pre-calculate indices for speed
    triu_indices = np.triu_indices(n_circles, k=1)
    
    best_centers = centers.copy()
    
    # Run optimization
    # We try to increase rho. If optimization fails (residual > tol), we reduce step or rho
    # But since we start valid, we just increase rho and re-optimize.
    
    # Better approach: Binary search or incremental with warm start
    # Incremental with warm start
    
    current_rho = 0.05
    # Start with a small rho where grid is valid
    # Grid spacing is approx 0.2, so 2*rho <= 0.2 => rho <= 0.1
    # But we need to move them. Let's start rho=0.05.
    
    for _ in range(100): # Iterations
        # Define bounds for current rho
        low = current_rho
        high = 1.0 - current_rho
        bounds = [(low, high)] * (2 * n_circles)
        
        # Objective wrapper
        def obj(x):
            return overlap_energy(x, current_rho)
        
        # Optimize
        res = minimize(obj, centers.flatten(), method='L-BFGS-B', bounds=bounds, 
                       options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 1000})
        
        if res.success or res.fun < 1e-8:
            # Valid packing found for current_rho
            best_centers = res.x.reshape(-1, 2)
            current_rho += step
        else:
            # Failed to resolve overlaps
            # Reduce rho slightly and step size
            current_rho -= step * 0.5
            step *= 0.8
            
        if current_rho >= max_rho:
            break
            
    # Final rho found
    final_rho = current_rho - step # approximate, but we have best_centers
    
    # Actually, the last successful rho is what best_centers corresponds to.
    # But best_centers might have been updated in the last failed attempt? 
    # No, we only update best_centers on success.
    # However, the loop structure above updates best_centers inside success.
    # Let's just use best_centers.
    
    centers = best_centers
    
    # Phase 2: Optimize individual radii using Linear Programming
    # Variables: r_0, ..., r_25
    # Maximize sum(r_i)
    # Constraints:
    # 1. r_i >= 0
    # 2. r_i <= x_i
    # 3. r_i <= 1 - x_i
    # 4. r_i <= y_i
    # 5. r_i <= 1 - y_i
    # 6. r_i + r_j <= dist(i, j) for all i < j
    
    # Distances matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    # LP Setup
    # c: coefficients for objective (max sum => min -sum)
    c = -np.ones(n_circles)
    
    # A_ub, b_ub for inequality constraints A_ub @ r <= b_ub
    # Constraints are of form coeff_r <= bound
    # We need to construct matrix A_ub of shape (num_constraints, n_circles)
    
    constraints_A = []
    constraints_b = []
    
    # Boundary constraints: r_i <= min(x, 1-x, y, 1-y)
    # This can be split into 4 constraints per circle or just 1 tight one.
    # r_i <= x_i  => [0..1..0] @ r <= x_i
    # r_i <= 1-x_i
    # ...
    # And r_i >= 0 is handled by bounds
    
    for i in range(n_circles):
        x, y = centers[i]
        # r_i <= x
        row = np.zeros(n_circles)
        row[i] = 1.0
        constraints_A.append(row)
        constraints_b.append(x)
        
        # r_i <= 1-x
        constraints_A.append(row)
        constraints_b.append(1.0 - x)
        
        # r_i <= y
        constraints_A.append(row)
        constraints_b.append(y)
        
        # r_i <= 1-y
        constraints_A.append(row)
        constraints_b.append(1.0 - y)
        
    # Pairwise constraints: r_i + r_j <= dist_ij
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            row = np.zeros(n_circles)
            row[i] = 1.0
            row[j] = 1.0
            constraints_A.append(row)
            constraints_b.append(dists[i, j])
            
    A_ub = np.array(constraints_A)
    b_ub = np.array(constraints_b)
    
    # Bounds for r: [0, None]
    r_bounds = [(0, None)] * n_circles
    
    # Solve LP
    # linprog minimizes c^T x
    res_lp = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=r_bounds, method='highs')
    
    if res_lp.success:
        radii = res_lp.x
    else:
        # Fallback to equal radii if LP fails (should not happen with valid centers)
        # Estimate radius from min distance / 2
        min_dist = np.min(dists[triu_indices])
        radii = np.full(n_circles, min_dist / 2.0)
        # Clip by boundary
        for i in range(n_circles):
            r = min(centers[i, 0], 1-centers[i, 0], centers[i, 1], 1-centers[i, 1])
            radii[i] = min(radii[i], r)

    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii
