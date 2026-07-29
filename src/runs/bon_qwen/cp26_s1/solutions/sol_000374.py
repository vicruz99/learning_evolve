# sol_000374 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 469c683e) state=eff193ca sum of radii=1.442322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    Uses an iterative optimization strategy combining Linear Programming for radii
    and gradient ascent (based on dual variables) for centers.
    """
    N = 26
    np.random.seed(42) # For reproducibility

    # Initialize centers randomly in the middle of the square to avoid immediate boundary issues
    # Keeping them away from edges initially helps the optimizer find a good configuration
    centers = np.random.uniform(0.2, 0.8, (N, 2))
    
    # Precompute pair indices for efficiency
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append((i, j))
    num_pairs = len(pairs)
    
    # LP Structure:
    # Maximize sum(r_i)  =>  Minimize -sum(r_i)
    # Subject to:
    # 1. r_i + r_j <= dist(i, j)  for all pairs (i, j)
    # 2. r_i <= x_i
    # 3. r_i <= 1 - x_i
    # 4. r_i <= y_i
    # 5. r_i <= 1 - y_i
    # 6. r_i >= 0
    
    num_constraints = num_pairs + 4 * N
    A_ub = np.zeros((num_constraints, N))
    
    # Fill A_ub for pair constraints
    for idx, (i, j) in enumerate(pairs):
        A_ub[idx, i] = 1.0
        A_ub[idx, j] = 1.0
        
    # Fill A_ub for boundary constraints
    # r_i <= x_i
    for i in range(N):
        A_ub[num_pairs + i, i] = 1.0
    # r_i <= 1 - x_i
    for i in range(N):
        A_ub[num_pairs + N + i, i] = 1.0
    # r_i <= y_i
    for i in range(N):
        A_ub[num_pairs + 2*N + i, i] = 1.0
    # r_i <= 1 - y_i
    for i in range(N):
        A_ub[num_pairs + 3*N + i, i] = 1.0

    c_obj = -np.ones(N) # Objective: minimize -sum(r)
    bounds = [(0, None)] * N
    
    best_sum_r = 0.0
    best_centers = centers.copy()
    best_radii = np.zeros(N)
    
    # Optimization parameters
    lr = 0.005       # Learning rate for center updates
    noise_scale = 0.002 # Scale of random noise
    steps = 6000     # Number of optimization steps
    
    # Cache for distance computation
    # Using broadcasting for fast distance matrix calculation
    # centers shape: (N, 2)
    
    for step in range(steps):
        # 1. Compute pairwise distances
        # diff shape: (N, N, 2)
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # dists shape: (N, N)
        dists = np.sqrt(np.sum(diff**2, axis=2))
        
        # 2. Construct b_ub vector
        b_ub = np.zeros(num_constraints)
        
        # Pair distances
        for idx, (i, j) in enumerate(pairs):
            b_ub[idx] = dists[i, j]
            
        # Boundary bounds
        for i in range(N):
            b_ub[num_pairs + i] = centers[i, 0]             # x_i
            b_ub[num_pairs + N + i] = 1.0 - centers[i, 0]   # 1 - x_i
            b_ub[num_pairs + 2*N + i] = centers[i, 1]       # y_i
            b_ub[num_pairs + 3*N + i] = 1.0 - centers[i, 1] # 1 - y_i
            
        # 3. Solve LP
        try:
            # method='highs' is generally robust and fast
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options={'disp': False})
            
            if res.success:
                radii = res.x
                current_sum = np.sum(radii)
                
                # Update best solution
                if current_sum > best_sum_r:
                    best_sum_r = current_sum
                    best_centers = centers.copy()
                    best_radii = radii.copy()
                
                # 4. Compute forces from dual variables (shadow prices)
                # duals corresponds to constraints in b_ub order
                duals = res.ineqlin.marginals
                
                forces = np.zeros((N, 2))
                
                # Forces from pair constraints (repulsion)
                # Dual variable for r_i + r_j <= D_ij is the sensitivity to D_ij.
                # Increasing D_ij increases objective.
                # Gradient of D_ij wrt c_i is unit vector (c_i - c_j)/D_ij.
                for idx, (i, j) in enumerate(pairs):
                    dual = duals[idx]
                    if dual > 1e-7: # Only consider active constraints
                        dist = dists[i, j]
                        if dist > 1e-9:
                            direction = (centers[i] - centers[j]) / dist
                            # Force magnitude proportional to dual
                            forces[i] += dual * direction
                            forces[j] -= dual * direction
                
                # Forces from boundary constraints (attraction to center)
                # If r_i <= x_i is active (dual > 0), increasing x_i helps.
                # Gradient of x_i wrt c_i is (1, 0).
                for i in range(N):
                    # x constraint: r_i <= x_i
                    dual_x = duals[num_pairs + i]
                    if dual_x > 1e-7:
                        forces[i, 0] += dual_x
                        
                    # 1-x constraint: r_i <= 1 - x_i
                    # Increasing 1-x_i means decreasing x_i helps? 
                    # Wait, constraint is r_i <= 1 - x_i.
                    # RHS is 1 - x_i. If we decrease x_i, RHS increases.
                    # So we want to decrease x_i. Force should be negative x direction.
                    # Gradient of RHS (1-x) wrt x is -1.
                    dual_1x = duals[num_pairs + N + i]
                    if dual_1x > 1e-7:
                        forces[i, 0] -= dual_1x
                        
                    # y constraint: r_i <= y_i
                    dual_y = duals[num_pairs + 2*N + i]
                    if dual_y > 1e-7:
                        forces[i, 1] += dual_y
                        
                    # 1-y constraint: r_i <= 1 - y_i
                    dual_1y = duals[num_pairs + 3*N + i]
                    if dual_1y > 1e-7:
                        forces[i, 1] -= dual_1y

                # 5. Update centers
                # Adaptive learning rate or fixed? Fixed with noise is fine.
                # Add random noise to help escape local minima
                noise = np.random.normal(0, noise_scale, (N, 2))
                
                # Apply forces and noise
                centers += lr * forces + noise
                
                # Clip centers to stay within [0, 1] to prevent numerical instability
                # Although forces should keep them somewhat reasonable, clipping is safe.
                # Actually, centers must be strictly inside if radii > 0, but [0,1] is the domain.
                centers = np.clip(centers, 0.0, 1.0)
                
        except Exception:
            # If LP fails, centers might be degenerate. Perturb them.
            centers += np.random.normal(0, 0.05, (N, 2))
            centers = np.clip(centers, 0.0, 1.0)

    # Final validation and cleanup
    # The best_radii are optimal for best_centers, but due to floating point,
    # we might want to re-solve LP for best_centers to ensure consistency,
    # or just trust the stored radii.
    # Let's re-solve one last time for consistency.
    
    # Re-compute distances for best_centers
    diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    
    b_ub = np.zeros(num_constraints)
    for idx, (i, j) in enumerate(pairs):
        b_ub[idx] = dists[i, j]
    for i in range(N):
        b_ub[num_pairs + i] = best_centers[i, 0]
        b_ub[num_pairs + N + i] = 1.0 - best_centers[i, 0]
        b_ub[num_pairs + 2*N + i] = best_centers[i, 1]
        b_ub[num_pairs + 3*N + i] = 1.0 - best_centers[i, 1]
        
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            best_radii = res.x
            best_sum_r = np.sum(best_radii)
    except:
        pass

    return best_centers, best_radii, float(best_sum_r)
