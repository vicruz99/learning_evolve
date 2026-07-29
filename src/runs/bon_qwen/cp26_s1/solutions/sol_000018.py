# sol_000018 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f223c9a2) state=40fae800 sum of radii=1.852873 correctness=1.0
# stdout(first 200): Iter 0, Sum Radii: 1.85287, LR: 0.04900 Iter 100, Sum Radii: 1.85287, LR: 0.00650 Iter 200, Sum Radii: 1.85287, LR: 0.00086 Iter 300, Sum Radii: 1.85287, LR: 0.00011 Iter 400, Sum Radii: 1.85287, LR: 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def compute_forces_from_marginals(centers, marginals, n):
    """
    Computes forces on centers based on LP marginals.
    Marginals correspond to constraints:
    0 to n*(n-1)/2 - 1 : Pairwise r_i + r_j <= dist(i,j)
    Remaining : Boundary constraints r_i <= bounds
    """
    forces = np.zeros_like(centers)
    
    # Pairwise constraints
    pair_idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            lam = marginals[pair_idx]
            # If marginal is non-zero, the constraint is active.
            # We assume positive marginal implies increasing RHS (distance) improves objective.
            # Force pushes i away from j.
            if lam > 1e-9:
                vec = centers[i] - centers[j]
                dist = np.linalg.norm(vec)
                if dist > 1e-9:
                    direction = vec / dist
                    forces[i] += direction * lam
                    forces[j] -= direction * lam
            pair_idx += 1
            
    # Boundary constraints
    # For each circle i, 4 constraints: x, 1-x, y, 1-y
    # Order in marginals: x_i, 1-x_i, y_i, 1-y_i for each i?
    # Let's check construction order in solve_lp.
    # In solve_lp:
    # for i in range(n):
    #   r_i <= x_i
    #   r_i <= 1-x_i
    #   r_i <= y_i
    #   r_i <= 1-y_i
    # So offset for circle i in boundary section is 4*i.
    
    bound_offset = n * (n - 1) // 2
    for i in range(n):
        # r_i <= x_i (RHS is centers[i, 0])
        # Increasing x_i helps. Force +x.
        lam_x_pos = marginals[bound_offset + 4 * i]
        if lam_x_pos > 1e-9:
            forces[i, 0] += lam_x_pos
            
        # r_i <= 1 - x_i (RHS is 1 - centers[i, 0])
        # Increasing 1-x_i (decreasing x_i) helps. Force -x.
        lam_x_neg = marginals[bound_offset + 4 * i + 1]
        if lam_x_neg > 1e-9:
            forces[i, 0] -= lam_x_neg
            
        # r_i <= y_i (RHS is centers[i, 1])
        # Increasing y_i helps. Force +y.
        lam_y_pos = marginals[bound_offset + 4 * i + 2]
        if lam_y_pos > 1e-9:
            forces[i, 1] += lam_y_pos
            
        # r_i <= 1 - y_i (RHS is 1 - centers[i, 1])
        # Increasing 1-y_i (decreasing y_i) helps. Force -y.
        lam_y_neg = marginals[bound_offset + 4 * i + 3]
        if lam_y_neg > 1e-9:
            forces[i, 1] -= lam_y_neg
            
    return forces

def solve_lp(centers, n):
    """
    Solves the LP for radii given centers.
    Returns radii, marginals, and sum_radii.
    """
    # Objective: Minimize -sum(r) => c = [-1, -1, ..., -1]
    c_obj = np.ones(n) * -1.0
    
    # Constraints A_ub r <= b_ub
    # 1. Pairwise: r_i + r_j <= dist(i, j)
    num_pairs = n * (n - 1) // 2
    num_bound = 4 * n
    num_constraints = num_pairs + num_bound
    
    A_ub = np.zeros((num_constraints, n))
    b_ub = np.zeros(num_constraints)
    
    idx = 0
    # Pairwise
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            A_ub[idx, i] = 1.0
            A_ub[idx, j] = 1.0
            b_ub[idx] = dist
            idx += 1
            
    # Boundary
    for i in range(n):
        # r_i <= x_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = centers[i, 0]
        idx += 1
        
        # r_i <= 1 - x_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - centers[i, 0]
        idx += 1
        
        # r_i <= y_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = centers[i, 1]
        idx += 1
        
        # r_i <= 1 - y_i
        A_ub[idx, i] = 1.0
        b_ub[idx] = 1.0 - centers[i, 1]
        idx += 1
        
    # Bounds for r: r_i >= 0
    bounds = [(0, None) for _ in range(n)]
    
    # Solve using HiGHS solver which supports marginals
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if not res.success:
        # Fallback or handle error
        # If infeasible (should not happen with valid centers), return zeros
        return np.zeros(n), np.zeros(num_constraints), 0.0
        
    radii = res.x
    # Marginals might not be available in all scipy versions/methods, 
    # but 'highs' usually provides them in result.marginals.
    # Note: result.marginals length should match num_constraints
    if hasattr(res, 'marginals') and res.marginals is not None:
        marginals = res.marginals
    else:
        # If marginals not available, return zeros (optimization will stall or need heuristic)
        marginals = np.zeros(num_constraints)
        
    sum_radii = np.sum(radii)
    return radii, marginals, sum_radii

def run_packing():
    n = 26
    
    # Initialize centers
    # Using a grid layout to start with good separation
    rng = np.random.default_rng(42)
    
    # 5 rows, 6 columns grid
    # We have 26 circles. 5*5=25, so 5x5 grid plus 1.
    # Or 5x6 grid (30 spots) and pick 26?
    # Let's just scatter them slightly randomly on a grid to break symmetry.
    
    # Grid points
    grid_x = np.linspace(0.1, 0.9, 6)
    grid_y = np.linspace(0.1, 0.9, 5)
    
    points = []
    for y in grid_y:
        for x in grid_x:
            points.append([x, y])
    # points has 30 items. Take first 26.
    initial_centers = np.array(points[:n])
    
    # Add small random perturbation
    initial_centers += rng.uniform(-0.05, 0.05, size=initial_centers.shape)
    # Clip to [0,1]
    np.clip(initial_centers, 0.0, 1.0, out=initial_centers)
    
    centers = initial_centers.copy()
    
    # Optimization parameters
    learning_rate = 0.05
    decay = 0.98
    max_iter = 1500
    
    best_sum_radii = 0.0
    best_centers = centers.copy()
    best_radii = np.zeros(n)
    
    for it in range(max_iter):
        # Solve LP
        radii, marginals, current_sum = solve_lp(centers, n)
        
        if current_sum > best_sum_radii:
            best_sum_radii = current_sum
            best_centers = centers.copy()
            best_radii = radii.copy()
            
        # Compute forces
        forces = compute_forces_from_marginals(centers, marginals, n)
        
        # Update centers
        centers += learning_rate * forces
        
        # Clip centers to valid range [0, 1]
        np.clip(centers, 1e-6, 1.0 - 1e-6, out=centers)
        
        # Decay learning rate
        learning_rate *= decay
        
        if it % 100 == 0:
            print(f"Iter {it}, Sum Radii: {current_sum:.5f}, LR: {learning_rate:.5f}")
            
    # Use best solution found
    centers = best_centers
    radii = best_radii
    
    # Final check/cleanup: re-solve LP with best centers to ensure consistency
    # (Though best_radii was captured from a successful solve)
    # But centers might have moved slightly if we didn't capture state perfectly.
    # Let's just re-solve to be safe.
    radii, _, final_sum = solve_lp(centers, n)
    
    # Ensure no NaNs
    if np.isnan(centers).any() or np.isnan(radii).any():
        print("Warning: NaN detected, resetting to valid config")
        # Fallback to simple grid if something broke
        centers = initial_centers
        radii, _, final_sum = solve_lp(centers, n)

    return centers, radii, final_sum
