# sol_000240 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 06f8ea92) state=d0c80cd5 sum of radii=1.759172 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
import scipy.optimize as opt
import math

def solve_lp_for_radii(centers):
    """
    Given fixed centers, solve LP to maximize sum of radii.
    
    Constraints:
    1. r_i >= 0
    2. r_i <= x_i
    3. r_i <= 1 - x_i
    4. r_i <= y_i
    5. r_i <= 1 - y_i
    6. r_i + r_j <= distance(i, j)
    
    Objective: Maximize sum(r_i)  <=> Minimize -sum(r_i)
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Objective coefficients: -1 for all r_i
    c = -np.ones(n)
    
    # Inequality constraints matrix A_ub * r <= b_ub
    # We collect all constraints here.
    
    constraints_list = []
    bounds_list = []
    
    # 1. Non-negativity: r_i >= 0
    # This is handled by bounds in linprog
    
    for i in range(n):
        x, y = centers[i]
        
        # 2. r_i <= x_i  => -r_i >= -x_i (standard form: A*r <= b)
        # Actually linprog minimizes c^T x subject to A_ub x <= b_ub
        # r_i <= x_i  =>  [0...1...0] * r <= x_i
        row = np.zeros(n)
        row[i] = 1.0
        constraints_list.append((row, x))
        
        # 3. r_i <= 1 - x_i
        constraints_list.append((row, 1.0 - x))
        
        # 4. r_i <= y_i
        constraints_list.append((row, y))
        
        # 5. r_i <= 1 - y_i
        constraints_list.append((row, 1.0 - y))
        
        # Bounds: r_i >= 0
        bounds_list.append((0, None))

    # 6. r_i + r_j <= dist_ij
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            row = np.zeros(n)
            row[i] = 1.0
            row[j] = 1.0
            constraints_list.append((row, dist))
            
    if not constraints_list:
        return np.zeros(n)

    A_ub = np.array([item[0] for item in constraints_list])
    b_ub = np.array([item[1] for item in constraints_list])
    
    # Solve LP
    # linprog minimizes, so we minimize -sum(r)
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds_list, method='highs')
    
    if res.success:
        return res.x
    else:
        # Fallback if LP fails (shouldn't happen usually)
        return np.zeros(n)

def compute_sum_radii(centers):
    """Helper to compute sum of radii for a set of centers."""
    radii = solve_lp_for_radii(centers)
    return np.sum(radii)

def get_initial_centers(n, method='random'):
    """Generate initial centers."""
    if method == 'random':
        centers = np.random.rand(n, 2)
        # Ensure they are somewhat separated to start? 
        # No, let LP handle radii. But very close centers will result in 0 radii.
        # Let's spread them out.
        indices = np.random.choice(100, n, replace=False)
        # Map to grid roughly
        rows = 10
        cols = 10
        r_idx = indices // cols
        c_idx = indices % cols
        centers = np.column_stack([
            (c_idx + 0.5) / cols,
            (r_idx + 0.5) / rows
        ])
        # Add small jitter
        centers += np.random.uniform(-0.02, 0.02, (n, 2))
        centers = np.clip(centers, 0.01, 0.99)
        return centers
    elif method == 'grid':
        # Try to fit in a grid
        # 26 circles. 5x6 grid has 30 slots.
        # Let's pick 26 random slots in 5x6 grid
        rows = 5
        cols = 6
        grid_x = np.linspace(0.1, 0.9, cols) # 6 points
        grid_y = np.linspace(0.1, 0.9, rows) # 5 points
        xs, ys = np.meshgrid(grid_x, grid_y)
        all_points = np.column_stack([xs.ravel(), ys.ravel()])
        
        # Pick 26 random distinct points
        idx = np.random.choice(len(all_points), n, replace=False)
        return all_points[idx]

def refine_packing(centers):
    """
    Use local optimization to refine centers and radii simultaneously.
    We optimize variables [x_0, y_0, r_0, ..., x_25, y_25, r_25]
    to maximize sum(r_i).
    """
    n = centers.shape[0]
    radii_init = solve_lp_for_radii(centers)
    
    # Flatten variables
    x0 = np.concatenate([centers.flatten(), radii_init])
    
    def objective(vars_flat):
        centers_opt = vars_flat[:2*n].reshape(n, 2)
        radii_opt = vars_flat[2*n:]
        return -np.sum(radii_opt) # Minimize negative sum

    def penalty(vars_flat):
        centers_opt = vars_flat[:2*n].reshape(n, 2)
        radii_opt = vars_flat[2*n:]
        
        penalty_val = 0.0
        scale = 1000.0
        
        # Boundary constraints
        for i in range(n):
            x, y = centers_opt[i]
            r = radii_opt[i]
            # r <= x
            if r - x > 0: penalty_val += scale * (r - x)**2
            # r <= 1-x
            if r - (1-x) > 0: penalty_val += scale * (r - (1-x))**2
            # r <= y
            if r - y > 0: penalty_val += scale * (r - y)**2
            # r <= 1-y
            if r - (1-y) > 0: penalty_val += scale * (r - (1-y))**2
            # r >= 0
            if r < 0: penalty_val += scale * r**2
            
        # Overlap constraints
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(centers_opt[i] - centers_opt[j])
                sum_r = radii_opt[i] + radii_opt[j]
                if sum_r - dist > 0:
                    penalty_val += scale * (sum_r - dist)**2
                    
        return penalty_val

    # Bounds for variables
    bounds = []
    for _ in range(n):
        bounds.append((0, 1)) # x
        bounds.append((0, 1)) # y
        bounds.append((0, 1)) # r

    # Optimization
    res = opt.minimize(
        objective, 
        x0, 
        method='SLSQP',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    best_vars = res.x
    best_centers = best_vars[:2*n].reshape(n, 2)
    best_radii = best_vars[2*n:]
    
    # Post-process: Re-run LP on optimized centers to ensure radii are optimal for those centers
    # and to clean up any numerical slack.
    final_radii = solve_lp_for_radii(best_centers)
    
    # Clip radii slightly to be safe if needed, but LP should handle it.
    # However, ensure non-negativity
    final_radii = np.maximum(final_radii, 0)
    
    return best_centers, final_radii, np.sum(final_radii)

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # Strategy: Multiple restarts
    # 1. Random restarts with LP evaluation
    # 2. Grid-based restarts with LP evaluation
    # 3. Refine best candidates
    
    num_restarts = 50
    
    candidates = []
    
    for _ in range(num_restarts):
        # Try random grid perturbation
        centers = get_initial_centers(n, method='random')
        s = compute_sum_radii(centers)
        candidates.append((s, centers))
        
    # Sort candidates
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 5 and refine them
    top_k = 5
    for _, centers in candidates[:top_k]:
        try:
            c, r, s = refine_packing(centers)
            if s > best_sum:
                best_sum = s
                best_centers = c
                best_radii = r
        except Exception as e:
            pass
            
    # If no good result found (unlikely), fallback to grid
    if best_sum < 1.0:
        centers = get_initial_centers(n, method='grid')
        best_radii = solve_lp_for_radii(centers)
        best_centers = centers
        best_sum = np.sum(best_radii)

    # Final validation and clamping to ensure strict constraints
    # Sometimes optimization might drift slightly outside.
    # We clamp centers to be at least radius away from boundaries.
    for i in range(n):
        x, y = best_centers[i]
        r = best_radii[i]
        # Enforce boundary
        best_centers[i, 0] = np.clip(x, r, 1 - r)
        best_centers[i, 1] = np.clip(y, r, 1 - r)
        
    # Re-solve LP one last time to ensure radii are optimal for clamped centers
    final_radii = solve_lp_for_radii(best_centers)
    
    # Check overlaps and adjust if necessary (very small adjustments)
    # If overlaps exist, reduce radii slightly
    # But LP guarantees non-overlap for the centers provided.
    # So if we clamp centers, we might violate overlap?
    # Clamping moves center towards interior, which usually helps overlap (increases distance from boundary, but might decrease distance to neighbors?).
    # Actually clamping x to [r, 1-r] keeps it valid w.r.t boundaries.
    # It might move center closer to another circle.
    # So we should re-run LP on the clamped centers.
    
    best_radii = final_radii
    best_sum = np.sum(best_radii)
    
    return best_centers, best_radii, best_sum
