# sol_000244 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state e7c70ed6) state=d0493c30 sum of radii=1.337967 correctness=1.0
# stdout(first 200): Optimization failed at restart 0: The user-provided objective function must return a scalar value. Optimization failed at restart 1: The user-provided objective function must return a scalar value. Op
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog, minimize
import time

def solve_radii(centers):
    """
    Solves the LP to maximize sum of radii given fixed centers.
    Returns sum_radii, radii, and gradient of sum_radii w.r.t centers.
    """
    n = centers.shape[0]
    
    # Variables: r_0, ..., r_{n-1}
    # Objective: Maximize sum(r_i) -> Minimize -sum(r_i)
    c_obj = -np.ones(n)
    
    # Bounds for r_i: 0 <= r_i <= distance to nearest wall
    # We extract wall distances to set bounds, which is efficient.
    bounds = []
    wall_constraints_active = np.zeros(n) # To track if wall limits r_i
    
    # Precompute distances to walls for bounds
    x = centers[:, 0]
    y = centers[:, 1]
    
    dist_to_left = x
    dist_to_right = 1 - x
    dist_to_bottom = y
    dist_to_top =  - y + 1 # 1 - y
    
    # Upper bound for r_i is min(dist to walls)
    # However, strictly speaking, r_i <= x_i is a constraint.
    # If we put it in bounds, we don't get the dual variable.
    # But for gradient, if r_i is limited by wall, increasing x_i helps.
    # Let's stick to bounds for simplicity and handle wall gradient separately or ignore (rarely optimal to be limited by wall without neighbor contact in dense packing).
    # Actually, for dense packing, contacts with neighbors dominate.
    
    r_max = np.minimum(np.minimum(dist_to_left, dist_to_right), np.minimum(dist_to_bottom, dist_to_top))
    # Clip negative bounds to 0 just in case centers are out of [0,1]
    r_max = np.maximum(r_max, 0.0)
    
    for i in range(n):
        bounds.append((0, r_max[i]))
        
    # Constraints: r_i + r_j <= dist_ij
    # Matrix A_ub * r <= b_ub
    # We construct this sparsely or as a list of rows? 
    # For N=26, dense matrix is small enough (325 rows x 26 cols).
    
    num_pairs = n * (n - 1) // 2
    A_ub = np.zeros((num_pairs, n))
    b_ub = np.zeros(num_pairs)
    
    row_idx = 0
    # Precompute distances to avoid repeated sqrt? No, we need them for gradient anyway.
    # But for LP setup we need them.
    
    # Storing indices for gradient calculation
    pair_indices = [] 
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            A_ub[row_idx, i] = 1
            A_ub[row_idx, j] = 1
            b_ub[row_idx] = dist
            pair_indices.append((i, j))
            
            row_idx += 1
            
    # Solve LP
    # method='highs' is generally fastest and most robust
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            radii = res.x
            sum_radii = -res.fun
            
            # Compute Gradient w.r.t centers
            # Gradient of sum(r) w.r.t center_i comes from active distance constraints.
            # The dual variables (marginals) for inequality constraints correspond to A_ub constraints.
            # res.ineqlin.marginals gives duals for A_ub @ x <= b_ub.
            # If available. In older scipy, might be None.
            
            grad_centers = np.zeros_like(centers)
            
            # Check if marginals are available
            if hasattr(res, 'ineqlin') and hasattr(res.ineqlin, 'marginals') and res.ineqlin.marginals is not None:
                duals = res.ineqlin.marginals
                # duals[k] corresponds to k-th row in A_ub
                # Constraint k: r_i + r_j <= dist_ij
                # Shadow price lambda_k indicates how much objective increases if dist_ij increases by 1.
                # dist_ij = sqrt((xi-xj)^2 + (yi-yj)^2)
                # d(dist_ij)/d(xi) = (xi - xj) / dist_ij
                
                # We iterate through pairs
                # Note: pair_indices order must match A_ub rows
                for idx, (i, j) in enumerate(pair_indices):
                    lam = duals[idx]
                    if lam > 1e-9: # Only consider active constraints
                        # Distance between i and j
                        # Recompute to be safe or store?
                        dx = centers[i, 0] - centers[j, 0]
                        dy = centers[i, 1] - centers[j, 1]
                        dist = np.sqrt(dx*dx + dy*dy)
                        
                        if dist > 1e-9:
                            factor = lam / dist
                            grad_centers[i, 0] += factor * dx
                            grad_centers[i, 1] += factor * dy
                            grad_centers[j, 0] -= factor * dx
                            grad_centers[j, 1] -= factor * dy
            else:
                # Fallback: finite difference or just return zeros?
                # If no marginals, we can't compute gradient easily.
                # But 'highs' usually provides them.
                pass
                
            return sum_radii, radii, grad_centers
        else:
            return 0, np.zeros(n), np.zeros_like(centers)
    except Exception:
        return 0, np.zeros(n), np.zeros_like(centers)

def objective_with_grad(centers_flat):
    """
    Objective function for minimize.
    Returns (value, gradient).
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Ensure centers are within [0, 1] to avoid invalid bounds in LP
    # Although minimize with bounds handles this, clipping helps stability.
    centers = np.clip(centers, 1e-6, 1 - 1e-6)
    
    sum_r, radii, grad = solve_radii(centers)
    
    # We want to maximize sum_r, so minimize -sum_r
    val = -sum_r
    
    # Gradient of -sum_r is -grad
    grad_flat = (-grad).flatten()
    
    return val, grad_flat

def run_packing():
    n_circles = 26
    
    # Strategy: Multiple restarts with local optimization
    best_sum_r = 0
    best_centers = None
    best_radii = None
    
    num_restarts = 10
    
    # Seed for reproducibility? Or random is fine.
    # Using random seeds helps diversity.
    
    for restart in range(num_restarts):
        # Generate initial centers
        # Heuristic: Grid based with noise
        # Try to fit 26 points in square.
        # sqrt(26) ~ 5.1. 5x5 grid is 25.
        # Let's create a 6x5 grid and pick 26? Or just random.
        # Random is robust if we optimize well.
        
        np.random.seed(restart * 12345)
        
        # Generate points in [0.1, 0.9] initially
        init_centers = np.random.uniform(0.1, 0.9, size=(n_circles, 2))
        
        # Flatten for scipy
        x0 = init_centers.flatten()
        
        # Bounds for centers: [0, 1] for each coordinate
        # Actually, circles must be inside, so centers must be in [0,1].
        # But effectively [r, 1-r]. Since r is unknown, [0,1] is safe box.
        bnds = [(0, 1) for _ in range(2 * n_circles)]
        
        # Optimization
        # Using L-BFGS-B as it supports bounds and gradients
        try:
            res = minimize(
                fun=objective_with_grad,
                x0=x0,
                method='L-BFGS-B',
                bounds=bnds,
                options={'maxiter': 500, 'ftol': 1e-9, 'gtol': 1e-6}
            )
            
            if res.success or res.fun < -best_sum_r: # We minimize -sum, so lower is better
                sum_r = -res.fun
                if sum_r > best_sum_r:
                    best_sum_r = sum_r
                    best_centers = res.x.reshape(-1, 2)
                    # Recompute radii for final result
                    _, best_radii, _ = solve_radii(best_centers)
        except Exception as e:
            print(f"Optimization failed at restart {restart}: {e}")
            continue

    # If best_radii is None (all failed), fallback
    if best_radii is None:
        best_centers = np.random.uniform(0.1, 0.9, size=(n_circles, 2))
        _, best_radii, _ = solve_radii(best_centers)
        best_sum_r = np.sum(best_radii)

    # Final validation and return
    # The validate function checks strict constraints.
    # Our solver ensures constraints, but numerical noise might exist.
    # We trust the LP solution.
    
    # Just to be safe, ensure centers are strictly inside if radii are close to boundary?
    # LP bounds handled r <= dist_wall, so it should be valid.
    
    return best_centers, best_radii, float(best_sum_r)

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    # Basic check
    # from validation code
    import numpy as np
    def validate_packing(centers, radii):
        n = centers.shape[0]
        if np.isnan(centers).any() or np.isnan(radii).any(): return False
        for i in range(n):
            if radii[i] < 0: return False
            x, y = centers[i]
            r = radii[i]
            if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
                return False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if dist < radii[i] + radii[j] - 1e-9:
                    return False
        return True

    is_valid = validate_packing(centers, radii)
    print(f"Valid: {is_valid}")
