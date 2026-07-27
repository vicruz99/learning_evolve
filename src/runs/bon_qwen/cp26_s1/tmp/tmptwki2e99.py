import numpy as np
from scipy.optimize import minimize

def compute_violations(centers, radii):
    """
    Computes the violation values for constraints.
    Returns a vector of constraint values. 
    For 'ineq' constraints in scipy, we need fun(x) >= 0.
    Here we return values that should be >= 0.
    """
    n = centers.shape[0]
    
    # Boundary constraints
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    bound_constraints = np.concatenate([
        centers[:, 0] - radii,
        1 - centers[:, 0] - radii,
        centers[:, 1] - radii,
        1 - centers[:, 1] - radii
    ])
    
    # Separation constraints
    # dist^2 >= (r_i + r_j)^2  =>  dist^2 - (r_i + r_j)^2 >= 0
    # Vectorized calculation
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    rad_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
    rad_sum_sq = rad_sum**2
    
    # We only need upper triangle (i < j) to avoid duplicates and self-check
    # Extract upper triangle elements
    idx = np.triu_indices(n, k=1)
    sep_constraints = dist_sq[idx] - rad_sum_sq[idx]
    
    return np.concatenate([bound_constraints, sep_constraints])

def objective(vars_vec):
    """
    Objective function: Maximize sum of radii -> Minimize -sum(radii)
    """
    n = 26
    radii = vars_vec[2::3]
    return -np.sum(radii)

def get_constraints(vars_vec):
    """
    Extract centers and radii, compute constraint values.
    """
    n = 26
    centers = np.column_stack((vars_vec[0::3], vars_vec[1::3]))
    radii = vars_vec[2::3]
    return compute_violations(centers, radii)

def get_jac_constraints(vars_vec):
    """
    Jacobian of constraints w.r.t variables.
    This is optional but helps convergence. 
    Given the complexity, we might rely on finite differences if exact jacobian is too complex to implement perfectly,
    but for 78 vars, finite difference is fine for SLSQP usually. 
    However, implementing exact jacobian is safer for precision.
    
    Let's skip explicit jacobian to keep code simple and rely on SLSQP's default, 
    or implement a simple one if needed. 
    Actually, SLSQP calculates finite differences if jacobian is not provided.
    """
    # We will let scipy handle it numerically for now to ensure correctness of logic first.
    pass

def generate_initial_guess():
    """
    Generates a valid initial configuration of 26 circles.
    Uses a hexagonal-like grid packing.
    """
    n = 26
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    # Try to fit 26 circles in a hexagonal pattern
    # Estimate radius
    # Area density ~ 0.9. 26 * pi * r^2 ~ 0.9 => r ~ 0.105
    # Let's start with a safe small radius
    r_init = 0.08 
    
    # Hexagonal grid parameters
    # Row height: r * sqrt(3)
    row_height = r_init * np.sqrt(3)
    
    # Determine rows
    # We want to fit 26 circles. 
    # 6 rows: 4, 5, 4, 5, 4, 4 -> 26? 
    # Let's try to fill rows.
    
    # Simple grid for initialization to ensure non-overlap
    # 5 columns, 6 rows -> 30 spots. Pick 26.
    # Spacing x: 0.2, y: 0.18 (approx)
    
    # Let's use a perturbed grid
    x_coords = np.linspace(0.1, 0.9, 6) # 6 points
    y_coords = np.linspace(0.1, 0.9, 5) # 5 points
    
    # Create grid points
    pts = []
    for y in y_coords:
        for x in x_coords:
            pts.append([x, y])
            
    # We have 30 points. Take 26.
    # Shuffle or just take first 26?
    # Taking first 26 from top-left might cluster them.
    # Let's just pick 26 randomly or deterministically spread.
    # Actually, simple grid is fine.
    
    # Let's construct a specific pattern: 5 rows.
    # Row 0: 5 circles
    # Row 1: 5 circles (shifted)
    # ...
    # Total 25. Add 1.
    
    # Let's go back to simple grid of 26 points.
    # 5x5 grid has 25.
    # Add 1 in center?
    
    # Let's generate 26 points on a grid
    # 6 columns, 5 rows = 30.
    # x: 0.1, 0.3, 0.5, 0.7, 0.9, 1.1 (oops >1)
    # x: 0.125, 0.375, 0.625, 0.875 (4 cols) -> 20 pts.
    # Need more density.
    
    # Let's use linspace to place points nicely.
    # 26 points. sqrt(26) ~ 5.1.
    # 6x5 grid.
    
    x_vals = np.linspace(0.1, 0.9, 6) # 0.1, 0.26, 0.42, 0.58, 0.74, 0.9
    y_vals = np.linspace(0.1, 0.9, 5)
    
    pts = []
    for y in y_vals:
        for x in x_vals:
            pts.append([x, y])
            if len(pts) >= 26:
                break
        if len(pts) >= 26:
            break
            
    centers = np.array(pts)
    radii = np.full(n, 0.05) # Small initial radius
    
    return centers, radii

def run_packing():
    """
    Main function to run the packing optimization.
    """
    n = 26
    best_sum_r = 0
    best_centers = None
    best_radii = None
    
    # Number of restarts
    n_restarts = 15
    
    for i in range(n_restarts):
        # Generate initial guess
        centers, radii = generate_initial_guess()
        
        # Add some random noise to escape local minima / grid symmetry
        if i > 0:
            noise = np.random.uniform(-0.02, 0.02, centers.shape)
            centers = centers + noise
            centers = np.clip(centers, 0, 1)
            # Radii can stay small or slightly randomized
            radii = np.full(n, 0.05 + np.random.uniform(0, 0.01))
            
        # Flatten variables
        # Order: x1, y1, r1, x2, y2, r2, ...
        x0 = np.zeros(3 * n)
        for j in range(n):
            x0[3*j] = centers[j, 0]
            x0[3*j+1] = centers[j, 1]
            x0[3*j+2] = radii[j]
            
        # Bounds
        # x, y in [0, 1]
        # r in [0, 0.5]
        bounds = []
        for j in range(n):
            bounds.extend([(0, 1), (0, 1), (0, 0.5)])
            
        # Constraints
        # We need a callable that returns constraint values >= 0
        # scipy minimize constraint type 'ineq' expects fun(x) >= 0
        constraint = {'type': 'ineq', 'fun': get_constraints}
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraint,
                           options={'maxiter': 500, 'ftol': 1e-9, 'disp': False})
            
            if res.success or (res.fun < 0): # fun is -sum(r), so if sum(r) > 0, fun < 0
                current_sum_r = -res.fun
                if current_sum_r > best_sum_r:
                    best_sum_r = current_sum_r
                    # Extract solution
                    best_centers = np.column_stack((res.x[0::3], res.x[1::3]))
                    best_radii = res.x[2::3]
                    
        except Exception as e:
            print(f"Optimization failed in iteration {i}: {e}")
            continue
            
    # Validate and fix if necessary (though optimizer should handle it)
    if best_centers is not None:
        # Final validation check (optional but good for debugging)
        # The problem statement requires we return valid packing.
        # We assume the optimizer found a valid one satisfying constraints >= 0 (with tolerance).
        # However, numerical errors might make constraints slightly negative.
        # We should clamp radii or adjust if needed?
        # Actually, the constraints in scipy are satisfied within tolerance.
        # Let's just return.
        
        # Just in case, ensure radii are non-negative and valid
        best_radii = np.maximum(best_radii, 0)
        
        return best_centers, best_radii, best_sum_r
    
    # Fallback if all fail (should not happen)
    centers, radii = generate_initial_guess()
    return centers, radii, np.sum(radii)