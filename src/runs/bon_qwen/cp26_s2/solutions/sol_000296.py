# sol_000296 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 2d1ce3e9) state=738092f1 sum of radii=2.610526 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_hexagonal_lattice(n, scale=1.0):
    """
    Generate a hexagonal lattice arrangement of n points.
    This provides a dense initial configuration.
    """
    points = []
    row = 0
    while len(points) < n:
        # Estimate columns needed for this row
        # In hex packing, rows alternate width roughly
        # But for initialization, a simple staggered grid is enough
        # We try to fit points in a square roughly
        cols = int(np.ceil(np.sqrt(n * scale)))
        
        y = row * np.sqrt(3) / 2 * scale
        # Shift even/odd rows
        shift = (scale / 2) if row % 2 == 1 else 0
        
        for col in range(cols):
            if len(points) >= n:
                break
            x = col * scale + shift
            points.append([x, y])
        row += 1
    
    return np.array(points[:n])

def calculate_constraints(centers, radii):
    """
    Calculate constraint violations.
    Returns a list of values that should be >= 0.
    We minimize -sum(radii), so we want constraints to be satisfied.
    For scipy minimize with 'SLSQP', constraints are defined as functions 
    returning values >= 0.
    However, handling n^2 constraints in one go might be slow.
    We will construct the constraint function to return all values.
    """
    n = len(radii)
    violations = []
    
    # Boundary constraints
    # x_i - r_i >= 0
    violations.extend(centers[:, 0] - radii)
    # 1 - x_i - r_i >= 0
    violations.extend(1 - centers[:, 0] - radii)
    # y_i - r_i >= 0
    violations.extend(centers[:, 1] - radii)
    # 1 - y_i - r_i >= 0
    violations.extend(1 - centers[:, 1] - radii)
    
    # Overlap constraints: dist_ij >= r_i + r_j
    # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    # To avoid square roots and keep it smooth, we can use squared distance?
    # But (r_i + r_j)^2 is also quadratic.
    # Actually, dist >= r_i + r_j is equivalent to dist^2 >= (r_i + r_j)^2 only if dist >= 0 and r_i+r_j >= 0, which is true.
    # However, the derivative of sqrt is better behaved? 
    # Let's stick to distance - sum_radii >= 0 for better conditioning near 0?
    # Or squared. Squared is polynomial.
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            # Constraint: dist >= r_sum  => dist^2 >= r_sum^2
            # Violation: r_sum^2 - dist_sq <= 0  => dist_sq - r_sum^2 >= 0
            violations.append(dist_sq - r_sum * r_sum)
            
    return np.array(violations)

def objective_function(variables, n):
    """
    Objective: Maximize sum of radii.
    Minimize: - sum of radii.
    variables: [x1, y1, r1, x2, y2, r2, ...]
    """
    centers = variables[:2*n].reshape((n, 2))
    radii = variables[2*n:]
    return -np.sum(radii)

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # We will run optimization multiple times with different initializations
    num_runs = 20
    
    # Initial configurations
    # 1. Hexagonal lattice scaled to fit in unit square
    # 2. Random positions
    
    # Pre-calculate a hex lattice centered and scaled
    hex_pts = get_hexagonal_lattice(n)
    # Normalize to [0, 1]
    min_xy = np.min(hex_pts, axis=0)
    max_xy = np.max(hex_pts, axis=0)
    span = max_xy - min_xy
    span = np.where(span == 0, 1, span) # avoid div by zero
    hex_pts_norm = (hex_pts - min_xy) / span
    # Add a small margin to be strictly inside
    hex_pts_init = hex_pts_norm * 0.9 + 0.05 
    
    # Initial radius guess: slightly smaller than what fits
    r_init = 0.05 
    
    initial_configs = []
    
    # Config 1: Hex lattice
    v1 = np.concatenate([hex_pts_init.flatten(), np.full(n, r_init)])
    initial_configs.append(v1)
    
    # Config 2: Random points with small radii
    np.random.seed(42) # Fixed seed for reproducibility in this run, but we vary logic
    for _ in range(10):
        centers = np.random.rand(n, 2) * 0.8 + 0.1 # Keep away from edges initially
        radii = np.full(n, 0.05)
        initial_configs.append(np.concatenate([centers.flatten(), radii]))
        
    # Config 3: Grid based
    # 5x5 grid + 1
    pts = []
    for i in range(5):
        for j in range(5):
            pts.append([i*0.2 + 0.1, j*0.2 + 0.1])
    # Add one in a gap
    pts.append([0.2, 0.2]) 
    # Normalize? Already in [0.1, 0.9]
    # Actually 5x5 with spacing 0.2 gives centers at 0.1, 0.3... 0.9.
    # Width 0.8. Radius 0.1 fits.
    # We have 26 points.
    grid_pts = np.array(pts[:n])
    v3 = np.concatenate([grid_pts.flatten(), np.full(n, 0.1)])
    initial_configs.append(v3)

    # Optimization loop
    for idx, x0 in enumerate(initial_configs):
        try:
            # Bounds for variables
            # x, y in [0, 1]
            # r in [0, 0.5] (max possible radius)
            bounds = [(0, 1)] * (2 * n) + [(0, 0.5)] * n
            
            # Constraints
            # We define a constraint function that returns the vector of inequalities
            # scipy expects fun(x) >= 0
            def constr_fun(x):
                c = x[:2*n].reshape((n, 2))
                r = x[2*n:]
                return calculate_constraints(c, r)
            
            cons = {'type': 'ineq', 'fun': constr_fun}
            
            # Run minimization
            # SLSQP is good for non-linear constraints
            res = minimize(objective_function, x0, args=(n,), method='SLSQP', 
                           bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_sum: # Check if we found a better sum (negated)
                # Validate the result manually to be sure
                c_res = res.x[:2*n].reshape((n, 2))
                r_res = res.x[2*n:]
                
                # Check for NaN or negative radii
                if np.isnan(r_res).any() or np.any(r_res < 0):
                    continue
                
                # Basic validity check
                valid = True
                # Boundary check
                if np.any(c_res[:, 0] - r_res < -1e-9) or np.any(c_res[:, 0] + r_res > 1 + 1e-9):
                    valid = False
                if np.any(c_res[:, 1] - r_res < -1e-9) or np.any(c_res[:, 1] + r_res > 1 + 1e-9):
                    valid = False
                
                # Overlap check
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = np.linalg.norm(c_res[i] - c_res[j])
                        if dist < r_res[i] + r_res[j] - 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    current_sum = np.sum(r_res)
                    if current_sum > best_sum:
                        best_sum = current_sum
                        best_centers = c_res.copy()
                        best_radii = r_res.copy()
                        
        except Exception as e:
            # If optimization fails, continue with next config
            pass

    # If no valid packing found (unlikely), return a safe default (small circles in grid)
    if best_centers is None:
        centers = np.zeros((n, 2))
        radii = np.zeros(n)
        # Simple grid
        count = 0
        for i in range(5):
            for j in range(5):
                if count < n:
                    centers[count] = [0.1 + i*0.2, 0.1 + j*0.2]
                    radii[count] = 0.1
                    count += 1
        # Fill remaining
        while count < n:
            centers[count] = [0.2, 0.2] # Just place somewhere valid? 
            # Actually 0.2, 0.2 is occupied.
            # Just place tiny circles
            radii[count] = 0.001
            centers[count] = [0.5, 0.5]
            count += 1
        best_centers = centers
        best_radii = radii
        best_sum = np.sum(radii)

    return best_centers, best_radii, float(best_sum)
