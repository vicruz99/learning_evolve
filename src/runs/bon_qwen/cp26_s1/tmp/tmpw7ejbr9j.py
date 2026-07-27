import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
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

def objective(params, n):
    """Objective function: minimize negative sum of radii"""
    radii = params[2*n:2*n+n]
    return -np.sum(radii)

def boundary_constraints(params, n):
    """Constraints for circles to be inside the unit square"""
    constraints = []
    for i in range(n):
        x = params[2*i]
        y = params[2*i+1]
        r = params[2*n+i]
        # x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[2*idx] - p[2*n+idx]})
        # x + r <= 1 => 1 - x - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: 1 - p[2*idx] - p[2*n+idx]})
        # y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: p[2*idx+1] - p[2*n+idx]})
        # y + r <= 1 => 1 - y - r >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, idx=i: 1 - p[2*idx+1] - p[2*n+idx]})
    return constraints

def non_overlap_constraints(params, n):
    """Constraints for non-overlapping circles"""
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            def dist_constraint(p, i=i, j=j, n=n):
                x1, y1 = p[2*i], p[2*i+1]
                x2, y2 = p[2*j], p[2*j+1]
                r1, r2 = p[2*n+i], p[2*n+j]
                dist_sq = (x1 - x2)**2 + (y1 - y2)**2
                r_sum = r1 + r2
                return dist_sq - r_sum**2
            constraints.append({'type': 'ineq', 'fun': dist_constraint})
    return constraints

def get_constraints(n):
    c_list = []
    c_list.extend(boundary_constraints(None, n)) # We need to pass n to the lambda, but boundary_constraints already does
    # Redefine to pass n correctly
    for i in range(n):
        c_list.append({'type': 'ineq', 'fun': lambda p, idx=i: p[2*idx] - p[2*n+idx]})
        c_list.append({'type': 'ineq', 'fun': lambda p, idx=i: 1 - p[2*idx] - p[2*n+idx]})
        c_list.append({'type': 'ineq', 'fun': lambda p, idx=i: p[2*idx+1] - p[2*n+idx]})
        c_list.append({'type': 'ineq', 'fun': lambda p, idx=i: 1 - p[2*idx+1] - p[2*n+idx]})
        
    for i in range(n):
        for j in range(i + 1, n):
            def make_fn(i_idx, j_idx, n_n):
                def fn(p):
                    x1, y1 = p[2*i_idx], p[2*i_idx+1]
                    x2, y2 = p[2*j_idx], p[2*j_idx+1]
                    r1, r2 = p[2*n_n+i_idx], p[2*n_n+j_idx]
                    return (x1-x2)**2 + (y1-y2)**2 - (r1+r2)**2
                return fn
            c_list.append({'type': 'ineq', 'fun': make_fn(i, j, n)})
    return c_list

def run_packing():
    n = 26
    best_sum = 0.0
    best_params = None
    
    # Initial configurations
    initial_configs = []
    
    # 1. Hexagonal lattice approximation
    # Try to fit 26 circles in a hexagonal pattern
    # Rows: 5, 5, 5, 5, 4, 2? Or 6, 5, 5, 5, 5?
    # Let's try a generic dense packing initialization
    # 6 rows: 5, 5, 5, 5, 5, 1 (Sum 26)
    # Or 5, 5, 6, 5, 5 (Sum 26) -> 5 rows
    # Let's try 5 rows with varying counts to fit 26
    # 6, 5, 5, 5, 5
    rows_config = [6, 5, 5, 5, 5]
    centers_init = []
    r_init = 0.09
    y = r_init
    for count in rows_config:
        # Horizontal spacing
        if count > 1:
            x_start = r_init
            # Adjust x_start to center the row if count is odd/even?
            # Simple grid placement first
            x = x_start
            for k in range(count):
                centers_init.append([x, y])
                x += 2 * r_init
        else:
            centers_init.append([0.5, y])
        y += np.sqrt(3) * r_init
    
    # Pad to 26 if needed or trim
    # The above might not produce exactly 26 or valid bounds immediately, 
    # but serves as a seed.
    # Let's ensure we have 26
    while len(centers_init) < n:
        centers_init.append([0.5, 0.5])
    centers_init = np.array(centers_init[:n])
    
    # Scale to fit roughly
    max_x = np.max(centers_init[:, 0])
    max_y = np.max(centers_init[:, 1])
    scale_x = 1.0 / max_x if max_x > 0 else 1.0
    scale_y = 1.0 / max_y if max_y > 0 else 1.0
    # Don't scale too much, just shift to center
    centers_init[:, 0] = (centers_init[:, 0] - np.min(centers_init[:, 0])) / (np.max(centers_init[:, 0]) - np.min(centers_init[:, 0]) + 1e-9) * 0.8 + 0.1
    centers_init[:, 1] = (centers_init[:, 1] - np.min(centers_init[:, 1])) / (np.max(centers_init[:, 1]) - np.min(centers_init[:, 1]) + 1e-9) * 0.8 + 0.1
    
    initial_configs.append(centers_init)
    
    # 2. 5x5 grid + 1
    grid = []
    for i in range(5):
        for j in range(5):
            grid.append([0.1 + i*0.2, 0.1 + j*0.2])
    grid.append([0.5, 0.5])
    grid = np.array(grid[:n])
    initial_configs.append(grid)

    # 3. Random
    rng = np.random.RandomState(42)
    rand_centers = rng.rand(n, 2)
    initial_configs.append(rand_centers)

    for config in initial_configs:
        # Flatten params: x0, y0, ..., x25, y25, r0, ..., r25
        params = np.concatenate([config.flatten(), np.ones(n) * 0.05])
        
        # Bounds for radii (non-negative)
        bounds = [(0, 1)] * (2*n) + [(0, 0.5)] * n
        
        cons = get_constraints(n)
        
        try:
            res = minimize(objective, params, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 1000, 'ftol': 1e-9})
            
            if res.success:
                sum_radii = -res.fun
                if sum_radii > best_sum:
                    # Validate before updating
                    c = res.x[:2*n].reshape(n, 2)
                    r = res.x[2*n:]
                    # Clamp radii to be non-negative just in case
                    r = np.maximum(r, 0)
                    # Check if valid with tolerance
                    if validate_packing(c, r):
                        best_sum = sum_radii
                        best_params = (c, r)
        except Exception:
            continue
            
    if best_params is None:
        # Fallback
        c = np.zeros((n, 2))
        r = np.zeros(n)
        for i in range(n):
            c[i] = [0.5, 0.5]
            r[i] = 0.0
        return c, r, 0.0
        
    return best_params[0], best_params[1], best_sum