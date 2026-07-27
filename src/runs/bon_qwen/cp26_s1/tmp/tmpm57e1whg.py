import numpy as np
from scipy.optimize import minimize

def compute_objective(vars_flat):
    """Objective function: minimize negative sum of radii."""
    n = 26
    radii = vars_flat[2*n:]
    return -np.sum(radii)

def compute_constraints(vars_flat):
    """Inequality constraints: boundaries and non-overlap."""
    n = 26
    centers = vars_flat[:2*n].reshape(n, 2)
    radii = vars_flat[2*n:]
    
    num_constraints = n * 4 + n * (n - 1) // 2
    cons = np.empty(num_constraints)
    
    # Boundary constraints: x >= r, 1-x >= r, y >= r, 1-y >= r
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        cons[4*i] = x - r
        cons[4*i + 1] = 1.0 - x - r
        cons[4*i + 2] = y - r
        cons[4*i + 3] = 1.0 - y - r
        
    # Pairwise non-overlap constraints: dist^2 >= (r_i + r_j)^2
    idx = n * 4
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist_sq = dx*dx + dy*dy
            r_sum = radii[i] + radii[j]
            cons[idx] = dist_sq - r_sum*r_sum
            idx += 1
            
    return cons

def run_packing():
    n = 26
    best_obj = np.inf
    best_vars = None
    
    # Variable bounds: centers in [0,1], radii in [0, 0.5]
    bounds = [(0.0, 1.0)] * 2*n + [(0.0, 0.5)] * n
    constraints = {'type': 'ineq', 'fun': compute_constraints}
    options = {'ftol': 1e-8, 'maxiter': 3000, 'disp': False}
    
    # Generate multiple initial guesses to avoid local optima
    rng = np.random.RandomState(42)
    initial_guesses = []
    
    # 1. Structured Hexagonal Grids (scaled variations)
    for scale in [0.92, 0.96, 1.0]:
        positions = []
        count = 0
        s = 0.21 * scale
        for r in range(6):
            for c in range(6):
                if count == n: break
                x = c * s + (r % 2) * s * 0.5
                y = r * s * np.sqrt(3) / 2
                positions.append([x, y])
                count += 1
            if count == n: break
        centers_init = np.array(positions)
        # Normalize to fit comfortably inside [0,1]
        c_min = centers_init.min(axis=0)
        c_max = centers_init.max(axis=0)
        c_range = c_max - c_min
        if c_range.max() > 1e-6:
            centers_init = (centers_init - c_min) / c_range * 0.85 + 0.075
        radii_init = np.full(n, 0.08)
        initial_guesses.append(np.concatenate([centers_init.flatten(), radii_init]))
        
    # 2. Random Initializations
    for _ in range(8):
        centers_init = rng.rand(n, 2) * 0.8 + 0.1
        radii_init = np.full(n, 0.06 + rng.rand() * 0.03)
        initial_guesses.append(np.concatenate([centers_init.flatten(), radii_init]))
        
    # Optimization loop
    for x0 in initial_guesses:
        res = minimize(compute_objective, x0, method='SLSQP', bounds=bounds, 
                       constraints=constraints, options=options)
        if res.fun < best_obj:
            best_obj = res.fun
            best_vars = res.x
            
    centers = best_vars[:2*n].reshape(n, 2)
    radii = best_vars[2*n:]
    
    # Strict numerical clamping to ensure validity
    centers = np.clip(centers, 0.0, 1.0)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = float(np.sum(radii))
    return centers, radii, sum_radii