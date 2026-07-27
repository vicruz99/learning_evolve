import numpy as np
from scipy.optimize import minimize

def eval_constraints(v, n):
    """
    Evaluate constraint functions.
    Returns an array where each element must be >= 0 for a valid packing.
    Constraints:
    1. r_i >= 0
    2. x_i - r_i >= 0
    3. 1 - x_i - r_i >= 0
    4. y_i - r_i >= 0
    5. 1 - y_i - r_i >= 0
    6. dist(i,j) - (r_i + r_j) >= 0
    """
    m = 5 * n + n * (n - 1) // 2
    vals = np.empty(m)
    idx = 0
    for i in range(n):
        x, y, r = v[3*i], v[3*i+1], v[3*i+2]
        vals[idx] = r
        idx += 1
        vals[idx] = x - r
        idx += 1
        vals[idx] = 1.0 - x - r
        idx += 1
        vals[idx] = y - r
        idx += 1
        vals[idx] = 1.0 - y - r
        idx += 1
        
    for i in range(n):
        xi, yi, ri = v[3*i], v[3*i+1], v[3*i+2]
        for j in range(i + 1, n):
            xj, yj, rj = v[3*j], v[3*j+1], v[3*j+2]
            d = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            vals[idx] = d - (ri + rj)
            idx += 1
    return vals

def compute_objective(v, n):
    """
    Objective function to minimize: negative sum of radii.
    Minimizing this is equivalent to maximizing sum of radii.
    """
    return -np.sum(v[2::3])

def generate_initial(n):
    """
    Generate an initial feasible configuration using a hexagonal grid pattern.
    This places circles close to a dense packing configuration.
    """
    pts = np.zeros((n, 3))
    r_init = 0.02
    dx = 2 * r_init
    dy = r_init * np.sqrt(3)
    idx = 0
    y = r_init + 0.15
    row = 0
    while idx < n:
        x = r_init + 0.15 + (row % 2) * (dx / 2)
        while x < 1.0 - r_init - 0.15 and idx < n:
            pts[idx, 0] = x
            pts[idx, 1] = y
            pts[idx, 2] = r_init
            idx += 1
            x += dx
        y += dy
        row += 1
    return pts.flatten()

def run_packing():
    n = 26
    best_v = None
    best_obj = 0.0
    
    # Bounds for variables: x in [0,1], y in [0,1], r >= 0
    bounds = [(0.0, 1.0)] * (2 * n) + [(0.0, None)] * n
    cons = {'type': 'ineq', 'fun': eval_constraints, 'args': (n,)}
    
    # Run multiple trials with different initial perturbations to escape local minima
    for trial in range(5):
        v0 = generate_initial(n)
        
        # Add small random noise to break symmetry and explore different configurations
        noise = np.random.normal(0, 0.005, size=v0.shape)
        v0 = v0 + noise
        
        # Ensure radii remain positive after noise
        v0[2::3] = np.clip(v0[2::3], 0.01, None)
        
        res = minimize(compute_objective, v0, args=(n,), method='SLSQP',
                       bounds=bounds, constraints=cons,
                       options={'maxiter': 3000, 'ftol': 1e-12, 'disp': False})
        
        # res.fun is negative sum of radii. Smaller means better.
        if res.success and res.fun < best_obj:
            best_obj = res.fun
            best_v = res.x
            
    centers = best_v[:2*n].reshape(n, 2)
    radii = best_v[2::3]
    return centers, radii, np.sum(radii)