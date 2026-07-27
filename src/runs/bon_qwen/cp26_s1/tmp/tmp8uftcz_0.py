import numpy as np
from scipy.optimize import minimize

def objective(x):
    """Objective function to maximize sum of radii (minimize negative sum)."""
    return -np.sum(x[2::3])

def constraint_boundary(x):
    """Constraints ensuring circles stay within the unit square."""
    n = 26
    X = x[0::3]
    Y = x[1::3]
    R = x[2::3]
    c = np.empty(4 * n)
    # x >= r, 1 - (x + r) >= 0, y >= r, 1 - (y + r) >= 0
    c[0::4] = X - R
    c[1::4] = 1.0 - (X + R)
    c[2::4] = Y - R
    c[3::4] = 1.0 - (Y + R)
    return c

def constraint_overlap(x):
    """Constraints ensuring circles do not overlap."""
    n = 26
    X = x[0::3]
    Y = x[1::3]
    R = x[2::3]
    m = n * (n - 1) // 2
    c = np.empty(m)
    idx = 0
    for i in range(n):
        xi, yi, ri = X[i], Y[i], R[i]
        xj = X[i+1:]
        yj = Y[i+1:]
        rj = R[i+1:]
        dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        c[idx:idx+len(rj)] = dist - (ri + rj)
        idx += len(rj)
    return c

def run_packing():
    n = 26
    # Initialize centers in a hexagonal grid pattern
    # Rows: 5, 5, 5, 5, 6 circles = 26 total
    rows = [5, 5, 5, 5, 6]
    centers = []
    for i, cols in enumerate(rows):
        for j in range(cols):
            x = j + 0.5 * (i % 2)
            y = i * np.sqrt(3) / 2
            centers.append([x, y])
            
    centers = np.array(centers)
    # Normalize to unit square and add small margin for optimization freedom
    centers -= centers.min(axis=0)
    centers /= centers.max(axis=0)
    centers = 0.9 * centers + 0.05
    
    # Initial radii
    radii = np.full(n, 0.09)
    
    # Flatten variables: [x1, y1, r1, x2, y2, r2, ...]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i] = centers[i, 0]
        x0[3*i+1] = centers[i, 1]
        x0[3*i+2] = radii[i]
        
    # Bounds for variables
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n
    
    # Setup constraints
    cons = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap}
    ]
    
    # Run optimization
    res = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons, 
                   options={'maxiter': 10000, 'ftol': 1e-12, 'disp': False})
    
    x_opt = res.x if res.success else x0
        
    centers_opt = np.column_stack((x_opt[0::3], x_opt[1::3]))
    radii_opt = x_opt[2::3]
    radii_opt = np.maximum(radii_opt, 0.0)
    
    # Post-optimization refinement: clamp radii to exact feasible limits
    # This handles any numerical slack and guarantees validity
    X = centers_opt[:, 0]
    Y = centers_opt[:, 1]
    for i in range(n):
        max_r = min(X[i], 1.0 - X[i], Y[i], 1.0 - Y[i])
        for j in range(n):
            if i != j:
                dist = np.sqrt((X[i] - X[j])**2 + (Y[i] - Y[j])**2)
                max_r = min(max_r, dist / 2.0)
        radii_opt[i] = max_r
        
    sum_radii = np.sum(radii_opt)
    return centers_opt, radii_opt, sum_radii