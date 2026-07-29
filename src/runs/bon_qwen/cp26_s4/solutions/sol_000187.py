# sol_000187 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 083f9270) state=2dfc8aa5 sum of radii=2.582842 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def objective(x, n):
    """Objective function to maximize sum of radii (minimize negative sum)."""
    return -np.sum(x[2::3])

def boundary_x_ge_r(x, i):
    """Constraint: x_i >= r_i"""
    return x[3*i] - x[3*i+2]

def boundary_x_le_1_r(x, i):
    """Constraint: x_i + r_i <= 1"""
    return 1 - x[3*i] - x[3*i+2]

def boundary_y_ge_r(x, i):
    """Constraint: y_i >= r_i"""
    return x[3*i+1] - x[3*i+2]

def boundary_y_le_1_r(x, i):
    """Constraint: y_i + r_i <= 1"""
    return 1 - x[3*i+1] - x[3*i+2]

def radius_nonneg(x, i):
    """Constraint: r_i >= 0"""
    return x[3*i+2]

def non_overlap(x, i, j):
    """Constraint: distance squared >= (r_i + r_j)^2"""
    dx = x[3*i] - x[3*j]
    dy = x[3*i+1] - x[3*j+1]
    dr = x[3*i+2] + x[3*j+2]
    return dx*dx + dy*dy - dr*dr

def constraints_factory(n):
    """Factory to create list of constraints for n circles."""
    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': boundary_x_ge_r, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_x_le_1_r, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_y_ge_r, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': boundary_y_le_1_r, 'args': (i,)})
        cons.append({'type': 'ineq', 'fun': radius_nonneg, 'args': (i,)})
        
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq', 'fun': non_overlap, 'args': (i, j)})
    return cons

def generate_initial_guesses(n):
    """Generate multiple initial configurations to avoid local optima."""
    guesses = []
    
    # 1. Hexagonal packing approximation
    rows = 5
    cols_base = n // rows
    rem = n % rows
    row_counts = [cols_base + (1 if i < rem else 0) for i in range(rows)]
    
    r_est = 0.05
    centers_hex = []
    radii_hex = []
    dy = r_est * np.sqrt(3)
    y = r_est
    for i, k in enumerate(row_counts):
        x_start = r_est
        # Stagger odd rows
        if i % 2 == 1:
            x_start = r_est + r_est
        for j in range(k):
            x = x_start + j * (2 * r_est)
            centers_hex.append([x, y])
            radii_hex.append(r_est)
        y += dy
    
    if len(centers_hex) >= n:
        centers_hex = np.array(centers_hex[:n])
        radii_hex = np.array(radii_hex[:n])
        x0_hex = np.zeros(n * 3)
        for i in range(n):
            x0_hex[3*i] = centers_hex[i, 0]
            x0_hex[3*i+1] = centers_hex[i, 1]
            x0_hex[3*i+2] = radii_hex[i]
        guesses.append(x0_hex)

    # 2. Square Grid approximation
    centers_grid = []
    radii_grid = []
    step = 0.2
    start = 0.1
    count = 0
    for r in range(5):
        for c in range(5):
            if count < n:
                centers_grid.append([start + c*step, start + r*step])
                radii_grid.append(0.09)
                count += 1
    while count < n:
        centers_grid.append([0.5, 0.5])
        radii_grid.append(0.01)
        count += 1
        
    centers_grid = np.array(centers_grid)
    radii_grid = np.array(radii_grid)
    x0_grid = np.zeros(n * 3)
    for i in range(n):
        x0_grid[3*i] = centers_grid[i, 0]
        x0_grid[3*i+1] = centers_grid[i, 1]
        x0_grid[3*i+2] = radii_grid[i]
    guesses.append(x0_grid)
    
    # 3. Random initialization (clamped to avoid immediate bound violations)
    np.random.seed(123)
    centers_rand = 0.05 + 0.9 * np.random.rand(n, 2)
    radii_rand = np.ones(n) * 0.05
    x0_rand = np.zeros(n * 3)
    for i in range(n):
        x0_rand[3*i] = centers_rand[i, 0]
        x0_rand[3*i+1] = centers_rand[i, 1]
        x0_rand[3*i+2] = radii_rand[i]
    guesses.append(x0_rand)
    
    return guesses

def run_packing():
    n = 26
    # Bounds for x, y in [0, 1] and r in [0, 1]
    bounds = [(0, 1) for _ in range(n * 2)] + [(0, 1) for _ in range(n)]
    cons = constraints_factory(n)
    guesses = generate_initial_guesses(n)
    
    best_x = None
    best_val = np.inf
    
    # Run optimization for each initial guess
    for x0 in guesses:
        try:
            res = minimize(objective, x0, args=(n,), method='SLSQP', bounds=bounds, constraints=cons, 
                           options={'maxiter': 2000, 'ftol': 1e-10})
            # We minimize negative sum, so lower is better
            if res.fun < best_val:
                best_val = res.fun
                best_x = res.x
        except:
            pass
            
    if best_x is None:
        # Fallback solution
        centers = np.random.rand(n, 2)
        radii = np.ones(n) * 0.01
        return centers, radii, np.sum(radii)
        
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    for i in range(n):
        centers[i, 0] = best_x[3*i]
        centers[i, 1] = best_x[3*i+1]
        radii[i] = best_x[3*i+2]
        
    return centers, radii, np.sum(radii)
