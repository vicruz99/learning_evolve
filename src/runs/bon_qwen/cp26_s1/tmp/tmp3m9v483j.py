import numpy as np
from scipy.optimize import minimize

def compute_max_radius(centers):
    """Compute the maximum feasible equal radius for a given set of centers."""
    n = centers.shape[0]
    r_min = 1.0
    
    # Boundary constraints
    for i in range(n):
        r_min = min(r_min, 
                    centers[i, 0], 1.0 - centers[i, 0], 
                    centers[i, 1], 1.0 - centers[i, 1])
                    
    # Pairwise non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            r_min = min(r_min, dist / 2.0)
            
    return r_min

def objective_func(vars):
    """Objective: maximize radius r (minimize -r)."""
    return -vars[-1]

def constraint_func(vars):
    """Constraints: boundaries and pairwise distances."""
    n = (len(vars) - 1) // 2
    centers = np.asarray(vars[:n*2]).reshape(n, 2)
    r = vars[-1]
    
    con = []
    # Boundary constraints
    for i in range(n):
        con.append(centers[i, 0] - r)
        con.append(1.0 - centers[i, 0] - r)
        con.append(centers[i, 1] - r)
        con.append(1.0 - centers[i, 1] - r)
        
    # Pairwise constraints: dist^2 >= 4r^2
    for i in range(n):
        for j in range(i + 1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            con.append(dx*dx + dy*dy - 4.0*r*r)
            
    return np.array(con)

def run_packing():
    n = 26
    
    # Initial configuration: hexagonal-ish grid
    pts = []
    rows_cols = [5, 5, 5, 5, 6]
    y_vals = np.linspace(0.1, 0.9, 5)
    
    for i, num_cols in enumerate(rows_cols):
        y = y_vals[i]
        # Shift odd rows for hexagonal packing
        shift = 0.1 if i % 2 == 1 else 0.0
        
        if num_cols == 1:
            x_vals = [0.5]
        else:
            x_vals = np.linspace(0.1, 0.9, num_cols) + shift
            
        for x in x_vals:
            # Clamp to valid range
            x = max(0.001, min(0.999, x))
            pts.append([x, y])
            
    # Flatten initial variables
    x0 = np.zeros(n * 2 + 1)
    x0[:n*2] = np.array(pts).flatten()
    x0[-1] = 0.1  # Initial radius guess
    
    bounds = [(0.0, 1.0)] * (n * 2) + [(0.0, 0.5)]
    cons = {'type': 'ineq', 'fun': constraint_func}
    
    # Run optimization
    res = minimize(objective_func, x0, method='SLSQP', 
                   bounds=bounds, constraints=cons, 
                   options={'maxiter': 5000, 'ftol': 1e-12})
                   
    best_centers = res.x[:n*2].reshape(n, 2)
    
    # Recompute exact feasible radius to guarantee validity
    r_feas = compute_max_radius(best_centers)
    best_radii = np.full(n, r_feas)
    total = float(np.sum(best_radii))
    
    return best_centers, best_radii, total