# sol_000133 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 0b92a944) state=93f1d0a6 sum of radii=2.541421 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import linprog

def get_optimal_radii(centers):
    """
    Solves the Linear Programming problem to maximize sum of radii 
    given fixed centers.
    Constraints: r_i + r_j <= distance(i, j) and r_i <= boundary distance.
    """
    n = centers.shape[0]
    # Objective: maximize sum(r) <=> minimize -sum(r)
    c = -np.ones(n)
    
    rows = []
    b_vals = []
    
    for i in range(n):
        xi, yi = centers[i]
        
        # Boundary constraints: r_i <= xi, r_i <= 1-xi, r_i <= yi, r_i <= 1-yi
        # r_i <= xi
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b_vals.append(xi)
        
        # r_i <= 1 - xi
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b_vals.append(1 - xi)
        
        # r_i <= yi
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b_vals.append(yi)
        
        # r_i <= 1 - yi
        row = np.zeros(n)
        row[i] = 1
        rows.append(row)
        b_vals.append(1 - yi)
        
        # Pairwise constraints: r_i + r_j <= dist(i, j)
        # Optimization: only add constraint if dist < 1.0, because max(r_i) + max(r_j) <= 1.0
        for j in range(i + 1, n):
            dist = np.sqrt((xi - centers[j,0])**2 + (yi - centers[j,1])**2)
            if dist < 1.0: 
                row = np.zeros(n)
                row[i] = 1
                row[j] = 1
                rows.append(row)
                b_vals.append(dist)
            
    if len(rows) == 0:
        return np.zeros(n), 0.0

    A_ub = np.array(rows)
    b_ub = np.array(b_vals)
    
    bounds = [(0, None) for _ in range(n)]
    
    try:
        # Solve LP
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
        if res.success:
            return res.x, -res.fun
    except:
        pass
        
    return np.zeros(n), 0.0

def run_packing():
    n = 26
    best_sum = 0.0
    best_centers = None
    best_radii = None
    
    # List of initial configurations to try
    configs = []
    
    # 1. Random initialization
    configs.append(np.random.rand(n, 2))
    
    # 2. Grid 5x5 (25 circles) + 1 circle in a gap
    # This is a strong starting point with sum ~ 2.54
    grid_centers = []
    coords = [0.1, 0.3, 0.5, 0.7, 0.9]
    for x in coords:
        for y in coords:
            grid_centers.append([x, y])
    # Place 26th circle in a gap, e.g., (0.2, 0.2)
    grid_centers.append([0.2, 0.2])
    configs.append(np.array(grid_centers))
    
    # 3. Another random configuration with points slightly inset
    configs.append(np.random.rand(n, 2) * 0.8 + 0.1)

    for start_centers in configs:
        centers = start_centers.copy()
        
        # Get initial radii and sum
        current_radii, current_sum = get_optimal_radii(centers)
        
        # Hill climbing optimization
        step = 0.05
        for it in range(600):
            # Perturb centers
            new_centers = centers + np.random.normal(0, step, size=centers.shape)
            new_centers = np.clip(new_centers, 0, 1)
            
            new_radii, new_sum = get_optimal_radii(new_centers)
            
            if new_sum > current_sum:
                centers = new_centers
                current_radii = new_radii
                current_sum = new_sum
                step *= 0.98  # Reduce step size on success
            else:
                step *= 0.995 # Reduce step size slowly on failure
        
        if current_sum > best_sum:
            best_sum = current_sum
            best_centers = centers
            best_radii = current_radii
            
    # Refinement phase with smaller steps
    if best_centers is not None:
        current_centers = best_centers
        current_radii, current_sum = get_optimal_radii(current_centers)
        step = 0.01
        for it in range(1000):
            new_centers = current_centers + np.random.normal(0, step, size=current_centers.shape)
            new_centers = np.clip(new_centers, 0, 1)
            
            new_radii, new_sum = get_optimal_radii(new_centers)
            
            if new_sum > current_sum:
                current_centers = new_centers
                current_radii = new_radii
                current_sum = new_sum
                step *= 0.999
            else:
                step *= 0.9995
        
        best_centers = current_centers
        best_radii = current_radii
        best_sum = current_sum

    return best_centers, best_radii, best_sum
