import numpy as np
from scipy.optimize import minimize

def run_packing():
    # Constants
    n = 26
    
    # 1. Generate a good initial configuration using a hexagonal lattice
    # We arrange points in 6 rows to utilize hexagonal density.
    # Pattern: 5, 5, 5, 5, 5, 1 points in rows (shifted alternately)
    # This forms a compact cluster.
    
    centers = []
    rows_config = [5, 5, 5, 5, 5, 1]
    row_idx = 0
    
    # We build the points in a canonical hex lattice with unit distance 2 (for easier scaling later)
    # Actually, let's just use coordinates and let the optimizer handle scale.
    # Initial spacing roughly 0.25
    spacing_x = 0.2
    spacing_y = 0.2 * np.sqrt(3) / 2
    
    y = 0.15 # Start near bottom
    
    for count in rows_config:
        # Determine x start based on row parity (staggered)
        if row_idx % 2 == 0:
            x_start = 0.15
        else:
            x_start = 0.15 + spacing_x / 2
            
        for i in range(count):
            x = x_start + i * spacing_x
            centers.append([x, y])
        
        y += spacing_y
        row_idx += 1
        
    centers = np.array(centers)
    
    # Ensure we have exactly 26 points
    if len(centers) > n:
        centers = centers[:n]
    elif len(centers) < n:
        # Fallback: add points if needed (though logic above guarantees 26)
        pass

    # Initial radii guess: uniform, slightly less than half min distance
    # Calculate min distance in initial config
    min_dist = np.inf
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            d = np.linalg.norm(centers[i] - centers[j])
            if d < min_dist:
                min_dist = d
    r_init = min_dist * 0.4  # Initial radius to avoid overlap
    radii = np.full(n, r_init)

    # 2. Optimization
    # We flatten centers and radii into a single vector for scipy.optimize
    # Variables: x1, y1, r1, x2, y2, r2, ...
    # Shape: 26 * 3 = 78 variables
    
    def objective(vars):
        # Negative sum of radii (since we minimize)
        return -np.sum(vars[2::3])

    def constraints(vars):
        cons = []
        
        # Reshape to centers and radii
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # Boundary constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
        # Formatted as inequality g(vars) >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: v[0::3] - v[2::3]})          # x - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: 1 - (v[0::3] + v[2::3])})    # x + r <= 1
        cons.append({'type': 'ineq', 'fun': lambda v: v[1::3] - v[2::3]})          # y - r >= 0
        cons.append({'type': 'ineq', 'fun': lambda v: 1 - (v[1::3] + v[2::3])})    # y + r <= 1
        
        # Non-negative radii
        cons.append({'type': 'ineq', 'fun': lambda v: v[2::3]})

        # Non-overlap constraints: dist(i, j) >= r_i + r_j
        # dist^2 >= (r_i + r_j)^2
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        for i in range(n):
            for j in range(i + 1, n):
                def make_constraint(idx1, idx2):
                    def constraint_func(v):
                        xi, yi, ri = v[idx1*3], v[idx1*3+1], v[idx1*3+2]
                        xj, yj, rj = v[idx2*3], v[idx2*3+1], v[idx2*3+2]
                        return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                    return constraint_func
                
                cons.append({'type': 'ineq', 'fun': make_constraint(i, j)})
        
        return cons

    # Flatten initial state
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii

    # Bounds: radii >= 0, coords in [0, 1]
    bounds = []
    for i in range(n):
        bounds.extend([(0, 1), (0, 1), (0, 0.5)]) # r <= 0.5 loosely

    # Use SLSQP method
    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints(x0), # Note: constraints need to be callable or list
        options={'maxiter': 1000, 'ftol': 1e-9}
    )
    
    # If SLSQP fails to provide valid constraints list format dynamically, 
    # we might need to define constraints as a list of dicts without lambda closures or using a wrapper.
    # The lambda closures in the loop above capture `i` and `j` correctly in Python 3.
    # However, passing constraints to minimize usually expects a list of Constraint objects or dicts.
    # The 'fun' in dict must be a callable. The lambda captures the specific i,j.
    # But we passed `constraints(x0)` which returns the list.
    # Wait, constraints function depends on vars? No, the structure is static.
    # The functions inside depend on `i` and `j` from loop.
    
    # Re-evaluating constraint construction to be safe
    # We will reconstruct the list of constraint dicts properly.
    
    cons_list = []
    cons_list.append({'type': 'ineq', 'fun': lambda v: v[0::3] - v[2::3]})
    cons_list.append({'type': 'ineq', 'fun': lambda v: 1 - (v[0::3] + v[2::3])})
    cons_list.append({'type': 'ineq', 'fun': lambda v: v[1::3] - v[2::3]})
    cons_list.append({'type': 'ineq', 'fun': lambda v: 1 - (v[1::3] + v[2::3])})
    cons_list.append({'type': 'ineq', 'fun': lambda v: v[2::3]})

    for i in range(n):
        for j in range(i + 1, n):
            # We need to pass i and j to the function. 
            # Using a closure with default argument to capture value.
            def make_constraint(idx1, idx2):
                def constraint_func(v):
                    xi, yi, ri = v[idx1*3], v[idx1*3+1], v[idx1*3+2]
                    xj, yj, rj = v[idx2*3], v[idx2*3+1], v[idx2*3+2]
                    return (xi - xj)**2 + (yi - yj)**2 - (ri + rj)**2
                return constraint_func
            cons_list.append({'type': 'ineq', 'fun': make_constraint(i, j)})

    # Run optimizer
    # Reset x0 just in case
    x0 = np.zeros(3 * n)
    x0[0::3] = centers[:, 0]
    x0[1::3] = centers[:, 1]
    x0[2::3] = radii

    result = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons_list,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )

    # Extract results
    best_x = result.x
    final_centers = np.column_stack((best_x[0::3], best_x[1::3]))
    final_radii = best_x[2::3]
    final_sum = np.sum(final_radii)

    # Post-process: ensure radii are non-negative and valid
    # Sometimes optimizer might go slightly negative or violate due to tolerance
    # But constraints should handle it.
    # Let's clip just in case for safety before validation
    final_radii = np.maximum(final_radii, 0)
    
    # Re-calculate sum
    final_sum = np.sum(final_radii)

    return final_centers, final_radii, final_sum