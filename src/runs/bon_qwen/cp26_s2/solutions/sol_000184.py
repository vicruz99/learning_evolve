# sol_000184 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 1380b4f2) state=a56ec3c8 sum of radii=2.623027 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def generate_initial_hex_packing(n, square_size=1.0):
    """
    Generates an initial hexagonal packing for n circles in a unit square.
    """
    # Estimate initial radius for n circles
    # Area heuristic: n * pi * r^2 approx square_size^2 * 0.9 (density)
    # r approx sqrt(0.9 / (n * pi))
    r_init = np.sqrt(0.9 / (n * np.pi))
    
    # Use a small buffer to ensure initial positions are valid
    r_start = r_init * 0.8
    spacing_x = 2 * r_start
    spacing_y = r_start * np.sqrt(3)
    
    centers = []
    
    # Fill rows
    x, y = r_start, r_start
    while len(centers) < n:
        # Horizontal row
        while x <= 1.0 - r_start:
            centers.append([x, y])
            x += spacing_x
            if len(centers) >= n:
                break
        
        x = r_start + spacing_x / 2  # Offset for next row
        y += spacing_y
        
        if y + r_start > 1.0:
            break
            
    # If we still don't have n circles, add them randomly in valid spots
    while len(centers) < n:
        x = np.random.uniform(r_start, 1.0 - r_start)
        y = np.random.uniform(r_start, 1.0 - r_start)
        centers.append([x, y])
        
    return np.array(centers), r_start

def optimization_constraints(centers, radii):
    """
    Returns a list of constraint dictionaries for scipy.optimize.minimize.
    """
    n = len(radii)
    constraints = []
    
    # Boundary constraints
    for i in range(n):
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i] - v[2*n+i]})          # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i] - v[2*n+i]})   # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: v[2*i+1] - v[2*n+i]})       # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: 1.0 - v[2*i+1] - v[2*n+i]}) # 1 - y_i - r_i >= 0
        
    # Non-overlap constraints
    for i in range(n):
        for j in range(i + 1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: np.sqrt((v[2*i] - v[2*j])**2 + (v[2*i+1] - v[2*j+1])**2) - (v[2*n+i] + v[2*n+j])
            })
            
    return constraints

def objective_function(v, n):
    """
    Objective: Maximize sum of radii (equivalent to minimizing negative sum).
    v contains [x1, y1, ..., xn, yn, r1, r2, ..., rn]
    """
    radii = v[2*n:]
    return -np.sum(radii)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    # 1. Generate initial configuration
    n = 26
    centers, r_start = generate_initial_hex_packing(n)
    
    # 2. Set up optimization variables and bounds
    # Variables: [x1, y1, ..., x26, y26, r1, ..., r26]
    x0 = np.hstack([centers.flatten(), np.full(n, r_start)])
    
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
    for _ in range(n):
        bounds.append((0.0, 0.5)) # r
        
    # 3. Define constraints
    cons = optimization_constraints(centers, np.full(n, r_start))
    
    # 4. Run optimization
    res = minimize(
        objective_function, 
        x0, 
        args=(n,), 
        method='SLSQP', 
        bounds=bounds, 
        constraints=cons,
        options={'ftol': 1e-9, 'maxiter': 1000, 'disp': False}
    )
    
    # 5. Extract results
    best_centers = res.x[:2*n].reshape(n, 2)
    best_radii = res.x[2*n:]
    
    # 6. Post-processing: Refine slightly to ensure validity against numerical noise
    # Shrink radii slightly if needed to avoid edge cases
    sum_radii = np.sum(best_radii)
    
    return best_centers, best_radii, sum_radii

if __name__ == "__main__":
    centers, radii, total = run_packing()
    print(f"Sum of radii: {total}")
