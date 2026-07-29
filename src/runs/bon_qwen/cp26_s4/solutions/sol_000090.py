# sol_000090 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state f395aea4) state=317cf7b6 sum of radii=2.617322 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def calculate_objective(vars):
    """Calculate the negative sum of radii (to be minimized)."""
    radii = vars[52:]
    return -np.sum(radii)

def boundary_constraint(vars, n):
    """Ensure circles are inside the unit square [0, 1]x[0, 1]."""
    centers = vars[:52].reshape((n, 2))
    radii = vars[52:]
    constraints = []
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        constraints.append(x - r)
        constraints.append(y - r)
        constraints.append(1 - x - r)
        constraints.append(1 - y - r)
    return np.array(constraints)

def overlap_constraint(vars, n):
    """Ensure circles do not overlap."""
    centers = vars[:52].reshape((n, 2))
    radii = vars[52:]
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            constraints.append(dist - (radii[i] + radii[j]))
    return np.array(constraints)

def generate_hexagonal_init(n):
    """Generate an initial hexagonal packing configuration."""
    rows = 6
    cols_per_row = []
    for i in range(rows):
        cols_per_row.append(5 if i % 2 == 0 else 4)
    
    # Adjust to exactly n circles
    total = sum(cols_per_row)
    if total > n:
        while sum(cols_per_row) > n:
            # Remove from the end
            if cols_per_row[-1] > 1:
                cols_per_row[-1] -= 1
            else:
                cols_per_row.pop()
    elif total < n:
        while sum(cols_per_row) < n:
            # Add to the end or distribute
            if len(cols_per_row) < rows + 2:
                 cols_per_row.append(4)
            else:
                # Increase existing rows if possible
                cols_per_row[0] += 1
    
    # If we still don't have exactly n, just adjust the last row
    current_n = sum(cols_per_row)
    diff = n - current_n
    if diff != 0:
        if len(cols_per_row) > 0:
            cols_per_row[-1] += diff

    centers = []
    r_est = 0.095 # Initial estimate
    dy = np.sqrt(3) * r_est
    
    y = r_est
    for i, num_cols in enumerate(cols_per_row):
        dx = 2 * r_est
        x_start = r_est
        if i % 2 == 1:
            x_start += r_est
        
        x = x_start
        for _ in range(num_cols):
            centers.append([x, y])
            x += dx
        y += dy
        
    return np.array(centers)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    best_sum = 0
    best_result = None
    
    # Generate initial centers
    init_centers = generate_hexagonal_init(n)
    init_radii = np.full(n, 0.09)
    init_vars = np.hstack([init_centers.flatten(), init_radii])
    
    # Bounds: [0, 1] for coords, [0, 1] for radii
    bounds = []
    for _ in range(n):
        bounds.extend([(0.0, 1.0), (0.0, 1.0)])
    for _ in range(n):
        bounds.extend([(0.0, 1.0)])
        
    # Constraints
    cons = [
        {'type': 'ineq', 'fun': boundary_constraint, 'args': (n,)},
        {'type': 'ineq', 'fun': overlap_constraint, 'args': (n,)}
    ]
    
    # Run optimization
    for _ in range(5): # Multiple restarts
        # Add random noise to initialization
        current_vars = init_vars + np.random.normal(0, 0.01, size=len(init_vars))
        
        try:
            res = minimize(
                calculate_objective, 
                current_vars, 
                method='SLSQP', 
                bounds=bounds, 
                constraints=cons, 
                options={'maxiter': 2000, 'ftol': 1e-8}
            )
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum:
                    best_sum = current_sum
                    best_result = res.x
        except Exception:
            continue
            
    if best_result is not None:
        centers = best_result[:52].reshape((n, 2))
        radii = best_result[52:]
        # Clean up small negative radii due to numerical errors
        radii = np.maximum(radii, 0.0)
        return centers, radii, float(np.sum(radii))
    
    # Fallback if optimization fails
    return init_centers, np.full(n, 0.09), 26 * 0.09
