# sol_000274 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state c64acbd5) state=4fbf72ea sum of radii=2.458157 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

# Minimum gap to ensure numerical stability
EPS = 1e-7

def calculate_boundary_constr(vars, n):
    """Calculates constraints for boundary and radius positivity."""
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    
    # Centers must be at least (radius + EPS) away from boundaries
    cons_x1 = centers[:, 0] - radii - EPS
    cons_x2 = 1.0 - centers[:, 0] - radii - EPS
    cons_y1 = centers[:, 1] - radii - EPS
    cons_y2 = 1.0 - centers[:, 1] - radii - EPS
    
    # Radii must be non-negative (enforced by bounds, but included for safety)
    cons_r = radii - EPS
    
    return np.concatenate([cons_x1, cons_x2, cons_y1, cons_y2, cons_r])

def calculate_overlap_constr(vars, n):
    """Calculates non-overlap constraints for all circle pairs."""
    centers = vars[:2 * n].reshape(n, 2)
    radii = vars[2 * n:]
    
    constraints = []
    for i in range(n):
        for j in range(i + 1, n):
            dist_sq = np.sum((centers[i] - centers[j]) ** 2)
            r_sum = radii[i] + radii[j]
            # Distance between centers must be >= sum of radii + 2*EPS
            constraints.append(dist_sq - (r_sum + 2 * EPS) ** 2)
            
    return np.array(constraints)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    
    # 1. Initialization: Hexagonal Lattice
    # Initial radius chosen to be safely within bounds for a 5x5-like hex pattern
    r_init = 0.085
    centers = []
    
    y = r_init
    row_count = 0
    
    # Generate points until we have at least n
    while len(centers) < n:
        # Offset alternate rows
        if row_count % 2 == 0:
            x = r_init
        else:
            x = 2 * r_init
        
        # Place circles in the current row
        while x <= 1.0 - r_init:
            if len(centers) < n:
                centers.append([x, y])
            x += 2 * r_init
        y += np.sqrt(3) * r_init
        row_count += 1
        
    centers = np.array(centers[:n])
    radii = np.full(n, r_init)
    
    # Flatten variables: [x1, y1, ..., xn, yn, r1, ..., rn]
    x0 = np.concatenate([centers.flatten(), radii])
    
    # 2. Optimization
    # Bounds: x, y in [0, 1], r in [0, 0.5]
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # x_i
        bounds.append((0.0, 1.0)) # y_i
        bounds.append((0.0, 0.5)) # r_i
    
    def objective(vars):
        # Maximize sum of radii -> Minimize negative sum
        r = vars[2 * n:]
        return -np.sum(r)
    
    # Define constraints
    cons_boundary = {
        'type': 'ineq',
        'fun': lambda v: calculate_boundary_constr(v, n)
    }
    
    cons_overlap = {
        'type': 'ineq',
        'fun': lambda v: calculate_overlap_constr(v, n)
    }
    
    # Run optimizer
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                      constraints=[cons_boundary, cons_overlap],
                      options={'maxiter': 1000, 'ftol': 1e-9})
    
    if result.success:
        final_centers = result.x[:2 * n].reshape(n, 2)
        final_radii = result.x[2 * n:]
    else:
        # Fallback to initial configuration if optimization fails
        final_centers = centers
        final_radii = radii
    
    # Final validation and sum
    total_sum = np.sum(final_radii)
    return final_centers, final_radii, total_sum

# Verification (optional)
# if __name__ == "__main__":
#     c, r, s = run_packing()
#     print(f"Sum of radii: {s}")
