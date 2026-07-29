# sol_000233 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state dc099519) state=452c38a3 sum of radii=2.613548 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0 or np.isnan(radii[i]):
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-12 or x + r > 1 + 1e-12 or y - r < -1e-12 or y + r > 1 + 1e-12:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False
    return True

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n_circles = 26
    
    # Objective function: minimize negative sum of radii (maximize sum)
    def objective(vars):
        radii = vars[2*n_circles:]
        return -np.sum(radii)

    # Constraint: Non-overlap between circles
    def no_overlap(vars, i, j):
        x_i, y_i = vars[2*i], vars[2*i+1]
        x_j, y_j = vars[2*j], vars[2*j+1]
        r_i = vars[2*n_circles + i]
        r_j = vars[2*n_circles + j]
        dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
        return dist - r_i - r_j

    # Constraint: Boundary (inside unit square)
    def boundary_x(vars, i):
        return vars[2*i] - vars[2*n_circles + i]  # x - r >= 0
    
    def boundary_x_max(vars, i):
        return 1 - (vars[2*i] + vars[2*n_circles + i])  # 1 - (x + r) >= 0

    def boundary_y(vars, i):
        return vars[2*i+1] - vars[2*n_circles + i] # y - r >= 0

    def boundary_y_max(vars, i):
        return 1 - (vars[2*i+1] + vars[2*n_circles + i]) # 1 - (y + r) >= 0

    # Constraint: Radii non-negative
    def non_neg_radius(vars, i):
        return vars[2*n_circles + i]

    # Build constraints list
    constraints = []
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            constraints.append({'type': 'ineq', 'fun': lambda v, i=i, j=j: no_overlap(v, i, j)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_x(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_x_max(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_y(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: boundary_y_max(v, i)})
        constraints.append({'type': 'ineq', 'fun': lambda v, i=i: non_neg_radius(v, i)})

    # Bounds for variables
    bounds = [(0, 1)] * (2 * n_circles) + [(0, 0.5)] * n_circles

    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None

    # Multiple restarts to find global optimum
    n_restarts = 5
    for _ in range(n_restarts):
        # Initialize with a perturbed hexagonal-like pattern
        centers_init = np.zeros((n_circles, 2))
        radii_init = np.full(n_circles, 0.05) # Start small
        
        # Arrange in rows
        row_idx = 0
        col_idx = 0
        current_idx = 0
        
        # Try to distribute 26 circles in a roughly hexagonal layout
        # Rows of 5, 5, 5, 5, 4, 2
        row_counts = [5, 5, 5, 5, 4, 2]
        
        for r_idx, count in enumerate(row_counts):
            y = 0.15 + r_idx * 0.15
            # Shift x for alternating rows to mimic hexagonal packing
            x_start = 0.15 if r_idx % 2 == 0 else 0.25
            
            for c_idx in range(count):
                if current_idx >= n_circles:
                    break
                x = x_start + c_idx * 0.18
                # Add some randomness
                centers_init[current_idx] = [x + np.random.uniform(-0.02, 0.02),
                                             y + np.random.uniform(-0.02, 0.02)]
                current_idx += 1
        
        # Flatten variables
        x0 = np.concatenate([centers_init.flatten(), radii_init])
        
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-12})
            
            if res.success:
                current_sum = -res.fun
                if current_sum > best_sum_radii:
                    best_sum_radii = current_sum
                    best_centers = res.x[:2*n_circles].reshape(n_circles, 2)
                    best_radii = res.x[2*n_circles:]
        except Exception:
            continue

    # If optimization failed to find a valid solution better than initialization, return init
    if best_centers is None:
        best_centers = centers_init
        best_radii = radii_init
        best_sum_radii = np.sum(best_radii)

    # Final validation check (though optimization should ensure it)
    # If validation fails, clamp radii slightly to ensure validity
    if not validate_packing(best_centers, best_radii):
        # Fallback: reduce radii slightly
        scale = 0.99
        while not validate_packing(best_centers, best_radii):
            best_radii *= scale
            scale *= 0.95
            if scale < 0.1:
                break # Emergency break
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, float(best_sum_radii)

if __name__ == "__main__":
    centers, radii, sum_r = run_packing()
    print(f"Sum of radii: {sum_r}")
    print(f"Valid: {validate_packing(centers, radii)}")
