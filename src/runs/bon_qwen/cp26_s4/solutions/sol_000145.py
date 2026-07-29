# sol_000145 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 466799c7) state=d5cc4e03 sum of radii=2.593626 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing():
    n = 26

    # --- 1. Initialization ---
    # Initialize centers in a hexagonal lattice pattern
    centers = np.zeros((n, 2))
    radii = np.full(n, 0.09)  # Initial radii slightly below 0.1
    
    idx = 0
    # Hexagonal layout parameters
    y_pos = 0.1
    row_counts = [6, 5, 6, 5, 4]  # Sum = 26
    
    for i, count in enumerate(row_counts):
        x_start = 0.1
        if i % 2 == 1:
            x_start += 0.1  # Shift odd rows by half a diameter
            
        for j in range(count):
            if idx < n:
                centers[idx] = [x_start + j * 0.2, y_pos]
                idx += 1
        y_pos += 0.1 * np.sqrt(3)

    # --- 2. Optimization Setup ---
    # Objective: Minimize -sum(radii)
    # Constraints: Boundary containment and Non-overlap
    
    def objective(p):
        # Extract variables from the flat vector
        # p is ordered: [x_1, y_1, r_1, x_2, y_2, r_2, ...]
        r = p[2::3]
        return -np.sum(r)

    # Constraint: Boundary containment (r <= x <= 1-r, r <= y <= 1-r)
    # 4 constraints per circle: x-r >= 0, 1-x-r >= 0, y-r >= 0, 1-y-r >= 0
    def get_boundary_constraints(p):
        cons = []
        for i in range(n):
            x = p[3 * i]
            y = p[3 * i + 1]
            r = p[3 * i + 2]
            
            # x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx] - p[3*idx+2]})
            # 1 - x - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, idx=i: 1.0 - p[3*idx] - p[3*idx+2]})
            # y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, idx=i: p[3*idx+1] - p[3*idx+2]})
            # 1 - y - r >= 0
            cons.append({'type': 'ineq', 'fun': lambda p, idx=i: 1.0 - p[3*idx+1] - p[3*idx+2]})
        return cons

    # Constraint: Non-overlap (dist >= r_i + r_j)
    def get_overlap_constraints(p):
        cons = []
        for i in range(n):
            for j in range(i + 1, n):
                cons.append({'type': 'ineq', 'fun': lambda p, i=i, j=j: 
                    np.sum((p[3*i:3*i+2] - p[3*j:3*j+2])**2) - (p[3*i+2] + p[3*j+2])**2})
        return cons

    # Prepare initial vector
    p0 = np.zeros(3 * n)
    for i in range(n):
        p0[3*i] = centers[i, 0]
        p0[3*i+1] = centers[i, 1]
        p0[3*i+2] = radii[i]

    # Combine constraints
    constraints = get_boundary_constraints(p0) + get_overlap_constraints(p0)

    # Bounds: r_i >= 0
    bounds = [(0.0, 0.5) for _ in range(3*n)] # x, y in [0,1], r >= 0 (loosely handled by constraints)
    # More precise bounds for r
    for i in range(n):
        bounds[3*i+2] = (0.0, 0.5)
        bounds[3*i] = (0.0, 1.0)
        bounds[3*i+1] = (0.0, 1.0)

    # Run optimization
    # Using SLSQP for inequality constraints
    res = minimize(objective, p0, method='SLSQP', bounds=bounds, constraints=constraints, 
                   options={'maxiter': 1000, 'ftol': 1e-9})

    # --- 3. Extract Results ---
    final_p = res.x
    final_centers = np.zeros((n, 2))
    final_radii = np.zeros(n)
    
    for i in range(n):
        final_centers[i] = [final_p[3*i], final_p[3*i+1]]
        final_radii[i] = final_p[3*i+2]

    sum_radii = np.sum(final_radii)
    
    # Validate locally before returning
    if not validate_packing(final_centers, final_radii):
        # Fallback to initialization if optimization fails (should not happen)
        return centers, radii, np.sum(radii)

    return final_centers, final_radii, sum_radii

def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    n = centers.shape[0]
    if np.isnan(centers).any() or np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-9:
                return False
    return True

# Allow running the packing directly if this file is executed
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Optimization complete. Sum of radii: {s}")
    print(validate_packing(c, r))
