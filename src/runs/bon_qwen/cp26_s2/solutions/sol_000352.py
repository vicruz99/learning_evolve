# sol_000352 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state a8bfd9ed) state=0d5e2534 sum of radii=2.620084 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # --- 1. Initialization ---
    # We initialize with a grid layout to ensure a valid starting configuration.
    # A 6x5 grid gives 30 spots, we pick the first 26.
    # Spacing is chosen to allow initial radii of ~0.05 easily.
    
    initial_centers = []
    x_coords = [0.1 + i * 0.15 for i in range(6)] # 0.1, 0.25, 0.4, 0.55, 0.7, 0.85
    y_coords = [0.1 + j * 0.2 for j in range(5)]  # 0.1, 0.3, 0.5, 0.7, 0.9
    
    count = 0
    for x in x_coords:
        for y in y_coords:
            if count < n:
                initial_centers.append([x, y])
                count += 1
            else:
                break
        if count >= n:
            break
            
    initial_centers = np.array(initial_centers)
    initial_radii = np.full(n, 0.05)
    
    # --- 2. Variable Setup ---
    # Variables vector: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    # Total 78 variables.
    x0 = []
    for i in range(n):
        x0.append(initial_centers[i, 0])
        x0.append(initial_centers[i, 1])
        x0.append(initial_radii[i])
    x0 = np.array(x0)
    
    # Bounds for variables
    # x, y in [0, 1], r in [0, 1]
    bounds = [(0.0, 1.0)] * (3 * n)
    
    # --- 3. Constraints ---
    constraints = []
    
    # Boundary constraints:
    # x_i - r_i >= 0
    # 1 - x_i - r_i >= 0
    # y_i - r_i >= 0
    # 1 - y_i - r_i >= 0
    for i in range(n):
        idx = 3 * i
        # x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i] - v[3*i + 2]
        })
        # 1 - x - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i] - v[3*i + 2]
        })
        # y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: v[3*i + 1] - v[3*i + 2]
        })
        # 1 - y - r >= 0
        constraints.append({
            'type': 'ineq',
            'fun': lambda v, i=i: 1.0 - v[3*i + 1] - v[3*i + 2]
        })
        
    # Non-overlap constraints:
    # (x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2
    # => (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
    for i in range(n):
        for j in range(i + 1, n):
            idx_i = 3 * i
            idx_j = 3 * j
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, i=i, j=j: \
                    (v[3*i] - v[3*j])**2 + (v[3*i+1] - v[3*j+1])**2 - (v[3*i+2] + v[3*j+2])**2
            })
            
    # --- 4. Objective Function ---
    # Minimize -sum(radii)
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)
    
    # --- 5. Optimization ---
    # Use SLSQP which supports bounds and inequality constraints
    # maxiter set high to allow convergence
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
    )
    
    # --- 6. Extract Results ---
    final_vars = res.x
    centers = np.zeros((n, 2))
    radii = np.zeros(n)
    
    for i in range(n):
        centers[i, 0] = final_vars[3*i]
        centers[i, 1] = final_vars[3*i + 1]
        radii[i] = final_vars[3*i + 2]
        
    # Ensure radii are non-negative (solver might push to 0)
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Helper function to run and print result for verification if executed directly
if __name__ == "__main__":
    centers, radii, s = run_packing()
    print(f"Sum of radii: {s}")
    # Basic validation
    print(f"Valid: {validate_packing(centers, radii)}")
