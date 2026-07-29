# sol_000099 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 80fa60f2) state=29e9fd6e sum of radii=2.553987 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    """
    Packs 26 circles in a unit square [0,1]x[0,1] to maximize the sum of radii.
    
    Returns:
        centers: np.array of shape (26, 2)
        radii: np.array of shape (26,)
        sum_radii: float
    """
    n = 26
    
    # 1. Initialize centers and radii
    # We use a dense grid initialization to help the optimizer find a global-like optimum.
    # A 6x5 grid gives 30 points, we select 26.
    initial_points = []
    
    # Create a grid of points
    # 6 columns, 5 rows
    # Spacing roughly 1/7 to allow room for optimization
    x_vals = np.linspace(1/8, 7/8, 6)
    y_vals = np.linspace(1/8, 7/8, 5)
    
    for y in y_vals:
        for x in x_vals:
            initial_points.append([x, y])
            if len(initial_points) == n:
                break
        if len(initial_points) == n:
            break
            
    initial_centers = np.array(initial_points)
    initial_radii = np.ones(n) * 0.05  # Start with small valid radii
    
    # 2. Define variables for optimization
    # Format: [x1, y1, r1, x2, y2, r2, ..., x26, y26, r26]
    x0 = np.zeros(3 * n)
    for i in range(n):
        x0[3*i]   = initial_centers[i, 0]
        x0[3*i+1] = initial_centers[i, 1]
        x0[3*i+2] = initial_radii[i]
        
    # Bounds: x, y in [0, 1], r >= 0 (upper bound 0.5 is safe)
    bounds = []
    for _ in range(n):
        bounds.append((0, 1))   # x
        bounds.append((0, 1))   # y
        bounds.append((0, 0.5)) # r
        
    # 3. Objective function: Minimize negative sum of radii
    def objective(vars):
        radii = vars[2::3]
        return -np.sum(radii)
        
    # 4. Constraints
    def get_constraints(n):
        constraints_list = []
        
        # --- Boundary Constraints ---
        # x_i - r_i >= 0
        # 1 - (x_i + r_i) >= 0
        # y_i - r_i >= 0
        # 1 - (y_i + r_i) >= 0
        
        # We can vectorize these slightly or just return a large array
        # SLSQP accepts a vector of constraint values
        
        def boundary_constraints(vars):
            cons = np.zeros(4 * n)
            for i in range(n):
                idx = 3 * i
                x = vars[idx]
                y = vars[idx+1]
                r = vars[idx+2]
                
                c_idx = 4 * i
                cons[c_idx]   = x - r
                cons[c_idx+1] = 1.0 - (x + r)
                cons[c_idx+2] = y - r
                cons[c_idx+3] = 1.0 - (y + r)
            return cons

        constraints_list.append({
            'type': 'ineq',
            'fun': boundary_constraints,
            'jac': None # Let SLSQP approximate
        })

        # --- Non-overlap Constraints ---
        # (x_i - x_j)^2 + (y_i - y_j)^2 - (r_i + r_j)^2 >= 0
        
        def nonoverlap_constraints(vars):
            # Reshape to separate centers and radii
            # vars is 1D array of size 3n
            centers = vars.reshape(-1, 3)[:, :2] # (n, 2)
            radii = vars.reshape(-1, 3)[:, 2]    # (n,)
            
            # Compute pairwise squared distances
            # Broadcasting: (n, 1, 2) - (1, n, 2) -> (n, n, 2)
            diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=2)
            
            # Compute pairwise sum of radii squared
            r_sum = radii[:, np.newaxis] + radii[np.newaxis, :]
            r_sum_sq = r_sum**2
            
            # Constraint: dist_sq >= r_sum_sq
            # We only need the upper triangle (i < j)
            # np.triu_indices returns indices for upper triangle
            triu_idx = np.triu_indices(n, k=1)
            
            constraints_values = dist_sq[triu_idx] - r_sum_sq[triu_idx]
            return constraints_values

        constraints_list.append({
            'type': 'ineq',
            'fun': nonoverlap_constraints,
            'jac': None
        })
        
        return constraints_list

    constraints = get_constraints(n)

    # 5. Run Optimization
    # Method SLSQP handles bounds and non-linear constraints well
    res = minimize(
        objective, 
        x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints, 
        options={
            'maxiter': 2000, 
            'ftol': 1e-12,
            'disp': False
        }
    )
    
    # 6. Extract results
    if res.success:
        final_vars = res.x
    else:
        # If optimization didn't technically succeed, return the best found so far
        # (SLSQP usually returns a valid point even if it stops early)
        final_vars = res.x

    centers = final_vars.reshape(-1, 3)[:, :2]
    radii = final_vars.reshape(-1, 3)[:, 2]
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# Validation helper (read-only from prompt, kept here for context)
def validate_packing(centers, radii):
    """
    Validate that circles don't overlap and are inside the unit square
    """
    import numpy as np # Re-import just in case context is lost, though usually available
    n = centers.shape[0]

    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
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
