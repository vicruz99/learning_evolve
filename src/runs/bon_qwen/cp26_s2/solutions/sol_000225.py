# sol_000225 | problem=circle_packing_26 entrypoint=run_packing
# generation=0 parent=seed (state 96713eb2) state=bf659973 sum of radii=2.615061 correctness=1.0
# stdout(first 200): 
# NOTE: model code as-parsed; at eval time the harness also injects a preamble
#       (validator source + construction globals) via envs/<problem>.py.

import numpy as np
from scipy.optimize import minimize

def get_hex_grid(n):
    """
    Generates an initial configuration of n centers in a hexagonal grid pattern.
    """
    centers = []
    # Estimate spacing based on hexagonal density for n circles in unit area
    # Area of equilateral triangle of side d is (sqrt(3)/4) * d^2
    # Each circle occupies approx 2 * Area_triangle = (sqrt(3)/2) * d^2
    # d ~ sqrt(1 / (n * sqrt(3)/2))
    d = np.sqrt(1.0 / (n * np.sqrt(3) / 2.0))
    
    h = d * np.sqrt(3) / 2.0
    
    # We will fill rows
    row_y = 0
    # Start with some offset to center the grid
    x_offset = 0.0
    y_offset = 0.0
    
    # Simple hex grid generation
    # We try to fit rows
    row_idx = 0
    while len(centers) < n:
        # Alternate row lengths: 5, 6, 5, 6... roughly
        # Determine number of points in this row
        # Approx number of points per row ~ sqrt(2/3) * sqrt(n) * something
        # Let's just try to place as many as fit in x-dimension
        # Actually, for 26, let's hardcode a nice distribution or just place them
        # A 5x5 grid is 25. We need 26.
        # Hex grid with d ~ 0.2. 1/d = 5.
        # So 5 points per row fits well.
        
        # Points in this row
        # Odd rows (1, 3...) shifted by d/2
        shift = (d / 2.0) if (row_idx % 2 == 1) else 0.0
        
        # How many points fit in x?
        # x coordinates: shift, shift+d, shift+2d...
        # Max x should be <= 1 - margin.
        # Let's just generate points and clip/shift later or fit them
        
        # We want to cover the square.
        # Let's just place points with spacing d
        current_x = shift
        count = 0
        while current_x < 1.0 - d/2.0 and len(centers) < n:
            centers.append([current_x, row_y])
            current_x += d
            count += 1
        
        row_y += h
        row_idx += 1
        
    # If we have more than n (unlikely with the loop), trim
    centers = centers[:n]
    
    # Normalize/Shift to center in [0,1]x[0,1] if necessary, 
    # but random scaling during opt might fix it. 
    # Let's scale to fit tightly in [0.1, 0.9] to leave room for radii initially?
    # Actually, the optimizer can expand.
    # Let's just return them.
    
    return np.array(centers)

def run_packing() -> tuple[np.ndarray, np.ndarray, float]:
    n = 26
    num_vars = 3 * n  # x, y, r for each circle
    
    def objective(vars):
        # vars: [x1, y1, r1, x2, y2, r2, ...]
        radii = vars[2::3]
        # We want to maximize sum(radii), so minimize -sum(radii)
        return -np.sum(radii)

    def constraints(vars):
        # vars shape (3*n,)
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        
        # 1. Boundary constraints
        # x - r >= 0
        # 1 - x - r >= 0
        # y - r >= 0
        # 1 - y - r >= 0
        c_bounds = np.concatenate([
            x - r,
            1.0 - x - r,
            y - r,
            1.0 - y - r
        ])
        
        # 2. Non-overlap constraints
        # dist^2 >= (r_i + r_j)^2
        # dist^2 - (r_i + r_j)^2 >= 0
        
        # Vectorized distance squared calculation
        # x and y are arrays of size n
        # Create difference matrices
        # X_diff[i, j] = x[i] - x[j]
        X = x[:, np.newaxis] - x[np.newaxis, :]
        Y = y[:, np.newaxis] - y[np.newaxis, :]
        DistSq = X**2 + Y**2
        
        # Radius sum matrix
        R = r[:, np.newaxis] + r[np.newaxis, :]
        RSumSq = R**2
        
        # Constraint matrix
        C_matrix = DistSq - RSumSq
        
        # Extract upper triangle (i < j)
        # triu_indices gives indices for i < j? No, i <= j usually.
        # We need i < j.
        rows, cols = np.triu_indices(n, k=1)
        c_overlap = C_matrix[rows, cols]
        
        return np.concatenate([c_bounds, c_overlap])

    # Bounds for variables
    # x, y in [0, 1]
    # r in [0, 0.5] (since diameter <= 1)
    bounds = []
    for _ in range(n):
        bounds.append((0.0, 1.0)) # x
        bounds.append((0.0, 1.0)) # y
        bounds.append((0.0, 0.5)) # r
        
    best_vars = None
    best_obj = -np.inf
    
    # Try multiple initializations
    # 1. Hexagonal grid
    # 2. Random grids
    
    inits = []
    
    # Hex grid init
    hex_centers = get_hex_grid(n)
    # Scale hex_centers to be inside [0.1, 0.9] to ensure initial feasibility roughly
    # Actually, just place them.
    # Shift to center?
    # Let's just use the raw coordinates, they are in [0, ~1]
    # But maybe scale down slightly to ensure no initial overlaps if d is large?
    # get_hex_grid uses d ~ 0.2. Radius init 0.05. Safe.
    
    r_init = np.ones(n) * 0.05
    vars_hex = np.zeros(3 * n)
    vars_hex[0::3] = hex_centers[:, 0]
    vars_hex[1::3] = hex_centers[:, 1]
    vars_hex[2::3] = r_init
    inits.append(vars_hex)
    
    # Random inits
    for _ in range(5):
        x_rand = np.random.rand(n) * 0.8 + 0.1 # In [0.1, 0.9]
        y_rand = np.random.rand(n) * 0.8 + 0.1
        r_rand = np.ones(n) * 0.04
        
        vars_rand = np.zeros(3 * n)
        vars_rand[0::3] = x_rand
        vars_rand[1::3] = y_rand
        vars_rand[2::3] = r_rand
        inits.append(vars_rand)
        
    for i, x0 in enumerate(inits):
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                           constraints={'type': 'ineq', 'fun': constraints},
                           options={'maxiter': 2000, 'ftol': 1e-12, 'disp': False})
            
            # Check if valid (constraints satisfied)
            # Since SLSQP tries to satisfy constraints, we check fun value
            # But we also need to check the constraints explicitly for numerical safety
            c_val = constraints(res.x)
            min_c = np.min(c_val)
            
            # If feasible (or very close) and better objective
            if min_c >= -1e-6: # Tolerance for numerical error
                current_sum = -res.fun
                if current_sum > best_obj:
                    best_obj = current_sum
                    best_vars = res.x.copy()
        except Exception:
            pass
            
    if best_vars is None:
        # Fallback to a simple grid if optimization failed
        # 5x5 grid with one extra?
        # Just use the hex init result or last result
        # Re-run one last time with hex
        res = minimize(objective, vars_hex, method='SLSQP', bounds=bounds, 
                       constraints={'type': 'ineq', 'fun': constraints},
                       options={'maxiter': 2000})
        best_vars = res.x
        
    # Extract centers and radii
    centers = np.column_stack((best_vars[0::3], best_vars[1::3]))
    radii = best_vars[2::3]
    
    # Post-processing: ensure strict validity by shrinking radii slightly if needed
    # Check overlaps again
    # Validate
    # (Simulating the validation logic)
    valid = True
    for k in range(n):
        if np.isnan(centers[k]).any() or np.isnan(radii[k]):
            valid = False
            break
        if radii[k] < 0:
            valid = False
            break
        x, y = centers[k]
        r = radii[k]
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            # Shrink radius to fit boundary
            # Max possible radius at this center
            max_r_boundary = min(x, 1-x, y, 1-y)
            if r > max_r_boundary + 1e-9:
                radii[k] = max_r_boundary
    for k in range(n):
        for m in range(k + 1, n):
            dist = np.sqrt(np.sum((centers[k] - centers[m]) ** 2))
            if dist < radii[k] + radii[m] - 1e-9:
                # Overlap detected, shrink radii
                # Simple heuristic: average the overlap
                excess = (radii[k] + radii[m] - dist) / 2 + 1e-7
                radii[k] -= excess / 2
                radii[m] -= excess / 2
    
    # Re-check boundaries after shrinking
    for k in range(n):
        x, y = centers[k]
        r = radii[k]
        max_r = min(x, 1-x, y, 1-y)
        if r > max_r:
            radii[k] = max_r
            
    # Ensure non-negative
    radii = np.maximum(radii, 0.0)
    
    sum_radii = np.sum(radii)
    
    return centers, radii, sum_radii

# To run and print result
if __name__ == "__main__":
    c, r, s = run_packing()
    print(f"Sum of radii: {s}")
    # print(c)
    # print(r)
